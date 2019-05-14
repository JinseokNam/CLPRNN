import time
import random
import argparse
import numpy as np
from itertools import count
from collections import OrderedDict
from datetime import datetime
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from tensorboardX import SummaryWriter

from models import Reinforce, ActorCritic, Environment
from evals import (list2matrix,
                   example_f1_score, macro_f1, micro_f1,
                   precision_k, nDCG_k,
                   compute_tp_fp_fn, safe_div)
from data_loaders import DatasetFactory

dtype = torch.float
device = None


class ModelConfig(object):
    def __init__(self):
        self.nonlinearity = 'tanh'
        self.output_size = -1
        self.embedding_size = -1
        self.context_dim = -1
        self.ctx_proj_size = -1
        self.feature_dim = -1
        self.rnn_input_size = -1
        self.rnn_hidden_size = -1
        self.hidden_size = -1
        self.state_size = -1        # state_size == hidden_size
        self.bottleneck_size = -1
        self.use_bottleneck = False
        self.num_layers = -1
        self.max_trials = -1
        self.minibatch_size = -1
        self.STOP = -1
        self.entropy_penalty = -1
        self.value_weight = 0
        self.reward_function = 'exf1'
        self.dropout_prob = 0
        self.gamma = 1

    def load(self, arg):
        self.nonlinearity = arg['nonlinearity']
        self.output_size = arg['output_size']
        self.embedding_size = arg['embedding_size']
        self.context_dim = arg['context_dim']
        self.ctx_proj_size = arg['ctx_proj_size']
        self.feature_dim = arg['feature_dim']
        self.rnn_input_size = arg['rnn_input_size']
        self.rnn_hidden_size = arg['rnn_hidden_size']
        self.hidden_size = arg['hidden_size']
        self.state_size = arg['hidden_size']
        self.bottleneck_size = arg['bottleneck_size']
        self.use_bottleneck = arg['use_bottleneck']
        self.num_layers = arg['num_layers']
        self.max_trials = arg['max_trials']
        self.minibatch_size = arg['minibatch_size']
        self.STOP = arg['STOP']
        self.entropy_penalty = arg['entropy_penalty']
        self.reward_function = arg['reward_function']
        self.value_weight = arg['value_weight']
        self.dropout_prob = arg['dropout_prob']
        self.gamma = arg['gamma']


def load_data(dataset_name, batch_size=128, valid_size=0.1, shuffle=True, label_order='freq2rare', num_workers=1, cv_fold=0, seed=None):

    # Load data
    train_dataset = DatasetFactory.get_dataset(dataset_name, label_order=label_order, fold=cv_fold)
    test_dataset = DatasetFactory.get_dataset(dataset_name, train=False, label_order=label_order, fold=cv_fold,
                                              feature_mean=train_dataset.get_feature_mean(),
                                              feature_variance=train_dataset.get_feature_variance(),
                                              label_vocabs=train_dataset.get_label_vocabs())

    num_train = len(train_dataset)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))

    if shuffle:
        np.random.seed(seed)
        np.random.shuffle(indices)

    train_idx, sub_train_idx, valid_idx = indices[split:], indices[-split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    sub_train_sampler = SubsetRandomSampler(sub_train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    def sp_collate_fn(data):
        instances, labels = zip(*data)
        batch_size = len(instances)
        I = []
        J = []
        V = []
        for index in range(len(instances)):
            I.append(instances[index][0] + index)
            J.append(instances[index][1])
            V.append(instances[index][2])

        I = np.concatenate(I)
        J = np.concatenate(J)
        V = np.concatenate(V)

        I = np.vstack([I, J])

        instances = (I, V)

        # Merge labels (from tuple of 1D tensor to 2D tensor).
        labels = list(labels)

        return instances, labels

    def non_sp_collate_fn(data):
        instances, labels = zip(*data)
        batch_size = len(instances)
        instances = np.stack(instances, 0)

        # Merge labels (from tuple of 1D tensor to 2D tensor).
        labels = list(labels)

        return instances, labels

    if dataset_name == 'mediamill':
        collate_fn = non_sp_collate_fn
    else:
        collate_fn = sp_collate_fn

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, sampler=train_sampler,
        num_workers=num_workers, collate_fn=collate_fn, drop_last=False
    )
    sub_train_loader = DataLoader(
        train_dataset, batch_size=batch_size, sampler=sub_train_sampler,
        num_workers=num_workers, collate_fn=collate_fn, drop_last=False
    )
    valid_loader = DataLoader(
        train_dataset, batch_size=batch_size, sampler=valid_sampler,
        num_workers=num_workers, collate_fn=collate_fn, drop_last=False
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size,
        num_workers=num_workers, collate_fn=collate_fn, drop_last=False
    )

    return train_loader, sub_train_loader, valid_loader, test_loader


def prepare_exp(dataset_name, max_epoch, learning_rate, weight_decay, batch_size, embedding_size, rnn_input_size, rnn_hidden_size, hidden_size,
        bottleneck_size, nonlinear_func='tanh', dropout_prob=0., num_layers=3,
        entropy_penalty=0., value_weight=0.5, gamma=1., alpha=0.9, label_order='freq2rare', reward_function='exf1', cv_fold=0, seed=None):
    opt_config = {'max_epoch': max_epoch,
                  'learning_rate': learning_rate,
                  'weight_decay': weight_decay}

    valid_size = 0.05 if dataset_name == 'amazoncat' else 0.1
    data_config = {'batch_size': batch_size,
                   'valid_size': valid_size,
                   'shuffle': True,
                   'label_order': label_order,
                   'cv_fold': cv_fold,
                   'num_workers': 1}
    train_loader, sub_train_loader, valid_loader, test_loader = load_data(dataset_name, seed=seed, **data_config)

    NUM_ACTIONS = train_loader.dataset.get_num_labels()
    DIM_FEAT = train_loader.dataset.get_feature_dim()
    label_card = train_loader.dataset.get_label_cardinality()
    max_labelset_size = train_loader.dataset.get_max_labelset_size()
    EOS_ID = train_loader.dataset.get_stop_label_id()

    model_config = ModelConfig()
    model_config.load({'minibatch_size': data_config['batch_size'],
                       'output_size': NUM_ACTIONS,
                       'embedding_size': embedding_size,
                       'context_dim': DIM_FEAT,
                       'ctx_proj_size': embedding_size,
                       'feature_dim': DIM_FEAT,
                       'rnn_input_size': rnn_input_size,
                       'rnn_hidden_size': rnn_hidden_size,
                       'hidden_size': hidden_size,
                       'use_bottleneck': bottleneck_size > 0,
                       'bottleneck_size': bottleneck_size,
                       'dropout_prob': dropout_prob,
                       'num_layers': num_layers,
                       'STOP': EOS_ID,
                       'max_trials': int(max_labelset_size*1.5),
                       'entropy_penalty': entropy_penalty,
                       'reward_function': reward_function,
                       'nonlinearity': nonlinear_func,
                       'value_weight': value_weight,
                       'gamma': gamma,
                       'alpha': alpha})

    data_loaders = [train_loader, sub_train_loader, valid_loader, test_loader]
    configs = [opt_config, data_config, model_config]

    return data_loaders, configs


def load_evaluation_functions():
    # 'subset accuracy': subset_accuracy,
    bipartition_evaluation_functions = {
        'example f1 score': example_f1_score,
        'macro f1 score': macro_f1,
        'micro f1 score': micro_f1
    }

    ranking_evaluation_functions = {
        'precision_k': precision_k,
        'nDCG_k': nDCG_k
    }
    # 'generated set size': compute_gen_set_size,

    return (bipartition_evaluation_functions, ranking_evaluation_functions)


def create_dropout_mask(dropout_prob, batch_size, x_size, y_size):
    keep_prob = 1-dropout_prob
    mask_x = torch.empty(batch_size, x_size, device=device, requires_grad=False).uniform_(0, 1)
    mask_x.bernoulli_(keep_prob) / (1./keep_prob)
    mask_y = torch.empty(batch_size, y_size, device=device, requires_grad=False).uniform_(0, 1)
    mask_y.bernoulli_(keep_prob) / (1./keep_prob)

    return (mask_x, mask_y)


def convert_labelset2seq(targets, predicted_labels, EOS_ID):

    batch_size = len(targets)
    target_length = np.array(list(map(len, targets)))
    max_length = np.max(target_length) + 1

    targets_ = np.zeros((max_length, batch_size), dtype=int)

    for i in range(len(targets)):
        trg = set(targets[i])
        pred = predicted_labels[i]
        pred_set = set(pred)
        true_pos = trg.intersection(pred_set)
        false_neg = list(trg.difference(pred_set))

        new_target = list(OrderedDict([(p, 1) for p in pred if p in true_pos]).keys())
        random.shuffle(false_neg)
        new_target += false_neg

        assert len(new_target) == len(targets[i]), '\n{}\n{}\n{}\n{}\n{}'.format(new_target, targets[i], pred_set, true_pos, false_neg)

        targets_[:len(new_target), i] = new_target
        targets_[len(new_target), i] = EOS_ID      # add the stop action back

        del trg, pred_set, true_pos, false_neg, new_target

    return targets_


def print_param_norm(parameters, writer, output_str, num_param_updates):
    total_norm = 0
    for p in list(filter(lambda p: p.grad is not None, parameters)):
        param_norm = p.grad.data.norm(2)
        total_norm += param_norm.item() ** 2

    total_norm = total_norm ** (1. / 2)

    writer.add_scalar(output_str, total_norm, num_param_updates)


def prepare_minibatch(data, targets, feature_dim, is_sparse_data=True, drop_EOS=True):
    if is_sparse_data:
        batch_size = len(targets)
        I, V = data
        data = torch.sparse.FloatTensor(torch.LongTensor(I),
                torch.FloatTensor(V), torch.Size([batch_size, feature_dim]))

        if device.type == 'cuda':
            data = data.cuda()

        data = data.to_dense()
    else:
        data = torch.FloatTensor(data)
        if device.type == 'cuda':
            data = data.cuda()


    if drop_EOS:
        for i, trg in enumerate(targets):
            # remove stop label from label sets
            assert len(trg) > 0
            targets[i] = trg[:-1]

    # update batch size
    non_zero_target = np.array(list(map(len, targets))) > 0
    data = data[non_zero_target.nonzero()]
    max_target_length = 0
    targets_ = []
    for trg in targets:
        if len(trg) == 0:
            continue
        if len(trg) > max_target_length:
            max_target_length = len(trg)
        targets_.append(trg)

    targets = targets_

    return (data, targets)


def print_result_summary(results, writer, dataset, epoch):
    for split_name, scores in results.items():
        for func_name, func_val in scores.items():
            eval_result = scores[func_name]
            if np.ndim(eval_result) == 0:
                writer.add_scalar('{}/{} on {}'.format(dataset, func_name, split_name), eval_result, epoch)
            elif np.ndim(eval_result) == 1:
                for k in range(len(eval_result)):
                    writer.add_scalar('{}/{}@{} on {}'.format(dataset, func_name, 2*k+1, split_name), eval_result[k], epoch)
            else:
                raise ValueError('Unknown return values!')


def main(dataset, pretrain_max_epoch, max_epoch, learning_rate, weight_decay, max_pretrain_grad_norm, max_grad_norm,
         batch_size, embedding_size, rnn_input_size, rnn_hidden_size, hidden_size, bottleneck_size, entropy_penalty, gamma, alpha,
         nonlinear_func='tanh', value_weight=0.5, reward_function='exf1', label_order='freq2rare', input_dropout_prob=0.2, dropout_prob=0.5, num_layers=1, cv_fold=0, seed=None, fixed_label_seq_pretrain=False):

    data_loaders, configs = prepare_exp(dataset, max_epoch, learning_rate, weight_decay, batch_size,
            embedding_size, rnn_input_size, rnn_hidden_size, hidden_size, bottleneck_size, nonlinear_func=nonlinear_func, dropout_prob=dropout_prob, num_layers=num_layers, label_order=label_order,
            entropy_penalty=entropy_penalty, value_weight=value_weight, reward_function=reward_function, gamma=gamma, alpha=alpha, cv_fold=cv_fold, seed=seed)

    train_loader, sub_train_loader, valid_loader, test_loader = data_loaders
    opt_config, data_config, model_config = configs

    BOS_ID = train_loader.dataset.get_start_label_id()
    EOS_ID = train_loader.dataset.get_stop_label_id()
    is_sparse_data = train_loader.dataset.is_sparse_dataset()

    criterion = nn.NLLLoss(ignore_index=0, reduction='none')
    model = ActorCritic(model_config)
    if device.type == 'cuda':
        model = model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=opt_config['learning_rate'], weight_decay=weight_decay)
    env = Environment(model_config)
    bipartition_eval_functions, ranking_evaluation_functions = load_evaluation_functions()

    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    model_arch_info = 'emb_{}_rnn_{}_hid_{}_bot_{}_inpdp_{}_dp_{}_{}'.format(embedding_size, rnn_hidden_size, hidden_size, bottleneck_size, input_dropout_prob, dropout_prob, nonlinear_func)
    rl_info = 'alpha_{}_gamma_{}_vw_{}_reward_{}_ent_{}'.format(alpha, gamma, value_weight, reward_function, entropy_penalty)
    optim_info = 'lr_{}_decay_{}_norm_{}-{}_bs_{}_epoch_{}-{}_fold_{}'.format(learning_rate, weight_decay, max_pretrain_grad_norm, max_grad_norm, batch_size, pretrain_max_epoch, max_epoch, cv_fold)

    if fixed_label_seq_pretrain and max_epoch == 0:
        # baseline models
        summary_comment = '_'.join([current_time, 'baseline', label_order, model_arch_info, optim_info])
    else:
        summary_comment = '_'.join([current_time, 'proposed', model_arch_info, rl_info, optim_info])

    summary_log_dir = os.path.join('runs', dataset, summary_comment)
    bipartition_model_save_path = os.path.join('models', dataset,  summary_comment + '_bipartition.pth')
    ranking_model_save_path = os.path.join('models', dataset,  summary_comment + '_ranking.pth')

    writer = SummaryWriter(log_dir=summary_log_dir)

    n_batches = 0
    for batch_idx, (data, targets) in enumerate(train_loader):
        n_batches += 1

    num_param_updates = 0
    best_bipartition_valid_score = -np.inf
    best_ranking_valid_score = -np.inf
    max_epoch = opt_config['max_epoch']

    input_dropout = nn.Dropout(p=input_dropout_prob)

    # pretrain or only supervised learning with a fixed label ordering
    for epoch in range(pretrain_max_epoch):

        print('==== {} ===='.format(epoch))
        avg_rewards = []
        for batch_idx, (data, targets) in enumerate(train_loader):

            data, targets = prepare_minibatch(data, targets, train_loader.dataset.get_feature_dim(), is_sparse_data, drop_EOS=label_order == 'mblp')
            batch_size = len(targets)
            assert data.shape[0] == batch_size, '{}\t{}'.format(data.shape[0], batch_size)

            data = input_dropout(data)

            if label_order != 'mblp':
                target_length = np.array(list(map(len, targets)))
                max_length = int(np.max(target_length))
                targets_ = np.zeros((max_length, batch_size), dtype=np.int64)

                for i in range(batch_size):
                    targets_[:len(targets[i]), i] = targets[i]

                targets = torch.tensor(targets_, dtype=torch.int64, device=device, requires_grad=False)
            else:
                max_target_length = np.max(np.array(list(map(len, targets))))
                max_sampling_steps = int(max_target_length * 1.5)

                env.clear_episode_temp_data()
                gen_actions_per_episode = []
                rewards_per_episode = []

                model = model.eval()
                prev_states = model.init_hidden(data, device)
                prev_actions = torch.tensor([BOS_ID] * batch_size, dtype=torch.int64, device=device)

                for t in range(max_sampling_steps):  # no infinite loop while learning

                    model_outputs, states = model(data, prev_actions, prev_states, state_value_grad=False)
                    gen_actions, _, done = env.step(model_outputs)

                    gen_actions_per_episode.append(gen_actions.data.cpu().numpy())

                    if done:
                        break

                    prev_actions = gen_actions
                    prev_states = states

                # gen_actions_per_episode: (batch_size, max_trials) # cols can be smaller.
                gen_actions_per_episode = np.array(gen_actions_per_episode).T

                # sort labels according to model predictions
                targets_ = convert_labelset2seq(targets, gen_actions_per_episode, EOS_ID)
                targets = torch.tensor(targets_, dtype=torch.int64, device=device, requires_grad=False)

                del gen_actions_per_episode

            model = model.train()
            prev_states = model.init_hidden(data, device)
            prev_actions = torch.tensor([BOS_ID] * batch_size, dtype=torch.int64, device=device, requires_grad=False)
            dropout_masks = create_dropout_mask(model_config.dropout_prob, batch_size, model_config.embedding_size * 2, model_config.rnn_hidden_size)

            losses = []
            for t in range(targets.size(0)):  # no infinite loop while learning
                model_outputs, states = model(data, prev_actions, prev_states, dropout_masks=dropout_masks, state_value_grad=False)

                logits = model_outputs[0]
                log_probs = F.log_softmax(logits, dim=-1)
                target_t = targets[t]

                losses.append(criterion(log_probs, target_t))

                prev_actions = target_t
                prev_states = states

            # loss: (seq_len, batch_size)
            loss = torch.stack(losses, dim=0)
            loss = torch.sum(loss, dim=0).mean()

            optimizer.zero_grad()
            loss.backward()

            output_str = '{}/Before gradient norm'.format(dataset)
            print_param_norm(model.parameters(), writer, output_str, num_param_updates)

            # torch.nn.utils.clip_grad_value_(model.parameters(), 100)
            if max_pretrain_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_pretrain_grad_norm)

            output_str = '{}/After gradient norm'.format(dataset)
            print_param_norm(model.parameters(), writer, output_str, num_param_updates)

            optimizer.step()

            num_param_updates += 1

        results = evaluation(OrderedDict([('sub_train', sub_train_loader), ('valid', valid_loader), ('test', test_loader)]),
                             model, env, bipartition_eval_functions, ranking_evaluation_functions, model_config.max_trials)

        print_result_summary(results, writer, dataset, epoch)

        for split_name, scores in results.items():
            if split_name is 'valid':
                if scores['example f1 score'] > best_bipartition_valid_score:
                    best_bipartition_valid_scores = scores['example f1 score']
                    save_model(epoch, model, optimizer, bipartition_model_save_path)

                if scores['nDCG_k'][-1] > best_ranking_valid_score:
                    best_ranking_valid_scores = scores['nDCG_k'][-1]
                    save_model(epoch, model, optimizer, ranking_model_save_path)


    def update_alpha(epoch, xlimit=6, alpha_max=1):
        updated_alpha = 1/(1+float(np.exp(xlimit - 2*xlimit/float(max_epoch)*epoch)))
        updated_alpha = min(updated_alpha, alpha_max)
        return updated_alpha

    del optimizer

    # joint learning
    rl_optimizer = optim.Adam(model.parameters(), lr=opt_config['learning_rate'], weight_decay=weight_decay)
    for epoch in range(max_epoch):
        if alpha == 'auto':
            alpha_e = update_alpha(epoch)
        else:
            assert float(alpha) >= 0 and float(alpha) <= 1
            alpha_e = float(alpha)

        print('==== {} ===='.format(epoch + pretrain_max_epoch))
        avg_rewards = []
        for batch_idx, (data, targets) in enumerate(train_loader):

            model = model.train()
            data, targets = prepare_minibatch(data, targets, train_loader.dataset.get_feature_dim(), is_sparse_data)
            batch_size = len(targets)
            assert data.shape[0] == batch_size, '{}\t{}'.format(data.shape[0], batch_size)

            data = input_dropout(data)

            dropout_masks = create_dropout_mask(model_config.dropout_prob, batch_size, model_config.embedding_size * 2, model_config.rnn_hidden_size)
            prev_states = model.init_hidden(data, device)
            prev_actions = torch.tensor([BOS_ID] * batch_size, dtype=torch.int64, device=device, requires_grad=False)

            max_target_length = np.max(np.array(list(map(len, targets))))
            max_sampling_steps = int(max_target_length * 1.5)

            env.clear_episode_temp_data()
            gen_actions_per_episode = []
            rewards_per_episode = []

            for t in range(max_sampling_steps):  # no infinite loop while learning

                model_outputs, states = model(data, prev_actions, prev_states, dropout_masks=dropout_masks)
                gen_actions, rewards, done = env.step(model_outputs, targets)

                gen_actions_per_episode.append(gen_actions.data.cpu().numpy())
                rewards_per_episode.append(rewards)

                if done:
                    break

                prev_actions = gen_actions
                prev_states = states

            num_non_empty = np.array([len(t) > 0 for t in targets]).sum()
            r = np.stack(rewards_per_episode, axis=1).sum(1).sum() / num_non_empty
            avg_rewards.append(r)

            ps_loss, adv_collection = calculate_loss(env, model_config)
            writer.add_scalar('{}/avg_advantages'.format(dataset), adv_collection.mean().data.cpu().numpy(), num_param_updates)

            # gen_actions_per_episode: (batch_size, max_trials) # cols can be smaller.
            gen_actions_per_episode = np.array(gen_actions_per_episode).T

            # sort labels according to model predictions
            targets_ = convert_labelset2seq(targets, gen_actions_per_episode, EOS_ID)
            targets = torch.tensor(targets_, dtype=torch.int64, device=device, requires_grad=False)

            del gen_actions_per_episode

            prev_states = model.init_hidden(data, device)
            prev_actions = torch.tensor([BOS_ID] * batch_size, dtype=torch.int64, device=device, requires_grad=False)
            dropout_masks = create_dropout_mask(model_config.dropout_prob, batch_size, model_config.embedding_size * 2, model_config.rnn_hidden_size)

            losses = []
            for t in range(targets.size(0)):  # no infinite loop while learning
                model_outputs, states = model(data, prev_actions, prev_states, dropout_masks=dropout_masks, state_value_grad=False)
                logits = model_outputs[0]
                log_probs = F.log_softmax(logits, dim=-1)
                target_t = targets[t]

                losses.append(criterion(log_probs, target_t))

                prev_actions = target_t
                prev_states = states

            # loss: (seq_len, batch_size)
            sup_loss = torch.stack(losses, dim=0).sum(0)

            loss = alpha_e * ps_loss + (1 - alpha_e) * sup_loss
            loss = loss.mean()

            rl_optimizer.zero_grad()
            loss.backward()

            output_str = '{}/Before gradient norm'.format(dataset)
            print_param_norm(model.parameters(), writer, output_str, num_param_updates)

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

            output_str = '{}/After gradient norm'.format(dataset)
            print_param_norm(model.parameters(), writer, output_str, num_param_updates)

            rl_optimizer.step()

            num_param_updates += 1

        results = evaluation(OrderedDict([('sub_train', sub_train_loader), ('valid', valid_loader), ('test', test_loader)]),
                             model, env, bipartition_eval_functions, ranking_evaluation_functions, model_config.max_trials)

        print_result_summary(results, writer, dataset, epoch + pretrain_max_epoch)

        for split_name, scores in results.items():
            if split_name is 'valid':
                if scores['example f1 score'] > best_bipartition_valid_score:
                    best_bipartition_valid_scores = scores['example f1 score']
                    save_model(epoch+pretrain_max_epoch, model, rl_optimizer, bipartition_model_save_path)

                if scores['nDCG_k'][-1] > best_ranking_valid_score:
                    best_ranking_valid_scores = scores['nDCG_k'][-1]
                    save_model(epoch+pretrain_max_epoch, model, rl_optimizer, ranking_model_save_path)

    writer.close()


def calculate_loss(env, model_config):
    policy_losses = []
    value_losses = []
    neg_entropy_losses = []
    advantages_collection = []
    masks = []
    for log_probs, values, entropy, mask, rewards in zip(*env.get_episode_results()):
        if device.type == 'cuda':
            rewards = rewards.cuda()
        advantages = (rewards - values.data).detach()
        advantages_collection.append(advantages)

        policy_losses.append(-log_probs * advantages)
        value_losses.append(F.smooth_l1_loss(values, rewards, reduction='none'))
        # entropy maximization for action space exploration
        neg_entropy_losses.append(-entropy)
        masks.append(mask)

    policy_loss = torch.stack(policy_losses)
    value_loss = torch.stack(value_losses)
    neg_entropy = torch.stack(neg_entropy_losses)
    masks = torch.tensor(np.stack(masks, axis=0), dtype=torch.float)
    if device.type == 'cuda':
        masks = masks.cuda()

    loss = policy_loss + model_config.value_weight * value_loss + model_config.entropy_penalty * neg_entropy
    loss = (loss * masks).sum(0)

    advantages = torch.stack(advantages_collection) * masks
    avg_advantages = torch.sum(advantages, dim=0) / torch.sum(masks, dim=0)

    # delete or reset temporary variables
    env.clear_episode_temp_data()

    return loss, avg_advantages


def predict(model, data_loader, env, max_trials, min_trials=-1):
    all_targets = []
    all_predictions = []

    BOS_ID = data_loader.dataset.get_start_label_id()
    is_sparse_data = data_loader.dataset.is_sparse_dataset()

    model = model.eval()

    for batch_idx, (data, targets) in enumerate(data_loader):
        data, targets = prepare_minibatch(data, targets, data_loader.dataset.get_feature_dim(), is_sparse_data)
        batch_size = data.size(0)

        prev_states = model.init_hidden(data, device)
        prev_actions = torch.tensor([BOS_ID] * batch_size, dtype=torch.int64, device=device)

        gen_actions_per_episode = []

        for t in range(max_trials):

            model_outputs, states = model(data, prev_actions, prev_states)
            no_stop = min_trials > t
            gen_actions, _, done = env.step(model_outputs, stochastic=False, no_stop_action=no_stop)

            gen_actions_per_episode.append(gen_actions.data.cpu().numpy())

            if done:
                break

            prev_actions = gen_actions
            prev_states = states

        all_targets += targets

        # convert generated actions into list of list of integers
        preds_ = np.split(np.stack(gen_actions_per_episode, axis=1), batch_size)
        preds = [None] * batch_size
        for idx, pred in enumerate(preds_):
            pred = pred[0]
            pp = np.argwhere(pred == env.STOP)
            if pp.shape[0] > 0:
                preds[idx] = list(pred[:pp[0,0]])
            else:
                preds[idx] = list(pred)

        all_predictions += preds

        # delete or reset temporary variables
        env.clear_episode_temp_data()

    return all_predictions, all_targets


def evaluation(data_loaders, model, env, bipartition_eval_functions, ranking_eval_functions, max_trials, min_trials=-1):
    assert min_trials != 0

    eval_scores = OrderedDict()
    for split_name in data_loaders.keys():
        eval_scores[split_name] = dict()
        for eval_func_name in bipartition_eval_functions.keys():
            eval_scores[split_name][eval_func_name] = None
        for eval_func_name in ranking_eval_functions.keys():
            eval_scores[split_name][eval_func_name] = None

    K = 5       # minimum size of label set prediction

    for split_name, data_loader in data_loaders.items():

        """ Bipartition evaluation measures """
        BOS_ID = data_loader.dataset.get_start_label_id()
        EOS_ID = data_loader.dataset.get_stop_label_id()

        predictions, targets = predict(model, data_loader, env, max_trials)

        targets = list2matrix(targets, EOS_ID, n_labels=env.output_size-2)
        predictions = list2matrix(predictions, EOS_ID, n_labels=env.output_size-2)

        pred_stats = {
            'predictions': predictions,
            'targets': targets,
        }

        # example-based bipartition measures
        exf1_start_time = time.time()
        tp, fp, fn = compute_tp_fp_fn(predictions, targets, axis=1)
        exf1 = safe_div(2*tp, 2*tp + fp + fn)
        eval_scores[split_name]['example f1 score'] = np.mean(exf1)

        # label-based bipartition measures
        tp, fp, fn = compute_tp_fp_fn(predictions, targets, axis=0)
        maf1 = np.mean(safe_div(2*tp, 2*tp + fp + fn))
        mif1 = 2*np.sum(tp) / \
            float(2*np.sum(tp) + np.sum(fp) + np.sum(fn))
        eval_scores[split_name]['macro f1 score'] = maf1
        eval_scores[split_name]['micro f1 score'] = mif1

        """ Ranking evaluation measures """
        predictions, targets = predict(model, data_loader, env, max_trials, min_trials=K)

        # convert label sequences into prediction scores
        predictions = list2matrix(predictions, EOS_ID, n_labels=env.output_size-2, to_score=True)
        targets = list2matrix(targets, EOS_ID, n_labels=env.output_size-2)

        pred_stats = {
            'predictions': predictions,
            'targets': targets,
            'K': K
        }

        for func_name, func in ranking_eval_functions.items():
            eval_scores[split_name][func_name] = func(**pred_stats)

    return eval_scores


def save_model(epoch, model, optim, filename):
    head, tail = os.path.split(filename)
    if not os.path.exists(head):
        os.makedirs(head)

    torch.save({'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optim.state_dict()}, filename)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch REINFORCE example')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='discount factor (default: 0.99)')
    parser.add_argument('--alpha', type=str, default='0.9',
                        help='weighting factor (default: 0.9)')
    parser.add_argument('--entropy-penalty', type=float, default=0.01,
                        help='entropy penalty (default: 0.01)')
    parser.add_argument('--embedding-size', type=int, default=256,
                        help='label embedding size (default: 256)')
    parser.add_argument('--rnn-input-size', type=int, default=256,
                        help='RNN input size (default: 256)')
    parser.add_argument('--rnn-hidden-size', type=int, default=1024,
                        help='RNN hidden size (default: 1024)')
    parser.add_argument('--hidden-size', type=int, default=512,
                        help='hidden layer size (default: 512)')
    parser.add_argument('--bottleneck-size', type=int, default=0,
                        help='bottleneck layer size (default: 0, i.e. no bottleneck layer')
    parser.add_argument('--minibatch-size', type=int, default=128,
                        help='Number of examples in a minibatch (default: 128)')
    parser.add_argument('--dropout-prob', type=float, default=0.0,
                        help='Dropout probability (default: 0.0)')
    parser.add_argument('--input-dropout-prob', type=float, default=0.0,
                        help='Dropout probability on input features (default: 0.0)')
    parser.add_argument('--num-layers', type=int, default=1,
                        help='Number of hidden layers (default: 1)')
    parser.add_argument('--pretrain-max-epoch', type=int, default=10,
                        help='Number of iterations (default: 10)')
    parser.add_argument('--max-epoch', type=int, default=100,
                        help='Number of iterations (default: 100)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate of Adam (default: 0.001)')
    parser.add_argument('--weight-decay', type=float, default=0.0,
                        help='Weight decay (default: 0.0)')
    parser.add_argument('--pretrain-grad-norm', type=float, default=0.00,
                        help='Maximum norm of the gradients during supervised only training (default: 0.00)')
    parser.add_argument('--grad-norm', type=float, default=0.00,
                        help='Maximum norm of the gradients (default: 0.00)')
    parser.add_argument('--seed', type=int, default=543, metavar='N',
                        help='random seed (default: 543)')
    parser.add_argument('--value-weight', type=float, default=1.0)
    parser.add_argument('--reward-function', type=str, choices=['exf1', 'ndcg1', 'ndcg3', 'ndcg5'], default='exf1')
    parser.add_argument('--use-cuda', action='store_true')
    parser.add_argument('--nonlinear-func', type=str, choices=['relu', 'tanh'], default='tanh', help='Type of nonlinear function')
    parser.add_argument('--dataset', type=str, choices=['mediamill', 'delicious', 'eurlex', 'rcv1', 'bibtex', 'wiki10', 'amazoncat'], required=True, help='Dataset')
    parser.add_argument('--label-order', type=str, choices=['mblp', 'freq2rare', 'rare2freq', 'same', 'fixed-random', 'always-random'], default='mblp')
    parser.add_argument('--fixed-label-seq-pretrain', action='store_true')
    parser.add_argument('--cv-fold', type=int, default=0)
    args = parser.parse_args()

    if args.use_cuda and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    torch.manual_seed(args.seed)
    fixed_label_seq_pretrain = args.fixed_label_seq_pretrain
    value_weight = args.value_weight
    reward_function = args.reward_function
    dataset = args.dataset
    alpha = args.alpha
    gamma = args.gamma
    entropy_penalty = args.entropy_penalty
    emb_size = args.embedding_size
    rnn_input_size = args.rnn_input_size
    rnn_hidden_size = args.rnn_hidden_size
    hid_size = args.hidden_size
    bottleneck_size = args.bottleneck_size
    num_layers = args.num_layers
    dropout_prob = args.dropout_prob
    input_dropout_prob = args.input_dropout_prob
    batch_size = args.minibatch_size
    max_epoch = args.max_epoch
    pretrain_max_epoch = args.pretrain_max_epoch
    learning_rate = args.lr
    weight_decay = args.weight_decay
    max_pretrain_grad_norm = args.pretrain_grad_norm
    max_grad_norm = args.grad_norm
    nonlinear_func = args.nonlinear_func
    label_order = args.label_order
    cv_fold = args.cv_fold

    main(dataset, pretrain_max_epoch, max_epoch, learning_rate, weight_decay, max_pretrain_grad_norm, max_grad_norm,
         batch_size, emb_size, rnn_input_size, rnn_hidden_size, hid_size, bottleneck_size, 
         entropy_penalty, gamma, alpha, nonlinear_func=nonlinear_func, label_order=label_order, reward_function=reward_function, value_weight=value_weight,
         input_dropout_prob=input_dropout_prob, dropout_prob=dropout_prob, num_layers=num_layers, cv_fold=cv_fold, seed=args.seed, fixed_label_seq_pretrain=fixed_label_seq_pretrain)
