import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from reward_functions import RewardCalculator, NDCGReward, F1ScoreReward, NonRepetitionReward, NegativeStepReward


class LNGRUCell(nn.Module):
    def __init__(self, input_size, hidden_size, bias=True, ln=True, elementwise_affine=True):
        super(LNGRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.i2h = nn.Linear(input_size, 3 * hidden_size, bias=bias)
        self.h2h = nn.Linear(hidden_size, 3 * hidden_size, bias=bias)

        # layer normalization
        self.ln_i2h = nn.LayerNorm(3 * hidden_size, elementwise_affine=elementwise_affine)
        self.ln_h2h = nn.LayerNorm(3 * hidden_size, elementwise_affine=elementwise_affine)

    def forward(self, input, hidden, dropout_masks=None):

        if dropout_masks is not None and self.training:
            i2h = dropout_masks[0] * input
            h2h = dropout_masks[1] * hidden
        else:
            i2h = input
            h2h = hidden

        i2h = self.i2h(i2h)
        h2h = self.h2h(h2h)

        i2h = self.ln_i2h(i2h)
        h2h = self.ln_h2h(h2h)

        i_r, i_u, i_n = i2h.chunk(3, 1)
        h_r, h_u, h_n = h2h.chunk(3, 1)

        resetgate = torch.sigmoid(i_r + h_r)
        updategate = torch.sigmoid(i_u + h_u)

        new_h = torch.tanh(i_n + resetgate * h_n)

        h = new_h + updategate * (hidden - new_h)

        return h


class PolicySearch(nn.Module):
    def __init__(self, config):
        super(PolicySearch, self).__init__()

        self.config = config


class Reinforce(PolicySearch):
    def __init__(self, config):
        super(Reinforce, self).__init__(config)

        self.input_projector = nn.Linear(config.feature_dim, config.rnn_hidden_size)
        self.input_nonlinear = nn.ReLU() if config.nonlinearity == 'relu' else nn.Tanh()
        self.embedding = nn.Embedding(config.output_size, config.embedding_size)
        self.ctx_proj = nn.Linear(config.context_dim, config.embedding_size)
        self.ctx_nonlinear = nn.ReLU() if config.nonlinearity == 'relu' else nn.Tanh()
        self.rnn = LNGRUCell(input_size=config.embedding_size * 2,
                             hidden_size=config.rnn_hidden_size)
        self.emb_hid_proj = nn.Linear(config.embedding_size, config.hidden_size, bias=True)
        self.rnn_hid_proj = nn.Linear(config.rnn_hidden_size, config.hidden_size, bias=False)
        self.hidden_nonlinear = nn.ReLU() if config.nonlinearity == 'relu' else nn.Tanh()
        self.hidden_dropout = nn.Dropout(p=config.dropout_prob)

        if config.use_bottleneck:
            self.output_bottleneck = nn.Linear(config.hidden_size, config.bottleneck_size)
            self.bottleneck_nonlinear = nn.ReLU() if config.nonlinearity == 'relu' else nn.Tanh()
            self.bottleneck_dropout = nn.Dropout(p=config.dropout_prob)
            self.policy_output = nn.Linear(config.bottleneck_size, config.output_size)
        else:
            self.policy_output = nn.Linear(config.hidden_size, config.output_size)

    def forward(self, ctx, xt, htm1, dropout_masks=None):
        # xt: (batch,)
        # htm1: (batch, hidden_dim)
        embedded_inputs = self.embedding(xt)

        # ctx_proj: (batch, rnn_input_size)
        ctx_proj = self.ctx_nonlinear(self.ctx_proj(ctx))

        # rnn_input: (batch, 2 * embedding_size)
        rnn_inputs = torch.cat((embedded_inputs, ctx_proj), dim=1)

        # rnn_outputs: (batch, hidden_size)
        rnn_outputs = self.rnn(rnn_inputs, htm1, dropout_masks)

        states = self.hidden_nonlinear(self.emb_hid_proj(embedded_inputs) + self.rnn_hid_proj(rnn_outputs))
        states = self.hidden_dropout(states)

        if self.config.use_bottleneck:
            reduced_states = self.bottleneck_nonlinear(self.output_bottleneck(states))
            reduced_states = self.bottleneck_dropout(reduced_states)

            # action_scores: (batch, output_size)
            action_scores = self.policy_output(reduced_states)
        else:
            action_scores = self.policy_output(states)

        outputs = [action_scores, states]

        return outputs, rnn_outputs

    def init_hidden(self, input_data, device):
        return self.input_nonlinear(self.input_projector(input_data))


class ActorCritic(Reinforce):
    def __init__(self, config):
        super(ActorCritic, self).__init__(config)
        self.value_hidden = nn.Linear(config.state_size, config.state_size)
        self.value_output = nn.Linear(config.state_size, 1)

    def forward(self, ctx, xt, htm1, state_value_grad=True, dropout_masks=None):
        [action_scores, states], rnn_outputs = super(ActorCritic, self).forward(ctx, xt, htm1, dropout_masks=dropout_masks)

        with torch.set_grad_enabled(state_value_grad):
            # state_values: (batch_size)
            # disconnect the backward path of the gradient wrt. states
            state_hidden = torch.tanh(self.value_hidden(states.detach()))
            state_values = self.value_output(state_hidden).view(-1)

        outputs = [action_scores, states, state_values]

        return outputs, rnn_outputs


class Environment(object):
    def __init__(self, config):
        super(Environment, self).__init__()

        self.gamma = config.gamma
        self.STOP = config.STOP
        self.output_size = config.output_size
        self.reward_function = config.reward_function

        self.saved_rewards = []
        self.saved_log_probs = []
        self.saved_state_values = []
        self.saved_entropy_terms = []
        self.saved_masks = []

        self.saved_actions = np.zeros((config.minibatch_size, config.max_trials), dtype=int)

        # temp variables
        self.stopped = np.zeros(config.minibatch_size, dtype=bool)

        self.config = config
        self.step_counter = 0

        self.max_reward_step = np.inf
        # reward
        self.reward_calculator = RewardCalculator()
        if self.reward_function.startswith('ndcg'):
            k = int(self.reward_function[4])
            assert k == 5
            self.max_reward_step = k
            self.reward_calculator.add(NDCGReward(self.STOP, k=k, weight=1.0))
            # self.reward_calculator.add(NegativeStepReward(weight=0.1))
        elif self.reward_function == 'exf1':
            self.reward_calculator.add(F1ScoreReward(self.STOP, weight=1))
        else:
            raise ValueError('{} not supported'.format(self.reward_function))


    def clear_episode_temp_data(self):
        del self.saved_rewards[:]
        del self.saved_log_probs[:]
        del self.saved_state_values[:]
        del self.saved_entropy_terms[:]
        del self.saved_masks[:]

        self.saved_actions.fill(0)
        self.stopped.fill(False)

        self.step_counter = 0
        self.reward_calculator.clear_history()

    def step(self, model_outputs, targets=None, stochastic=True, no_stop_action=False):
        if len(model_outputs) == 3:
            logits, states, values = model_outputs
        else:
            logits, states = model_outputs

        # choosing an action
        if stochastic:
            probs = F.softmax(logits, dim=-1)
            log_probs = F.log_softmax(logits, dim=-1)
            actions = probs.multinomial(num_samples=1).detach().squeeze()
        else:
            # only call the following for making predictions
            # targets is None
            if no_stop_action:
                # do NOT generate stop actions
                logits[:, self.STOP] = -np.inf

            _, actions = torch.max(logits, dim=1)
            
        actions_np = actions.data.cpu().numpy().astype(int)
        batch_size = actions_np.shape[0]
        is_stop_action_ = actions_np == self.config.STOP
        is_stop_action = np.zeros_like(self.stopped, dtype=bool)
        is_stop_action[:batch_size] = is_stop_action_

        rewards = None

        # targets are given in order to compute partial reward per step
        if targets:

            logp = torch.gather(log_probs, 1, actions.unsqueeze(1)).squeeze()
            entropy = -(log_probs * probs).sum(1)

            masking_condition = self.stopped[:batch_size] | self.step_counter >= self.max_reward_step
            self.saved_masks.append(1. - masking_condition)
            self.saved_log_probs.append(logp)
            self.saved_entropy_terms.append(entropy)
            self.saved_state_values.append(values)
            self.saved_actions[:batch_size, self.step_counter] = actions_np

            options = {
                'n_labels': self.output_size,
                'stopped': self.stopped[:batch_size],
                'targets': targets
            }

            rewards = self.reward_calculator.calc(self.saved_actions[:batch_size, :self.step_counter+1], **options)
            self.saved_rewards.append(rewards)

        self.stopped[is_stop_action] = True
        self.step_counter += 1
        done = True if np.all(self.stopped[:batch_size]) else False

        del actions_np

        return actions, rewards, done

    def get_episode_results(self):
        mb_sz = len(self.saved_log_probs[0])

        R = 0
        rewards = []
        for r in self.saved_rewards[::-1]:
            R = r + self.gamma * R
            rewards.insert(0, R)

        rewards = torch.tensor(rewards, dtype=torch.float, requires_grad=False)
        # print('Reward max: {}, min: {}'.format(rewards.max(), rewards.min()))

        results = []
        results += [self.saved_log_probs]
        results += [self.saved_state_values]
        results += [self.saved_entropy_terms]
        results += [self.saved_masks]
        results += [rewards]

        return results



