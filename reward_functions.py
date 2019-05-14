import time

import numpy as np


class RewardCalculator(object):
    def __init__(self):
        self.reward_calculators = []
        self.past_reward = []

    def add(self, reward_calc):
        self.reward_calculators.append(reward_calc)
        self.past_reward.append(0)

    def calc(self, actions, **options):
        reward_outputs = []

        for idx, reward_calc in enumerate(self.reward_calculators):
            output = reward_calc.compute(actions, **options)
            if isinstance(reward_calc, NDCGReward) or isinstance(reward_calc, F1ScoreReward):
                prev_output = self.past_reward[idx]
                self.past_reward[idx] = output
                output -= prev_output

            reward_outputs.append(output)

        total_reward = np.array(reward_outputs).sum(axis=0)

        return total_reward.astype('float32')

    def clear_history(self):
        self.past_reward = [0] * len(self.reward_calculators)
        for reward_calc in self.reward_calculators:
            reward_calc.reset_interim_vars()


class Reward(object):
    def __init__(self, weight):
        super(Reward, self).__init__()
        self.weight = weight

    def compute(self, actions, **kwargs):
        raise NotImplementedError("Implement episode-level reward")

    def reset_interim_vars(self):
        pass


class NegativeStepReward(Reward):
    def __init__(self, weight=1):
        super(NegativeStepReward, self).__init__(weight)

    def compute(self, actions, **kwargs):
        if 'targets' not in kwargs:
            raise ValueError('error')

        targets = kwargs['targets']

        target_lengths = np.array(list(map(len, targets)))
        return -self.weight / target_lengths


class F1ScoreReward(Reward):
    def __init__(self, stop_id, weight=1):
        super(F1ScoreReward, self).__init__(weight)

        self.stop_id = stop_id

    def compute(self, actions, **kwargs):

        if 'stopped' not in kwargs:
            raise ValueError('error')

        stopped = kwargs['stopped']

        if 'targets' not in kwargs:
            raise ValueError('error')

        targets = kwargs['targets']

        mb_sz = len(targets)
        rewards = np.zeros(mb_sz)

        for i in range(mb_sz):

            t = set(targets[i])
            p = set(actions[i])

            inter_sec = t.intersection(p)

            pred_size = len(p)
            trg_size = len(t)
            correct_pred_size = len(inter_sec)

            if (pred_size == 0 and trg_size == 0) or correct_pred_size == 0:
                rewards[i] = 0
            else:
                prec = correct_pred_size / pred_size if pred_size > 0 else 0
                recall = correct_pred_size / trg_size if trg_size > 0 else 0

                rewards[i] = 2 * (prec * recall) / (prec + recall)

        rewards *= self.weight

        return rewards


class NDCGReward(Reward):
    def __init__(self, stop_id, k=5, weight=1):
        super(NDCGReward, self).__init__(weight)

        self.stop_id = stop_id
        self.k = k
        self.k_indices = {1:0, 3:1, 5:2}
        self.max_k = np.max(list(self.k_indices.keys()))

        self.j = -1
        self.action_ranking = []
        self.targets = None
        self.stopped = None
        self.dcg_scores = None
        self.max_len = None

    def reset_interim_vars(self):
        self.j = -1
        self.targets = None
        self.dcg_scores = None

        if len(self.action_ranking) > 0:
            del self.action_ranking[:]

    def _ndcg(self, actions):
        assert self.j < actions.shape[1], '{}, {}'.format(self.j, actions.shape[1])
        assert self.targets is not None

        new_actions = actions[:, self.j]
        mask = new_actions == self.stop_id
        self.stopped[mask] = True
        rank = self.j + 1
        tgt_lengths = np.zeros(new_actions.shape[0])

        for idx, (targets_, action_, is_stop) \
                in enumerate(zip(*(self.targets, new_actions, self.stopped))):
            tgt_lengths[idx] = len(targets_)
            if rank > self.max_k:
                break

            if is_stop:
                self.dcg_scores[idx].append(0)
            else:
                _dcg = float(action_ in targets_) / np.log2(rank + 1)
                denom = np.sum(1 / np.log2(np.arange(1, self.max_len[idx] +1)+1))
                self.dcg_scores[idx].append(_dcg / denom)

        stats = np.array(self.dcg_scores)
        DCG_5 = np.sum(stats[:, :np.min([5, rank])], axis=1)

        DCG_scores = DCG_5

        return DCG_scores

    def compute(self, actions, **kwargs):
        if 'n_labels' not in kwargs:
            raise ValueError('error')

        n_labels = kwargs['n_labels']

        if 'targets' not in kwargs:
            raise ValueError('error')

        targets = kwargs['targets']
        mb_sz = len(targets)

        if self.j == -1:
            assert len(self.action_ranking) == 0
            for i in range(mb_sz):
                self.action_ranking.append(list())
            self.targets = [set(targets_) for targets_ in targets]
            self.stopped = np.zeros(mb_sz, dtype=np.bool)
            self.dcg_scores = [list() for _ in range(mb_sz)]

            # compute the normalization factor for nDCG
            # NOTE: no stop id appended
            tgt_lengths = np.array(list(map(len, self.targets)))
            self.max_len = np.clip(tgt_lengths, -np.inf, 5)

        # step increment
        self.j += 1

        if self.j <= 5:
            nDCG_scores = self._ndcg(actions)
        else:
            # return zero reward if the length of generated sequence is greateer than k (say 5)
            nDCG_scores = np.zeros(mb_sz)

        return nDCG_scores * self.weight


class SetSizePredictionReward(Reward):
    def __init__(self, stop_id, weight=1):
        super(SetSizePredictionReward, self).__init__(weight)

        self.stop_id = stop_id

    def compute(self, actions, **kwargs):
        if 'stopped' not in kwargs:
            raise ValueError('error')

        stopped = kwargs['stopped']

        if 'stop_action_pos' not in kwargs:
            raise ValueError('error')

        stop_action_pos = kwargs['stop_action_pos']

        if 'targets' not in kwargs:
            raise ValueError('error')

        targets = kwargs['targets']

        mb_sz = actions.shape[0]
        rewards = np.zeros(mb_sz)

        for i in range(mb_sz):
            if stop_action_pos[i] > -1 and not stopped[i]:

                t = set(targets[i]).difference(set([self.stop_id]))
                p = set(actions[i, :stop_action_pos[i]])

                rewards[i] = int(t == p)

        rewards *= self.weight

        return rewards


class NonRepetitionReward(Reward):
    def __init__(self, weight=1):
        super(NonRepetitionReward, self).__init__(weight)

    def compute(self, actions, **kwargs):
        if 'repeated_actions' not in kwargs:
            raise ValueError('error')

        repeated_actions = kwargs['repeated_actions']

        if 'stop_action_pos' not in kwargs:
            raise ValueError('error')

        stop_action_pos = kwargs['stop_action_pos']

        mb_sz = actions.shape[0]
        rewards = np.zeros(mb_sz)

        rewards[np.logical_and(repeated_actions, stop_action_pos == -1)] -= 1

        rewards *= self.weight

        return rewards
