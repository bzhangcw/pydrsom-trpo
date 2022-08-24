from garage import log_performance
from garage.np import discount_cumsum
from garage.np import unflatten_tensors
from garage.torch.optimizers import OptimizerWrapper
from garage.torch import filter_valids, compute_advantages
import torch
import torch.nn.functional as F
import numpy as np
import copy

from rl_algorithm import RLAlgorithm
from DRSOMOptimizer import DRSOMOptimizer


class TRPO_DRSOM(RLAlgorithm):

    def __init__(self,
                 env_spec,
                 policy,
                 value_function,
                 sampler,
                 policy_optimizer=None,
                 vf_optimizer=None,
                 num_train_per_epoch=1,
                 discount=0.99,
                 gae_lambda=1,
                 center_adv=True,
                 positive_adv=False):

        if vf_optimizer:
            self._vf_optimizer = vf_optimizer
        else:
            self._vf_optimizer = OptimizerWrapper(torch.optim.Adam, value_function)

        if policy_optimizer:
            self._policy_optimizer = policy_optimizer
        else:
            self._policy_optimizer = OptimizerWrapper((DRSOMOptimizer, dict(max_constraint_value=0.01)), policy)

        self._max_episode_length = env_spec.max_episode_length
        self._env_spec = env_spec
        self._sampler = sampler
        self._n_samples = num_train_per_epoch
        self._value_function = value_function

        self._discount = discount
        self._center_adv = center_adv
        self._positive_adv = positive_adv
        self._epsilon = 1e-8
        self._gae_lambda = gae_lambda

        self.policy = policy
        # self._old_policy = copy.deepcopy(self.policy)
        self._old_policy_m = copy.deepcopy(self.policy)

        self._mix_policy = copy.deepcopy(self.policy)
        self._pos_g_policy = copy.deepcopy(self.policy)
        self._neg_g_policy = copy.deepcopy(self.policy)
        self._pos_m_policy = copy.deepcopy(self.policy)
        self._neg_m_policy = copy.deepcopy(self.policy)

    def train(self, trainer):
        last_return = None
        for _ in trainer.step_epochs():
            for _ in range(self._n_samples):
                eps = trainer.obtain_episodes(trainer.step_itr)
                last_return = self._train_once(trainer.step_itr, eps)
                trainer.step_itr += 1

        return last_return

    def _train_once(self, itr, eps):
        obs = torch.Tensor(eps.padded_observations)
        rewards = torch.Tensor(eps.padded_rewards)
        returns = torch.Tensor(
            np.stack([
                discount_cumsum(reward, self._discount)
                for reward in eps.padded_rewards
            ]))
        valids = eps.lengths

        with torch.no_grad():
            baselines = self._value_function(obs)

        obs_flat = torch.Tensor(eps.observations)
        actions_flat = torch.Tensor(eps.actions)
        rewards_flat = torch.Tensor(eps.rewards)
        returns_flat = torch.cat(filter_valids(returns, valids))
        advs_flat = self._compute_advantage(rewards, valids, baselines)

        self._train(obs_flat, actions_flat, rewards_flat, returns_flat,
                    advs_flat, valids, itr)

        # self._old_policy.load_state_dict(self.policy.state_dict())

        undiscounted_returns = log_performance(itr,
                                               eps,
                                               discount=self._discount)
        return np.mean(undiscounted_returns)

    def _compute_advantage(self, rewards, valids, baselines):
        advantages = compute_advantages(self._discount, self._gae_lambda,
                                        self._max_episode_length, baselines,
                                        rewards)
        advantage_flat = torch.cat(filter_valids(advantages, valids))

        if self._center_adv:
            means = advantage_flat.mean()
            variance = advantage_flat.var()
            advantage_flat = (advantage_flat - means) / (variance + 1e-8)

        if self._positive_adv:
            advantage_flat -= advantage_flat.min()

        return advantage_flat

    def _train(self, obs, actions, rewards, returns, advs, valids, itr):
        for dataset in self._policy_optimizer.get_minibatch(obs, actions, rewards, advs):
            self._train_policy_first_order(*dataset, valids, itr)
        for dataset in self._vf_optimizer.get_minibatch(obs, returns):
            self._train_value_function(*dataset)

    def _train_policy_first_order(self, obs, actions, rewards, advs, valids, itr):
        """
        max g_{old}^{T} (theta - theta_{old})
        s.t. 1/2 (theta - theta_{old})^{T} F_{old} (theta - theta_{old}) <= delta
        """
        self._policy_optimizer.zero_grad()
        loss = self.get_gradients(obs, actions, advs)
        loss.backward()

        old = self._old_policy_m.get_param_value_new().detach()
        now = self.policy.get_param_value_new().detach()

        print('old policy para is')
        print(old)
        print('----------------------------------')


        print('params now is:')
        print(now)
        print('----------------------------------')

        m = now - old

        print('m is:')
        print(m)
        print('----------------------------------')

        alpha, g, Fg, Fm, params = self._policy_optimizer.compute_alpha(m=m,
                                                       f_constraint=lambda: self._compute_kl_constraint(obs), itr=itr)

        # param_shapes = [p.shape or torch.Size([1]) for p in params]
        # g_step = unflatten_tensors(g, param_shapes)
        # m_step = unflatten_tensors(m, param_shapes)
        # Fg_step = unflatten_tensors(Fg, param_shapes)
        # Fm_step = unflatten_tensors(Fm, param_shapes)

        # for g_step_, m_step_, Fg_step_, Fm_step_, param_ in zip(g_step, m_step, Fg_step, Fm_step, params):
        #     new_param_ = param_.data + alpha[0] * g_step_ + alpha[1] * m_step_ + alpha[2] * Fg_step_ + alpha[3] * Fm_step_
        #     param_.data = new_param_.data

        self._old_policy_m.set_param_value_new(now)


        params_new = now + alpha[0] * g + alpha[1] * m + alpha[2] * Fg + alpha[3] * Fm
        print('params_new is:')
        print(params_new)
        print('----------------------------------')
        self.policy.set_param_value_new(params_new)

        return loss

    def _compute_kl_constraint(self, obs):
        with torch.no_grad():
            old_dist = self._old_policy_m(obs)[0]

        new_dist = self.policy(obs)[0]

        kl_constraint = torch.distributions.kl.kl_divergence(
            old_dist, new_dist)
        return kl_constraint.mean()

    def get_gradients(self, obs, actions, advs):
        loss = self._compute_objective(obs, actions, advs)
        # loss.backward()
        # grad = policy.get_grads()
        return loss

    def _compute_objective(self, obs, actions, advs):

        with torch.no_grad():
            old_loglikelihood = self._old_policy_m(obs)[0].log_prob(actions)

        new_loglikelihood = self.policy(obs)[0].log_prob(actions)
        likelihood_ratio = (new_loglikelihood - old_loglikelihood).exp()

        # Calculate surrogate
        surrogate = likelihood_ratio * advs

        return surrogate.mean()

    def _train_value_function(self, obs, returns):
        self._vf_optimizer.zero_grad()
        loss = self._value_function.compute_loss(obs, returns)
        loss.backward()
        self._vf_optimizer.step()

        return loss
