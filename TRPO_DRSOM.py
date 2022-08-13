from lib2to3.pygram import pattern_symbols
from garage import log_performance
from garage.np import discount_cumsum
from garage.np.optimizers import BatchDataset
from garage.torch.optimizers import OptimizerWrapper
from garage.torch import filter_valids, compute_advantages
import torch
import torch.nn.functional as F
import numpy as np
import copy



class TRPO_DRSOM():

    def __init__(self, 
                 env_spec,
                 policy,
                 value_function,
                 sampler, 
                 policy_optimizer = None,
                 vf_optimizer = None, 
                 num_train_per_epoch = 1,
                 discount = 0.99,
                 center_adv = True, 
                 positive_adv = False):
                 
        if vf_optimizer:
            self._vf_optimizer = vf_optimizer
        else:
            self._vf_optimizer = OptimizerWrapper(torch.optim.Adam, value_function)

        if policy_optimizer:
            self._policy_optimizer = policy_optimizer
        else:
            self._policy_optimizer = OptimizerWrapper(torch.optim.Adam, policy)

        self._max_episode_length = env_spec.max_episode_length
        self._sampler = sampler
        self._vf_optimizer = vf_optimizer
        self._n_samples = num_train_per_epoch
        self._value_function = value_function

        self._discount = discount
        self._center_adv = center_adv
        self._positive_adv = positive_adv
        self._epsilon = 1e-8

        self.policy = policy
        self._old_policy = copy.deepcopy(self.policy)
        self._p_policy = copy.deepcopy(self.policy)
        self._m_policy = copy.deepcopy(self.policy)
        

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
                    advs_flat, valids)

        self._old_policy.load_state_dict(self.policy.state_dict())

        undiscounted_returns = log_performance(itr,
                                               eps,
                                               discount=self._discount)
        return np.mean(undiscounted_returns)


    def _compute_advantage(self, rewards, valids, baselines):
        advantages = compute_advantages(self._discount, self._gae_lambda,
                                        self.max_episode_length, baselines,
                                        rewards)
        advantage_flat = torch.cat(filter_valids(advantages, valids))

        if self._center_adv:
            means = advantage_flat.mean()
            variance = advantage_flat.var()
            advantage_flat = (advantage_flat - means) / (variance + 1e-8)

        if self._positive_adv:
            advantage_flat -= advantage_flat.min()

        return advantage_flat


    def _train(self, obs, actions, rewards, returns, advs, valids):
        for dataset in self._policy_optimizer.get_minibatch(obs, actions, rewards, advs):
            self._train_policy(*dataset, valids)
        for dataset in self._vf_optimizer.get_minibatch(obs, returns):
            self._train_value_function(*dataset)


    def _train_policy(self, obs, actions, rewards, advs, valids):

        d_vector = self.policy.get_param_values() - self._old_policy.get_param_values()
        grad = self.policy.get_grads()

        p_policy_params = self.policy.get_param_values() + d_vector * self._epsilon 
        m_policy_params = self.policy.get_param_values() + grad * self._epsilon

        self._p_policy.set_param_values(p_policy_params)
        self._m_policy.set_param_values(m_policy_params)

        grad_p = self.get_gradinets(obs, actions, advs, valids, self._p_policy)
        grad_m = self.get_gradinets(obs, actions, advs, valids, self._m_policy)

        hessian_g = (grad_p - grad) / self._epsilon
        hessian_m = (grad_m - grad) / self._epsilon

        # compute Q and G_{k}




    def get_gradinets(self, sub_obs, sub_actions, sub_advs, valids, policy):
        loss= self._compact_objective(sub_obs, sub_actions, sub_advs, valids, policy)
        loss = loss.mean()
        loss.backward()
        grad = policy.get_grads()
        return grad 

    def _compact_objective(self, obs, actions, advs, valids, policy):
        pattern_symbols
        


    def _train_value_function(self, obs, returns):
        self._vf_optimizer.zero_grad()
        loss = self._value_function.compute_loss(obs, returns)
        loss.backward()
        self._vf_optimizer.step()

        return loss