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

class TRPO_DRSOM(RLAlgorithm):

    def __init__(self, 
                 env_spec,
                 policy,
                 value_function,
                 sampler, 
                 policy_optimizer = None,
                 vf_optimizer = None, 
                 num_train_per_epoch = 1,
                 discount = 0.99,
                 gae_lambda = 1, 
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
        self._env_spec = env_spec
        self._sampler = sampler
        self._vf_optimizer = vf_optimizer
        self._n_samples = num_train_per_epoch
        self._value_function = value_function

        self._discount = discount
        self._center_adv = center_adv
        self._positive_adv = positive_adv
        self._epsilon = 1e-8
        self._hvp_reg_coeff = 1e-5
        self._cg_iters = 10
        self._radius = 0.01
        self._gae_lambda = gae_lambda

        self.policy = policy
        self._old_policy = copy.deepcopy(self.policy)
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
                    advs_flat, valids)

        self._old_policy.load_state_dict(self.policy.state_dict())

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


    def _train(self, obs, actions, rewards, returns, advs, valids):
        for dataset in self._policy_optimizer.get_minibatch(obs, actions, rewards, advs):
            self._train_policy_first_order(*dataset, valids)
        for dataset in self._vf_optimizer.get_minibatch(obs, returns):
            self._train_value_function(*dataset)


    def _train_policy_first_order(self, obs, actions, rewards, advs, valids):
        """
        max g_{old}^{T} (theta - theta_{old})
        s.t. 1/2 (theta - theta_{old})^{T} F_{old} (theta - theta_{old}) <= delta

        """

        g_vector, loss = self.get_gradients(obs, actions, advs, valids, self.policy)
        f_constraint = self._compute_kl_constraint(obs)

        params = self.policy.get_param_value()
        FVP = self._KL_HVP(f_constraint, params, self._hvp_reg_coeff)

        m_vector = self.policy.get_param_values() - self._old_policy.get_param_values()

        Fg = FVP(g_vector)
        Fm = FVP(m_vector)

        gFg = torch.dot(g_vector, Fg)
        mFm = torch.dot(m_vector, Fm)
        mFg = torch.dot(m_vector, Fg)

        gg = torch.dot(g_vector, g_vector)
        mg = torch.dot(m_vector, g_vector)

        G = torch.tensor([[gFg, mFg], [mFg, mFm]], requires_grad = False)
        c = torch.tensor([gg, mg], requires_grad = False)

        x = self._conjugate_gradient(FVP, G, c, self._cg_iters)

        Gx = torch.dot(G, x)

        alpha = np.sqrt(2 * self._radius * (1. / (torch.dot(x, Gx) + 1e-8))) * x

        params_new = alpha[0] * g_vector + alpha[1] * m_vector

        self.policy.set_params_values(params_new)

        return loss


    def _conjugate_gradient(self, FVP, b, cg_iters, residual_tol=1e-10):
        """Use Conjugate Gradient iteration to solve Ax = b. Demmel p 312.

        Args:
            f_Ax (callable): A function to compute Hessian vector product.
            b (torch.Tensor): Right hand side of the equation to solve.
            cg_iters (int): Number of iterations to run conjugate gradient
                algorithm.
            residual_tol (float): Tolerence for convergence.

        Returns:
            torch.Tensor: Solution x* for equation Ax = b.

        """
        p = b.clone()
        r = b.clone()
        x = torch.zeros_like(b)
        rdotr = torch.dot(r, r)

        for _ in range(cg_iters):
            z = FVP(p)
            v = rdotr / torch.dot(p, z)
            x += v * p
            r -= v * z
            newrdotr = torch.dot(r, r)
            mu = newrdotr / rdotr
            p = r + mu * p

            rdotr = newrdotr
            if rdotr < residual_tol:
                break
        return x



    def _compute_kl_constraint(self, obs):
        with torch.no_grad():
            old_dist = self._old_policy(obs)[0]

        new_dist = self.policy(obs)[0]

        kl_constraint = torch.distributions.kl.kl_divergence(
            old_dist, new_dist)
        return kl_constraint.mean()



    def _KL_HVP(self, func, params, reg_coeff=1e-5):
  
        param_shapes = [p.shape or torch.Size([1]) for p in params]
        f = func()
        f_grads = torch.autograd.grad(f, params, create_graph=True)

        def _eval(vector):
            
            unflatten_vector = unflatten_tensors(vector, param_shapes)

            assert len(f_grads) == len(unflatten_vector)
            grad_vector_product = torch.sum(
                torch.stack(
                    [torch.sum(g * x) for g, x in zip(f_grads, unflatten_vector)]))

            hvp = list(
                torch.autograd.grad(grad_vector_product, params,
                                    retain_graph=True))
            for i, (hx, p) in enumerate(zip(hvp, params)):
                if hx is None:
                    hvp[i] = torch.zeros_like(p)

            flat_output = torch.cat([h.reshape(-1) for h in hvp])
            return flat_output + reg_coeff * vector

        return _eval
    



    def _train_policy_second_order(self, obs, actions, rewards, advs, valids):

        self.generate_mix_policy()
        g_mix = self.get_gradients(obs, actions, advs, valids, self._mix_policy)
        g_lh = self.get_gradients(obs, actions, None, valids, self._mix_policy)

        m_vector = self.policy.get_param_values() - self._old_policy.get_param_values()
        g_vector = self.get_gradients(obs, actions, advs, valids, self.policy)

        # Here we treat the hessian of L function as the hessian of V function
        # for the momentum part
        pos_m_params = self._mix_policy.get_param_values() + m_vector * self._epsilon
        neg_m_params = self._mix_policy.get_param_values() - m_vector * self._epsilon
        self._pos_m_policy.set_param_values(pos_m_params)
        self._neg_m_policy.set_param_values(neg_m_params)

        # first component, for the momentum part
        inner_product_m = torch.dot(g_lh, m_vector)
        fst_m = inner_product_m * g_mix

        # second component, for the momentum part
        pos_m = self.get_gradients(obs, actions, advs, valids, self._pos_m_policy)
        neg_m = self.get_gradients(obs, actions, advs, valids, self._neg_m_policy)
        hm = (pos_m - neg_m) / (2 * self._epsilon)

        # for the gradient part
        pos_g_params = self._mix_policy.get_param_values() + g_vector * self._epsilon
        neg_g_params = self._mix_policy.get_param_values() - g_vector * self._epsilon
        self._pos_g_policy.set_param_values(pos_g_params)
        self._neg_g_policy.set_param_values(neg_g_params)

        # first component, for the gradient part
        inner_product_g = torch.dot(g_lh, g_vector)
        fst_g = inner_product_g * g_mix

        # second component, for the gradient part
        pos_g = self.get_gradients(obs, actions, advs, valids, self._pos_g_policy)
        neg_g = self.get_gradients(obs, actions, advs, valids, self._neg_g_policy)
        hg = (pos_g - neg_g) / (2 * self._epsilon)

        # add respectively
        hessian_m = fst_m + hm
        hessian_g = fst_g + hg

        # compute Q_{k}
        # TBA.

        # How to compute G_{k}
        # TBA. 


    def generate_mix_policy(self):
        a = np.random.uniform(0.0, 1.0)
        mix = a * self.policy.get_param_values() + (1 - a) * self._old_policy.get_param_values()
        self._mix_policy.set_param_values(mix)

    def get_gradients(self, obs, actions, advs, valids, policy):
        loss = self._compute_objective(obs, actions, advs, valids, policy)
        loss = loss.mean()
        loss.backward()
        grad = policy.get_grads()
        return grad, loss


    def _compute_objective(self, obs, actions, advs, valids, policy):

        with torch.no_grad():
            old_loglikelihood = self._old_policy(obs)[0].log_prob(actions)

        new_loglikelihood = self.policy(obs)[0].log_prob(actions)
        likelihood_ratio = (new_loglikelihood - old_loglikelihood).exp()

        # Calculate surrogate
        surrogate = likelihood_ratio * advs

        return surrogate

    def _train_value_function(self, obs, returns):
        self._vf_optimizer.zero_grad()
        loss = self._value_function.compute_loss(obs, returns)
        loss.backward()
        self._vf_optimizer.step()

        return loss