from cmath import isnan
from re import X
import warnings

from dowel import logger
import numpy as np
import torch
from torch.optim import Optimizer

from garage.np import unflatten_tensors


def _build_hessian_vector_product(func, params, reg_coeff=1e-5):
    """Computes Hessian-vector product using Pearlmutter's algorithm.

    `Pearlmutter, Barak A. "Fast exact multiplication by the Hessian." Neural
    computation 6.1 (1994): 147-160.`

    Args:
        func (callable): A function that returns a torch.Tensor. Hessian of
            the return value will be computed.
        params (list[torch.Tensor]): A list of function parameters.
        reg_coeff (float): A small value so that A -> A + reg*I.

    Returns:
        function: It can be called to get the final result.

    """
    param_shapes = [p.shape or torch.Size([1]) for p in params]
    f = func()
    f_grads = torch.autograd.grad(f, params, create_graph=True)

    def _eval(vector):
        """The evaluation function.

        Args:
            vector (torch.Tensor): The vector to be multiplied with
                Hessian.

        Returns:
            torch.Tensor: The product of Hessian of function f and v.

        """
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
        # return flat_output + reg_coeff * vector
        return flat_output

    return _eval


class DRSOMOptimizer(Optimizer):
    def __init__(self,
                 params,
                 max_constraint_value,
                 cg_iters=10,
                 hvp_reg_coeff=1e-5):
        super().__init__(params, {})
        self._cg_iters = cg_iters
        self._hvp_reg_coeff = hvp_reg_coeff
        self._max_constraint_value = max_constraint_value

    def compute_alpha(self, f_loss, f_constraint, itr, radius):
        params = []
        grads = []
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    params.append(p)
                    grads.append(p.grad.reshape(-1))
        flat_loss_grads = torch.cat(grads)

        f_Ax = _build_hessian_vector_product(f_constraint, params,
                                             self._hvp_reg_coeff)

        # damping = 1e-5
        #
        # def f_Ax(v):
        #     f = f_constraint()
        #     f_grads = torch.autograd.grad(f, params, create_graph=True)
        #     flat_grad_kl = torch.cat([f_grad.view(-1) for f_grad in f_grads])
        #     kl_v = (flat_grad_kl * v).sum()
        #
        #     grads = torch.autograd.grad(kl_v, params)
        #     flat_grad_grad_kl = torch.cat([grad.contiguous().view(-1) for grad in grads]).detach()
        #
        #     return flat_grad_grad_kl + v * damping

        # def f_Ax_1(v, params):
        #     f = f_constraint()
        #     f_grads = torch.autograd.grad(f, params, create_graph=True)
        #     flat_grad_kl = torch.cat([f_grad.view(-1) for f_grad in f_grads])
        #
        #     epsilon = 1e-8
        #
        #     prev_params = [p.clone() for p in params]
        #
        #     param_shapes = [p.shape or torch.Size([1]) for p in params]
        #     v = unflatten_tensors(v, param_shapes)
        #
        #     for prev_param, v_, param in zip(prev_params, v, params):
        #         new_param = prev_param.data + epsilon * v_
        #         param.data = new_param.data
        #
        #     f1 = f_constraint()
        #     f1_grads = torch.autograd.grad(f1, params, create_graph=True)
        #     flat_grad_kl_1 = torch.cat([f1_grad.view(-1) for f1_grad in f1_grads])
        #
        #     for prev, cur in zip(prev_params, params):
        #         cur.data = prev.data
        #
        #     return (flat_grad_kl_1 - flat_grad_kl) / epsilon

        g = flat_loss_grads.clone().detach()
        print(g)

        m_vector = f_Ax(g)
        m_tmp = m_vector.clone().detach()
        print(m_tmp)

        m = f_Ax(m_tmp)
        m_clone = m.clone().detach()
        print(m_clone)

        g_cube = f_Ax(m_clone)
        g_cube = g_cube.clone().detach()
        print(g_cube)

        # G = torch.tensor([[torch.dot(g, m_tmp), torch.dot(g, m_clone)], [torch.dot(m_tmp, m_tmp), torch.dot(m_tmp, m_clone)]], requires_grad=False)
        # c = torch.tensor([torch.dot(g, g), torch.dot(g, m_tmp)], requires_grad=False)

        c = torch.tensor([torch.dot(g, g), torch.dot(g, m_tmp), torch.dot(g, m_clone)], requires_grad=False)
        print(c)
        g_ = torch.unsqueeze(g, dim=0)
        g1_ = torch.unsqueeze(m_tmp, dim=0)
        g2_ = torch.unsqueeze(m_clone, dim=0)
        left = torch.cat((g_, g1_, g2_), axis=0)
        g3_ = torch.unsqueeze(g_cube, dim=0)
        right = torch.cat((g1_.t(), g2_.t(), g3_.t()), axis=1)
        G = torch.matmul(left, right)



        eig, _ = torch.eig(G)
        print(eig)
        print('----------------------------------------------------')
        rho = eig.min()
        if rho <= 0:
            G = G - (rho - 1e-8) * torch.eye(3, 3)
            eig, _ = torch.eig(G)
            print(eig)
        G = G.clone().detach()
        print(G)

        inverse = torch.pinverse(G)
        print(inverse)

        x = inverse @ c
        print(x)

        print('xTGx is:')
        print(torch.dot(x, G @ x))

        step_size = np.sqrt(2 * radius * (1. / (torch.dot(x, G @ x) + 1e-8)))
        print(step_size)

        if torch.isinf(step_size).sum() or torch.isnan(step_size).sum():
            print('inf or nan stepsize')

        alpha = step_size * x

        step_dir = alpha[0] * g + alpha[1] * m_tmp + alpha[2] * m_clone
        print(step_dir)

        param_shapes = [p.shape or torch.Size([1]) for p in params]
        step_dir = unflatten_tensors(step_dir, param_shapes)
        assert len(step_dir) == len(params)

        prev_params = [p.clone() for p in params]
        loss_before = f_loss()

        print('loss before mean is')
        print(loss_before)

        # ratio_list = 0.9 ** np.arange(10) 
        ratio_list = [1, ]

        for ratio in ratio_list:
            for step, prev_param, param in zip(step_dir, prev_params, params):
                new_param = prev_param.data + ratio * step
                param.data = new_param.data
            loss = f_loss()
            print(loss)
            constraint_val = f_constraint()
            print(constraint_val)

            if loss > loss_before and constraint_val <= radius:
                break

        # if loss <= loss_before or constraint_val > radius:
        #     for prev, cur in zip(prev_params, params):
        #         cur.data = prev.data
        #     print('reject!')
