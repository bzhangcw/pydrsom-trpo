from cmath import isnan
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
        return flat_output + reg_coeff * vector

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

    def compute_alpha(self, m, f_constraint, itr):
        params = []
        grads = []
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    params.append(p)
                    grads.append(p.grad.reshape(-1))
        flat_loss_grads = torch.cat(grads)

        g = flat_loss_grads.clone().detach()

        print(torch.isnan(g).sum())

        f_Ax = _build_hessian_vector_product(f_constraint, params,
                                             self._hvp_reg_coeff)

        tmp = torch.zeros_like(m)
        if torch.equal(tmp, m):
            per = 1e-8
            m = m + per * torch.ones_like(m)

        Fg = f_Ax(g)
        Fm = f_Ax(m)

        print('g is: ')
        print(g)
        print('-------------------------------')

        print('m is: ')
        print(m)
        print('-------------------------------')

        print('Fg is: ')
        print(Fg)
        print('-------------------------------')

        print('Fm is: ')
        print(Fm)
        print('-------------------------------')

        gg = torch.dot(g, g)
        gm = torch.dot(g, m)
        gFg = torch.dot(g, Fg)
        gFm = torch.dot(g, Fm)
        FFg = f_Ax(Fg)
        FFm = f_Ax(Fm)

        c = torch.tensor([gg, gm, gFg, gFm], requires_grad=False)
        print('c is:')
        print(c)
        print('------------------------------')

        g_ = torch.unsqueeze(g, dim=0)
        m_ = torch.unsqueeze(m, dim=0)
        Fg_ = torch.unsqueeze(Fg, dim=0)
        Fm_ = torch.unsqueeze(Fm, dim=0)
        FFg_ = torch.unsqueeze(FFg, dim=0)
        FFm_ = torch.unsqueeze(FFm, dim=0)

        left = torch.cat((g_, m_, Fg_, Fm_), axis=0)
        right = torch.cat((Fg_.t(), Fm_.t(), FFg_.t(), FFm_.t()), axis=1)

        G = torch.matmul(left, right).detach()
        print('G is:')
        print(G)
        print('-----------------------------')

        eigen, _ = torch.eig(G)
        var = eigen.min()
        if var <= 0:
            G = G - (var - 1e-8) * torch.eye(4)
        print('eigen of G is:')
        print(eigen)
        print('-----------------------------')

        inverse = torch.pinverse(G)
        x = inverse @ c
        print("x is:")
        print(x)
        print('----------------------------')

        print('xTGx is: ')
        print(torch.dot(x, G @ x))
        print('----------------------------')

        alpha = np.sqrt(2 * self._max_constraint_value * (1. / (torch.dot(x, G @ x) + 1e-8))) * x


        if torch.isnan(alpha).sum():
            print('nan step size!')
            alpha = torch.ones(4)

        print('alpha is:')
        print(alpha)
        print('--------------------------')

        # mFm = torch.dot(m, Fm)
        # mFg = torch.dot(m, Fg)
        # G = torch.tensor([[gFg, mFg + per], [mFg + per, mFm + per]], requires_grad=False)
        # print("G is:")
        # print(G)
        # coff = 1. / (G[0][0] * G[1][1] - G[0][1] * G[1][0])
        # inverse = coff * torch.tensor([[G[1][1], (-1) * G[0][1]], [(-1) * G[1][0], G[0][0]]], requires_grad=False)

        return alpha, g, Fg, Fm
