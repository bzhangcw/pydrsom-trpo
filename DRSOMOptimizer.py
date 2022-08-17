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

    def compute_alpha(self, g_vector, m_vector, f_constraint):
        params = []
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    params.append(p)

        f_Ax = _build_hessian_vector_product(f_constraint, params,
                                             self._hvp_reg_coeff)

        Fg = f_Ax(g_vector)

        Fm = f_Ax(m_vector.detach())

        gFg = torch.dot(g_vector, Fg)
        mFm = torch.dot(m_vector, Fm)
        mFg = torch.dot(m_vector, Fg)

        gg = torch.dot(g_vector, g_vector)
        mg = torch.dot(m_vector, g_vector)

        per = 0

        print(m_vector.shape)

        a = m_vector.shape[0]

        tmp = torch.zeros(a)

        if torch.equal(tmp, m_vector):
            per = 1e-8

        G = torch.tensor([[gFg, mFg+per], [mFg+per, mFm + per]], requires_grad=False)
        c = torch.tensor([gg, mg], requires_grad=False)

        coff = 1. / (G[0][0] * G[1][1] - G[0][1] * G[1][0])

        inverse = coff * torch.tensor([[mFm+per, (-1) * mFg + per], [(-1) * mFg + per, gFg]], requires_grad=False)

        x = inverse @ c


        alpha = np.sqrt(2 * self._max_constraint_value * (1. / (torch.dot(x, G @ x) + 1e-8))) * x


        return alpha
