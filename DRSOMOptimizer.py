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

        g_vector = flat_loss_grads / torch.norm(flat_loss_grads)

        print("g vector is:")
        print(g_vector)

        Fg = f_Ax(g_vector)

        print("Fg is: ")
        print(Fg)

        m_vector = Fg

        m_vector = m_vector / torch.norm(m_vector)

        Fm = f_Ax(m_vector)

        print("m vector is:")
        print(m_vector)
        print("Fm is: ")
        print(Fm)

      

        gFg = torch.dot(g_vector, Fg)
        mFm = torch.dot(m_vector, Fm)
        mFg = torch.dot(m_vector, Fg)
        FFg = f_Ax(Fg)
        FFm = f_Ax(Fm)

        print("gFg is:")
        print(gFg)

        gg = torch.dot(g_vector, g_vector)
        mg = torch.dot(m_vector, g_vector)

        print('mg is:')
        print(mg)

        # c = torch.tensor([gg, mg, gFg], requires_grad=False)
        # g_ = torch.unsqueeze(g_vector, dim=0)
        # m_ = torch.unsqueeze(m_vector, dim=0)
        # Fg_ = torch.unsqueeze(Fg, dim=0)
        # Fm_ = torch.unsqueeze(Fm, dim=0)
        # left = torch.cat((g_, m_, Fg_), axis=0)
        # print(left)
        # FFg_ = torch.unsqueeze(FFg, dim=0)
        # FFm_ = torch.unsqueeze(FFm, dim=0)
        # right = torch.cat((Fg_.t(), Fm_.t(), FFg_.t()), axis=1)
        # print(right)
        # G = torch.matmul(left, right)

        # # gFFg = torch.dot(g_vector, FFg)
        # # gFFm = torch.dot(g_vector, FFm)
        # # mFFg = torch.dot(m_vector, FFg)
        # # mFFm = torch.dot(m_vector, FFm)
        # # FgFg = torch.dot(Fg, Fg)
        # # FgFFg = torch.dot(Fg, FFg)
        # # FgFFm = torch.dot(Fg, FFm)
        # # FmFg = torch.dot(Fm, Fg)
        # # FmFFm = torch.dot(Fm, FFm)
        # # G_2 = torch.tensor([[gFg, mFg, gFFg, gFFm], [mFg, mFm, mFFg, mFFm], [FgFg, mFFg, FgFFg, FgFFm] , [FmFg, mFFm, FgFFm, FmFFm ]], requires_grad=False)

        # # per = 0
        # # a = m_vector.shape[0]
        # # tmp = torch.zeros(a)
        # # if torch.equal(tmp, m_vector):
        # #     per = 1e-8
        # #     G = G + per * torch.ones_like(G)

        # print('G is:')
        # print(G)

        # print('eig is:')
        # eig, _ = torch.eig(G)
        # print(eig)

        # var = eig.min()

        # if var < 0:
        #     G = (-var) * torch.eye(3,3) + G 

        # eig, _ = torch.eig(G)
        # print(eig)


        
        # def _conjugate_gradient(G, b, cg_iters, residual_tol=1e-10):
        #     p = b.clone()
        #     r = b.clone()
        #     x = torch.zeros_like(b)
        #     rdotr = torch.dot(r, r)

        #     for _ in range(cg_iters):
        #         z = G@p
        #         v = rdotr / torch.dot(p, z)
        #         x += v * p
        #         r -= v * z
        #         newrdotr = torch.dot(r, r)
        #         mu = newrdotr / rdotr
        #         p = r + mu * p

        #         rdotr = newrdotr
        #         if rdotr < residual_tol:
        #             break
        #     return x       
        # x = _conjugate_gradient(G, c, cg_iters=10)
        # x = x.detach()

        # print('x is:')
        # print(x)
        # print('Gx is: ')
        # print(G@x)
        # print('xTGx is:')
        # print(torch.dot(x, G@x))

        # alpha = np.sqrt( 2 * radius * (1. / torch.dot(x, G @ x) ) ) * x


        # ---------------------------------
        # Two directions
        per = 0
        a = m_vector.shape[0]
        tmp = torch.zeros(a)
        if torch.equal(tmp, m_vector):
            per = 1e-8
        G = torch.tensor([[gFg, mFg + per], [mFg + per, mFm + per]], requires_grad=False)
        print("G is:")
        print(G)

        print('eig is:')
        eigen, _ = torch.eig(G)
        print(eigen)

        if eigen.min() < 0:
            print("indefinite")


        c = torch.tensor([gg, mg], requires_grad=False)
        # coff = 1. / (G[0][0] * G[1][1] - G[0][1] * G[1][0])
        # inverse = coff * torch.tensor([[G[1][1], (-1) * (mFg + per)], [(-1) * (mFg + per), G[0][0]]], requires_grad=False)

        inverse = torch.pinverse(G)

        print('inverse is:')
        print(inverse)
        
        x = inverse @ c
        
        print("x is:")
        print(x)

        print("xTGx is: ")
        print(torch.dot(x, G@x))

        step_size = np.sqrt(2 * radius * (1. / ( torch.dot(x, G @ x) )))

        

        flag = 0
        if torch.isinf(step_size).sum() or torch.isnan(step_size).sum() :
            flag = 1
            step_size = 1
            print('inf or nan stepsize')

        print('step size is:')
        print(step_size)

        alpha = step_size * x

        step_dir = alpha[0] * g_vector + alpha[1] * m_vector

        param_shapes = [p.shape or torch.Size([1]) for p in params]
        step_dir = unflatten_tensors(step_dir, param_shapes)

        prev_params = [p.clone() for p in params]
        loss_before = f_loss()

        print('loss before mean is')
        print(loss_before.mean())

        if flag:
            ratio_list = 0.8 ** np.arange(15) * 10
        else:
            ratio_list = [1,]

        for ratio in ratio_list:
            for step, prev_param, param in zip(step_dir, prev_params, params):
                new_param = prev_param.data + ratio * step
                param.data = new_param.data

            loss = f_loss()
            constraint_val = f_constraint()

            print('constraint value is: ')
            print(constraint_val)

            print('loss mean is')
            print(loss.mean())

            if loss.mean() > loss_before.mean() and constraint_val < radius:
                print('yes!')
                break

        # return alpha, g_vector, m_vector
