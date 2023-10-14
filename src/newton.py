from typing import Callable

import numpy as np
import scipy.linalg as sla
from scipy.linalg import LinAlgError

from attrs import define, field
from attr import validators


@define(kw_only=True)
class NewtonMethod:
    fun: Callable = field(validator=validators.is_callable())
    grad: Callable = field(validator=validators.is_callable())
    hess: Callable = field(validator=validators.is_callable())
    alpha: float = field(default=0.5)
    beta: float = field(default=1)
    gamma: float = field(default=0.5)
    sigma: float = field(default=0.5)
    rho: float = field(default=1)
    epsilon: float = field(default=1e-6)
    max_iterations: int = field(default=100, validator=validators.instance_of(int))

    @alpha.validator
    @gamma.validator
    @sigma.validator
    def check(self, attribute, value):
        if value < 0 or value > 1:
            raise ValueError(f"{attribute.name} must be in the open interval (0, 1)")

    @beta.validator
    @rho.validator
    @epsilon.validator
    @max_iterations.validator
    def check(self, attribute, value):
        if value <= 0:
            raise ValueError(f"{attribute.name} must be greater than zero")

    def _direction(self, hess_k, grad_k):
        try:
            d_k = sla.solve(hess_k, -grad_k, assume_a="sym")
        except LinAlgError:  # hess is singular
            d_k = None
        except ValueError:
            raise ValueError("Hessian matrix provided is not square")
        return d_k

    def __call__(self, x_0: np.ndarray):
        x: list[np.ndarray] = [x_0]
        for k in range(self.max_iterations):
            x_k = x[k]

            # check for stationary point
            # using euclidian norm because it is useful later as well
            grad_k = self.grad(x_k)
            grad_2norm = sla.norm(grad_k, 2)
            if grad_2norm < self.epsilon:
                break

            # Newton's direction (with globalization) and check gamma
            mu = 0
            hess_k = self.hess(x_k)
            A = hess_k
            d_k = None
            d_k_2norm = inner_prod = 0
            while d_k is None or inner_prod > -self.gamma * grad_2norm * d_k_2norm:
                d_k = self._direction(A, grad_k)
                mu = max(2 * mu, self.rho)
                A = hess_k + mu * np.identity(len(hess_k))
                if d_k is not None:
                    d_k_2norm = sla.norm(d_k, 2)
                    inner_prod = grad_k.T @ d_k

            # check beta
            if d_k_2norm < self.beta * grad_2norm:
                d_k *= self.beta * grad_2norm / d_k_2norm

            # backtracking to satisfy Armijo
            t_k = 1
            f_k = self.fun(x_k)
            grad_norm_squared = -self.alpha * sla.norm(d_k, 2) ** 2
            while self.fun(x_k + t_k * d_k) > f_k + t_k * grad_norm_squared:
                t_k *= self.sigma
            s_k = t_k * d_k
            x.append(x_k + s_k)
        return x, k
