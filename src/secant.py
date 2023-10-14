from typing import Callable, Literal

import numpy as np
import scipy.linalg as sla

from attrs import define, field
from attr import validators


@define(kw_only=True)
class SecantMethod:
    fun: Callable = field(validator=validators.is_callable())
    grad: Callable = field(validator=validators.is_callable())
    H_0: np.ndarray
    correction_method: Literal["CP1", "DFP"]
    alpha: float = field(default=0.5)
    beta: float = field(default=1)
    gamma: float = field(default=0.5)
    sigma: float = field(default=0.5)
    epsilon: float = field(default=1e-6)
    max_iterations: int = field(default=100, validator=validators.instance_of(int))

    @alpha.validator
    @gamma.validator
    @sigma.validator
    def check(self, attribute, value):
        if value < 0 or value > 1:
            raise ValueError(f"{attribute.name} must be in the open interval (0, 1)")

    @beta.validator
    @epsilon.validator
    @max_iterations.validator
    def check(self, attribute, value):
        if value <= 0:
            raise ValueError(f"{attribute.name} must be greater than zero")

    def __call__(self, x_0: np.ndarray):
        x: list[np.ndarray] = [x_0]
        H_k = self.H_0
        for k in range(self.max_iterations):
            x_k = x[k]

            # check for stationary point
            # using euclidian norm because it is useful later as well
            grad_k = self.grad(x_k)
            grad_2norm = sla.norm(grad_k, 2)
            if grad_2norm < self.epsilon:
                break

            # secant direction
            d_k = -H_k @ grad_k

            # check gamma
            d_k_2norm = sla.norm(d_k, 2)
            inner_prod = grad_k.T @ d_k
            if inner_prod > -self.gamma * grad_2norm * d_k_2norm:
                H_k = np.identity(len(H_k))
                d_k = -grad_k
                d_k_2norm = grad_2norm

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

            # determine next H_k
            y_k = self.grad(x[k + 1]) - grad_k
            z_k = H_k @ y_k
            if self.correction_method == "CP1":
                w_k = s_k - z_k
                inner_prod = w_k.T @ y_k
                if inner_prod > 0:
                    outer_prod = w_k @ w_k.T
                    H_k += outer_prod / inner_prod
            elif self.correction_method == "DFP":
                inner_prod = s_k.T @ y_k
                if inner_prod > 0:
                    outer_prod = s_k @ s_k.T
                    H_k += outer_prod / inner_prod
                    inner_prod = z_k.T @ y_k
                    outer_prod = z_k @ z_k.T
                    H_k -= outer_prod / inner_prod
            else:
                raise AttributeError(f"Invalid {self.correction_method=}")
        return x, k
