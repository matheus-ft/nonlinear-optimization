from typing import Callable

import numpy as np
import scipy.linalg as sla

from attrs import define, field
from attr import validators


@define(kw_only=True)
class GradientMethod:
    fun: Callable = field(validator=validators.is_callable())
    grad: Callable = field(validator=validators.is_callable())
    alpha: float = field(default=0.5)
    sigma: float = field(default=0.5)
    epsilon: float = field(default=1e-6)
    max_iterations: int = field(default=100, validator=validators.instance_of(int))

    @alpha.validator
    @sigma.validator
    def check(self, attribute, value):
        if value < 0 or value > 1:
            raise ValueError(f"{attribute.name} must be in the open interval (0, 1)")

    @epsilon.validator
    @max_iterations.validator
    def check(self, attribute, value):
        if value <= 0:
            raise ValueError(f"{attribute.name} must be greater than zero")

    def __call__(self, x_0: np.ndarray) -> tuple[list[np.ndarray], int]:
        x: list[np.ndarray] = [x_0]
        iterations: int = 0
        for k in range(int(self.max_iterations)):
            x_k = x[k]

            # check for stationary point
            d_k = -self.grad(x_k)
            grad_inf_norm = np.max(np.abs(d_k))
            if grad_inf_norm < self.epsilon:
                break

            # backtracking to satisfy Armijo
            t_k = 1
            f_k = self.fun(x_k)
            grad_norm_squared = -self.alpha * sla.norm(d_k, 2) ** 2
            while self.fun(x_k + t_k * d_k) > f_k + t_k * grad_norm_squared:
                t_k *= self.sigma
            s_k = t_k * d_k
            x.append(x_k + s_k)
            iterations = k
        return x, iterations
