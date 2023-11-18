"""
Implementation of the requested functions, along with their respective gradients and hessians
Input: vector (np.ndarray): the point at which the function must be evaluated
Output: value of the function (float), value of the gradient (column matrix) or the value of the Hessian (2d matrix)
"""

import numpy as np


def array_to_vector(point: list):
    vector = np.expand_dims(point, axis=1)
    return vector


# Quadratic


def quadratic(vector: np.ndarray):
    n = vector.shape[0]
    soma = 0
    for i in range(n):
        soma += (i + 1) * (vector[i, 0] ** 2)
    return soma


def gradient_quadratic(vector: np.ndarray):
    n = vector.shape[0]
    grad = np.zeros_like(vector)
    for i in range(n):
        grad[i, 0] = 2 * (i + 1) * vector[i, 0]
    return grad


def hess_quadratic(vector: np.ndarray):
    n = vector.shape[0]
    hess = np.zeros((n, n))
    for i in range(n):
        hess[i, i] = 2 * (i + 1)
    return hess


# Rosenbrook


def rosenbrook(vector: np.ndarray):
    n = vector.shape[0]
    if n % 2 != 0:
        raise TypeError(f"Dimension {n=} must be an even number!")
    soma = 0
    for j in range(n // 2):
        soma += 10 * ((vector[2 * j + 1, 0] - (vector[2 * j, 0]) ** 2) ** 2) + (
            (vector[2 * j, 0] - 1) ** 2
        )
    return soma


def gradient_rosenbrook(vector: np.ndarray):
    n = vector.shape[0]
    if n % 2 != 0:
        raise TypeError(f"Dimension {n=} must be an even number!")
    grad = np.zeros_like(vector)
    for j in range(n // 2):
        grad[2 * j, 0] = -40 * (
            vector[2 * j + 1, 0] * (vector[2 * j, 0]) - (vector[2 * j, 0]) ** 3
        ) + 2 * (vector[2 * j, 0] - 1)
        grad[2 * j + 1, 0] = 20 * (vector[2 * j + 1, 0] - (vector[2 * j, 0]) ** 2)
    return grad


def hess_rosenbrook(vector: np.ndarray):
    n = vector.shape[0]
    if n % 2 != 0:
        raise TypeError(f"Dimension {n=} must be an even number!")
    hess = np.zeros((n, n))
    for j in range(n // 2):
        hess[2 * j, 2 * j] = -40 * (vector[2 * j + 1, 0] - 3 * (vector[2 * j, 0] ** 2))
        hess[2 * j, 2 * j + 1] = -40 * vector[2 * j, 0]
        hess[2 * j + 1, 2 * j] = -40 * vector[2 * j, 0]
        hess[2 * j + 1, 2 * j + 1] = 20
    return hess


# Styblinsky-Tang


def sty_tang(vector: np.ndarray):
    n = vector.shape[0]
    soma = 0
    for i in range(n):
        soma += vector[i, 0] ** 4 - 16 * vector[i, 0] ** 2 + 5 * vector[i, 0]
    return soma


def gradient_sty_tang(vector: np.ndarray):
    n = vector.shape[0]
    grad = np.zeros_like(vector)
    for i in range(int(n)):
        grad[i, 0] = 4 * vector[i, 0] ** 3 - 32 * vector[i, 0] + 5
    return grad


def hess_sty_tang(vector: np.ndarray):
    n = vector.shape[0]
    hess = np.zeros((n, n))
    for i in range(n):
        hess[i, i] = 12 * vector[i, 0] ** 2 - 32
    return hess


# Rastrigin


def rastrigin(vector: np.ndarray):
    n = vector.shape[0]
    soma = 0
    for i in range(n):
        soma += vector[i, 0] ** 2 - 10 * np.cos(2 * np.pi * vector[i, 0])
    return soma


def gradient_rastrigin(vector: np.ndarray):
    n = vector.shape[0]
    grad = np.zeros_like(vector)
    for i in range(int(n)):
        grad[i, 0] = 2 * vector[i, 0] + 20 * np.pi * np.sin(2 * np.pi * vector[i, 0])
    return grad


def hess_rastrigin(vector: np.ndarray):
    n = vector.shape[0]
    hess = np.zeros((n, n))
    for i in range(n):
        hess[i, i] = 2 + 40 * (np.pi) ** 2 * np.cos(2 * np.pi * vector[i, 0])
    return hess


# Tests

if __name__ == "__main__":
    x = [0, 0, 0]
    y = array_to_vector(x)
    fun = rastrigin(y)
    grad = gradient_rastrigin(y)
    hess = hess_rastrigin(y)
    print(fun)
    print(grad)
    print(hess)
