import numpy as np


def array_to_vector(point):
    vector = np.expand_dims(point, axis=1)
    return vector


"""
Implementation of the requested functions, along with their respective gradients and hessians
Input: vector - np.ndarray: the point at which the function must be evaluated
Output: value of the function (float), value of the gradient (column matrix) or the value of the Hessian (2d matrix)
"""

# Quadratic


def quadratic(vector):
    n = vector.shape[0]
    soma = 0
    for i in range(n):
        soma += (i + 1) * (vector[i, 0] ** 2)
    return soma


def gradient_quadratic(vector):
    n = vector.shape[0]
    grad = np.zeros_like(vector)
    for i in range(n):
        grad[i, 0] = 2 * (i + 1) * vector[i, 0]
    return grad


def hess_quadratic(vector):
    n = vector.shape[0]
    hess = np.zeros((n, n))
    for i in range(n):
        hess[i, i] = 2 * (i + 1)
    return hess

# Rosenbrook


def rosenbrook(vector):
    n = vector.shape[0]
    if n % 2:
        print("n deve ser par")
    else:
        soma = 0
        for j in range(int(n / 2) - 1):
            soma += 10 * \
                ((vector[2*(j + 1), 0] - (vector[2*j + 1, 0] ** 2))) + \
                ((vector[2*j + 1, 0] - 1) ** 2)
        return soma


def gradient_rosenbrook(vector):
    n = vector.shape[0]
    grad = np.zeros_like(vector)
    for j in range(int(n/2) - 1):
        grad[2 * j + 1, 0] = -40 * vector[2 * j + 1, 0] * \
            (vector[2 * (j + 1), 0]) - (vector[2 * j + 1, 0] ** 2) + \
            2 * (vector[2*j + 1, 0] - 1)
        grad[2 * (j + 1), 0] = 20 * \
            (vector[2*(j + 1), 0] - (vector[2*j + 1, 0]) ** 2)
    return grad


def hess_rosenbrook(vector):
    n = vector.shape[0]
    hess = np.zeros((n, n))
    for j in range(int(n/2) - 1):
        hess[2 * j + 1, 2 * j + 1] = -40 * \
            (vector[2*(j + 1), 0] - 3 * (vector[2*j + 1, 0] ** 2))
        hess[2 * j + 1, 2 * (j + 1)] = - 40 * vector[2*j + 1, 0]
        hess[2 * (j + 1), 2 * j + 1] = - 40 * vector[2*j + 1, 0]
        hess[2 * (j + 1), 2 * (j + 1)] = 20
    return hess


x = [1, 1, 1, 1]
y = array_to_vector(x)
fun = rosenbrook(y)
grad = gradient_rosenbrook(y)
hess = hess_rosenbrook(y)
print(fun)
print(grad)
print(hess)
