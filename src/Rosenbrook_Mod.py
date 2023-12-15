from sympy import Matrix
import numpy as np

from functions import rosenbrook, gradient_rosenbrook, hess_rosenbrook, array_to_vector


alpha = 10 ** (-4)  # Taxa de Decréscimo
Sigma = 0.5  # Fator de diminuição na busca unidimensional
E = 10 ** (-8)  # Precisao
M = 10000  # Numero de Iterações

x = [1, 3]
y = array_to_vector(x)

fun = rosenbrook(y)
grad = gradient_rosenbrook(y)
hess = hess_rosenbrook(y)

print(fun)
print(grad)
print(hess)

y = array_to_vector(x)
k = 0
G = Matrix(grad)
Norma = G.norm()
while Norma >= E and k < M:
    if k == 0:
        t = 1
    else:
        normay = (y - Guarday).norm(2)
        normaG = (G - GuardaG).norm(2)
        t = normay / normaG
    # t = 1
    w = alpha * (G.dot(G))
    while rosenbrook(y - t * G) > (rosenbrook(y) - t * w):
        t = Sigma * t
    Guarday = y
    y = y - t * G
    # print(y)
    GuardaG = grad
    grad = gradient_rosenbrook(y)
    G = Matrix(grad)
    Norma = G.norm()
    k = k + 1

print("Total de Iterações: %d" % k)
print(rosenbrook(y))
print(y)
