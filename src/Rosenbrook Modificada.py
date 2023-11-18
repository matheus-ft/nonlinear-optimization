
from typing import Callable
import sympy as sp
from sympy import *
import math, numpy as np
import scipy.linalg as sla

from attrs import define, field
from attr import validators

alpha = 10**(-4) #Taxa de Decréscimo
Sigma = 0.5 # Fator de diminuição na busca unidimensional
E = 10**(-8) #Precisao
M = 10000 #Numero de Iterações

def array_to_vector(point):
    vector = np.expand_dims(point, axis=1)
    return vector



def rosenbrook(vector):
    n = vector.shape[0]
    if n % 2:
        print("n deve ser par")
    else:
        soma = 0
        for j in range(0,int(n / 2)):
            soma += 10 * ((vector[2*j + 1, 0] - (vector[2*j, 0])**2)**2) + ((vector[2*j, 0] - 1) ** 2)
        return soma

def gradient_rosenbrook(vector):
    n = vector.shape[0]
    grad = np.zeros_like(vector)
    for j in range(int(n/2)):
        #print("Jotaaa = %f", j)
        grad[2 * j, 0] = -40 * (vector[2 * j + 1, 0] * (vector[2 *j, 0]) -(vector[2 * j, 0])**3) + 2 * (vector[2*j, 0] - 1)
        grad[2 *j + 1, 0] = 20 * (vector[2*j + 1, 0] - (vector[2*j, 0])**2)
    return grad

def hess_rosenbrook(vector):
    n = vector.shape[0]
    hess = np.zeros((n, n))
    for j in range(int(n/2)):
        hess[2 * j, 2 * j] = -40 * (vector[2*j + 1, 0] - 3 * (vector[2*j, 0] ** 2))
        hess[2 * j, 2 * j + 1] = - 40 * vector[2*j, 0]
        hess[2 * j + 1,2 * j] = - 40 * vector[2*j, 0]
        hess[2 *j + 1, 2 *j + 1] = 20
    return hess




x = [1,3]
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
Norma = G.norm();
while (Norma >= E and k < M):
  if k == 0:
    t = 1
  else:
    normay = (y - Guarday).norm(2)
    normaG = (G - GuardaG).norm(2)
    t = normay/normaG
  #t = 1
  w = alpha*(G.dot(G))
  while (rosenbrook(y - t*G) > (rosenbrook(y) - t*w)):
    t = Sigma*t
  Guarday = y
  y = y - t*G
  #print(y)
  GuardaG = grad
  grad = gradient_rosenbrook(y)
  G = Matrix(grad)
  Norma = G.norm()
  k = k + 1

print('Total de Iterações: %d' % k)
print(rosenbrook(y))
print(y)