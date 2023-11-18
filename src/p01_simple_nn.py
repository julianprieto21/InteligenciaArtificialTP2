import numpy as np
import math as m

#
# Primer Funcion:
#   Z_1 = w_1 * x + b_1
#   A_1 = sigmoide(Z_1)
#
# Segunda Funcion:
#   Z_2 = w_2 * x + b_2
#   A_2 = sigmoide(Z_2) = y
#


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def escalonada(x):
    return np.where(x >= 0, 1, 0)

def foward_primer_capa(x):
    W = np.array([[-1, 0], [0, -1], [1, 1]])  # 3x2
    b = np.array([[0.5], [0.5], [-4]])  # 3x1
    # print(W.shape, b.shape)
    Z = W.dot(x) + b  # 3x1
    # print(Z)
    A = escalonada(Z)  # 3x1
    # print(A)
    return A


def foward_segunda_capa(A1):
    W = np.array([1, 1, 4])  # 1x3
    b = np.array([-1])  # 1x1
    # print(W.shape, b.shape)
    Z = W.dot(A1) + b  # 1x1
    A = escalonada(Z)  # 1x1
    print(A)
    # print(Z.shape, A.shape)
    return A


# test
x = np.array([1,2.5]).reshape(2,1)  # 2x1

A1 = foward_primer_capa(x)
A2 = foward_segunda_capa(A1)
