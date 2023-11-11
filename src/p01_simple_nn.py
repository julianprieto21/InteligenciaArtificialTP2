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


def foward_primer_capa(x):
    W = np.random.randn(3, 2)  # 3x2
    b = np.random.randn(3)  # 3x1
    Z = W.dot(x) + b  # 3x1
    A = sigmoid(Z)  # 3x1
    return A


def foward_segunda_capa(A1):
    W = np.random.randn(1, 3)  # 1x3
    b = np.random.randn(1)  # 1x1
    Z = W.dot(A1) + b  # 1x1
    A = sigmoid(Z)  # 1x1
    return A


# test
# x = np.random.randn(2)  # 2x1

# A1 = foward_primer_capa(x)
# A2 = foward_segunda_capa(A1)

# print(A2)
