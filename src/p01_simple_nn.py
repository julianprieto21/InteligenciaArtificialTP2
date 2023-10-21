import numpy as np
import math as m

W1 = np.random.randn(3, 2)  # 3xm
b1 = np.random.randn(3)  # 3x1
W2 = np.random.randn(2, 3)  # 2x3
b2 = np.random.randn(2)  # 2x1
W3 = np.random.randn(1, 2)  # 1x2
b3 = np.random.randn(1)  # 1x1


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def foward_propagation(X, W1, b1, W2, b2, W3, b3):
    Z1 = np.dot(W1, X) + b1  # 3x1
    A1 = sigmoid(Z1)  # 3x1
    Z2 = np.dot(W2, A1) + b2  # 2x1
    A2 = sigmoid(Z2)  # 2x1
    Z3 = np.dot(W3, A2) + b3  # 1x1
    A3 = sigmoid(Z3)  # 1x1 = y
    return Z1, A1, Z2, A2, Z3, A3


def back_propagation(X, W1, b1, W2, b2, W3, b3, A1, A2, A3, a=1):
    c = 0
    print(
        f"""
      'W1': {W1.shape}
      'W2': {W2.shape}
      'W3': {W3.shape}
      'A1': {A1.shape}
      'A2': {A2.shape}  
      'A3': {A3.shape}
"""
    )
    while c == 0:
        # Calculo de derivadas
        _dW3 = np.dot(A3, (1 - A3))  # 1x1
        print(_dW3, _dW3.shape)
        dW3 = np.dot(A2, _dW3)  # 2x1
        print(dW3.shape)
        _dW2 = np.dot(np.dot(_dW3, W3), (A2 - A2**2))  # ?
        print(_dW2.shape)
        dW2 = np.dot(A1 @ _dW2)  #
        print(dW2.shape)
        # dW1 = _dW2 @ W2 @ A1 @ (1 - A1) @ X

        # Descenso por gradiente
        # W3 = W3 - a * dW3
        # W2 = W2 - a * dW2
        # W1 = W1 - a * dW1

        c += 1
    return W1, W2, W3


Z1, A1, Z2, A2, Z3, A3 = foward_propagation(np.array([1, 0]), W1, b1, W2, b2, W3, b3)
print(
    f"""
      'Z1': {Z1}
      'A1': {A1}
      'Z2': {Z2}
      'A2': {A2}
      'A2': {A2}
      'Z3': {Z3}
      'A3': {A3}    
  """
)
print(back_propagation(np.array([0, 1]), W1, b1, W2, b2, W3, b3, A1, A2, A3))
