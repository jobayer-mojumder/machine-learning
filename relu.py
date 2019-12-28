import numpy as np
import math

def relu(x):
   return np.where(x<0,0,x)

a1 = np.array([1, 0, -1])
a2 = np.array([1, 0, 10])
a3 = np.array([-1, 0, -10])

print("Relu:")
print(relu(a1))
print(relu(a2))
print(relu(a3))

def affine(x):
    w = np.array([[2, 0, 0],[0, 2, 0]])
    b = np.array([1, -1, 0])
    y = np.add(np.dot(x,w), b)
    return y

x1 = np.array([[1, 1],[0, -1]])
x2 = np.array([[1, 0],[1, -1]])
x3 = np.array([[1, 1],[1, 1]])

print("Affine Layer:")

print(affine(x1))
print(affine(x2))
print(affine(x3))


def Softmax(x):
  e = np.exp(x)
  return e/e.sum()

def DNN(x):
    y = affine(x)
    y = relu(y)
    y = affine(y)
    y = Softmax(y)
    return y

x1 = np.array([[1, 1],[0, -1]])
#print(DNN(x1))