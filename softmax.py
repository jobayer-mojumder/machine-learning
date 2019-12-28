import numpy as np ;
import matplotlib.pyplot as plt ;
import math ;

traind, trainl, testd, testl = np.load("mnist.npz").values() ;
traind=traind.astype("float32") ;

def softmax(x):
    e_x = np.exp(x)/np.sum(np.exp(x))
    return e_x

a_1 = np.array([-1, -1, 5])
a_2 = np.array([1, 1, 2])
print("Softmax0: " + str(softmax(a_1)))
print("Softmax1: " + str(softmax(a_2)))


def entropy(y):
    t = np.array([0, 0, 1])
    ce = -1*(np.sum(t*np.log(y)))
    return ce


y0 = np.array([0.1, 0.1, 0.8])
y1 = np.array([0.3, 0.3, 0.4])
y2 = np.array([0.8, 0.1, 0.1])

print("CE0: " + str(entropy(y0)))
print("CE1: " + str(entropy(y1)))
print("CE2: " + str(entropy(y2)))
