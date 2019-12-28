'''
Created on Nov 1, 2018

@author: fdai0114
'''

 # getting mnist data into your Python programs:

# First: type

# wget www.gepperth.net/alexander/downloads/mnist.pkl.gz

# Then: paste this code at the beginning of your file!

import numpy as np, matplotlib.pyplot as plt ;
import numpy.random as npr ;

# on command prompt: type
# wget www.gepperth.net/alexander/downloads/mnist.npz
traind,testd,trainl,testl = np.load('mnist.npz') .values() ;


# 1a) 
print(traind.shape, "nr samples", traind.shape[0]) ;
print("sampe shape is ", traind.shape[1],"x",traind.shape[2]) ;
print("labels shape",trainl.shape) ;

# 1b) 
img1000 = traind[999] ;
print ("sample 1000 shape", img1000.shape) ;
print ("class of sample 1000", trainl[999].argmax())
plt.imshow(img1000) ;
plt.show();

# 1c)
classVector = trainl.argmax(axis=1) ;
print ("minmax class is ", classVector.min(), classVector.max()) ;

# 1d) 
samplesPerClass = trainl.sum(axis=0) ;
print ("samples per class", samplesPerClass) ;
print ("sampels of class", samplesPerClass[5]) ;

# 1e) 
sample10 = traind[9] ;
print ("min max of 10 is", sample10.min(), np.max(sample10)) ;

# 1f) 

# 1g) 
classVector = trainl.argmax(axis=1) ;
class4Mask = (classVector == 4) ;
traind_class4 = traind[class4Mask] ;
print("class 4 stack is of shaPE", traind_class4.shape) ;

# 1h) 
class9Mask = (classVector == 9) ;
class49Mask = np.logical_or(class4Mask, class9Mask) ;
traind_class49 = traind[class49Mask] ;
print("class 49 stack is of shaPE", traind_class49.shape) ;

# 1j)
#traind_inv = 1-traind ;
traind -= 1 ;
traind *= -1 ;
img1000inv = traind[999] ;
plt.imshow(img1000inv) ;
plt.show();

# 1j
indices = np.arange(0,traind.shape[0]) ;
npr.shuffle(indices) ;
thousandRandomIndices = indices[0:1000] ;
randomizedStack = traind[thousandRandomIndices, 0:14,0:14] ;
print(randomizedStack.shape) ;
plt.imshow(randomizedStack[200]) ;
plt.show();




