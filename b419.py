import numpy as np ;
import matplotlib.pyplot as plt ;
import math ;

traind, trainl, testd, testl = np.load("mnist.npz").values() ;
traind=traind.astype("float32") ;

#3! gradient descent
def f(x):
  return x[0]**2. + 2*x[1]**2. ;

def gradf(x):
  return np.array([2.,4.])*x ;

it = 0

x = np.array([1.,3.]) ;
eps = 0.1 ;

for it in range(1,5):
  x -= eps*gradf(x) ;
  print("Step ",it, ": x"+str(it),"=",x) ;



#4a
s500 = np.ravel(traind[500]).reshape(1,-1) ;
c500 = s500.transpose().dot(s500) ;
print (c500.shape) ;

#4b
batch = traind[0:1000].reshape(-1,28*28);
C = (batch[:,np.newaxis,:]*batch[:,:,np.newaxis]).mean(axis=0) ;
print ("C shape=", C.shape) ;

#4c
eigvals, eigvecs =  np.linalg.eigh(C) ;
fig,ax=plt.subplots(1,5) ;
ax[0].set_title("4") ;
for i in range(0,5):
  ax[i].imshow(eigvecs[:,-1-i].reshape(28,28))
plt.show() ;

# 5
# transforming data using eigenvectors
proj = np.dot(batch, eigvecs) ;

# set all but last 5 elements to 0
# play by replacing KEEP=5 by KEEP=50. As more components are kept,
# reconstruction gets better
KEEP = 5
proj[:,0:-KEEP] = 0

# reproject back to original space
reproj = np.dot(proj,eigvecs.T) ;

# visualize
fig,ax=plt.subplots(1,3) ;
ax[0].set_title("5")
for i in range(0,3):
  visImg = reproj[i].reshape(28,28) ;
  ax[i].imshow(visImg) ;

plt.show()




