import numpy as np, tensorflow as tf ;

traind,testd,trainl,testl = np.load("mnist.npz").values() ;

data = tf.placeholder(tf.float64, shape=[None,28,28]) ;
targets = tf.placeholder(tf.float64, shape=[None,10]) ;

A0 = tf.reshape(data,[-1,784]) ;

L = 100 ;
max_iter = 200 ;
B = 100 ;

# affine layer 1
W1 = tf.Variable(np.random.uniform(-0.01,0.01,[784,L]), name="W1") ;
b1 = tf.Variable(np.random.uniform(-0.01,0.01,[1,L]), name="b1") ;
A1 = tf.matmul(A0, W1) + b1 ;

# ReLU layer 2
A2 = tf.nn.relu(A1) ;

# linear softmax MC: affine part
W3 = tf.Variable(np.random.uniform(-0.01,0.01,[L,10]), name="W3") ;
b3 = tf.Variable(np.random.uniform(-0.01,0.01,[1,10]), name="b3") ;
A3 = tf.matmul(A2,W3) + b3 ;

# in theory:
# A4 = tf.nn.softmax(A3) ;
# in practice:
loss = tf.nn.softmax_cross_entropy_with_logits_v2(targets, A3) ;

sess = tf.Session() ;
sess.run(tf.global_variables_initializer()) ;


ex10 = testd[0:10,:] ;

res10 = sess.run(A3,feed_dict={data:ex10}) ;
print(res10.shape) ;


# gradient taking is a symbolic operation, just for completeness
grads = tf.gradients(loss, W1) ;
np_grads = sess.run(grads,feed_dict={data:traind,targets:trainl}) ;
print(np_grads) ;

# better way:
#optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01) ;
optimizer = tf.train.AdamOptimizer(learning_rate=0.01) ;
gdOp = optimizer.minimize(loss) ;

# initialize Variables
sess.run(tf.global_variables_initializer()) ;


for it in range(0,max_iter):
  # MNIST is randomized already
  dataBatch = traind[it*B:(it+1)*B,:,:] ;
  targetBatch = trainl[it*B:(it+1)*B,:] ;

  sess.run(gdOp,feed_dict={data:dataBatch, targets:targetBatch}) ;
  lossValue = sess.run(tf.reduce_mean(loss),feed_dict={data:dataBatch, targets:targetBatch}) ;
  print("It=",it, "loss=",lossValue) ;

res10 = sess.run(A3,feed_dict={data:ex10}) ;
print(res10.argmax(axis=1),(testl[0:10,:]).argmax(axis=1)) ;

