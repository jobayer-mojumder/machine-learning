import numpy as np, tensorflow as tf ;

# exercise 1
max_iter = 20 ;
W = tf.Variable(np.ones([2])*2.) ;
loss = tf.reduce_sum(W*W) ;

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.2) ;
gdOp = optimizer.minimize(loss) ;

sess = tf.Session() ;
# initialize Variables
sess.run(tf.global_variables_initializer()) ;


print("Exercise 01") ;
for it in range(0,max_iter):
  sess.run(gdOp) ;
  lossValue = sess.run(tf.reduce_sum(loss)) ;
  weight = sess.run(W);
  print("It=",it, "loss=",lossValue, "W=", weight) ;

