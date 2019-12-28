import os
import numpy as np
import tensorflow as tf

traind, testd, trainl, testl = np.load('mnist.npz') .values()
s = tf.Session()

data = tf.placeholder(tf.float32, [None, 28, 28], name="data")
label = tf.placeholder(tf.float32, [None, 10], name="label")

data1000 = data[1000, :, :]
label1000 = label[1000, :]

with tf.Session() as sess:
    fdict = {data: traind}
    npRes = sess.run(data1000, feed_dict=fdict)
    print("1) ", npRes)
    print("2) ", npRes.shape)
    print("3) ", npRes.argmax())


with tf.Session() as sess:
    fdict = {label: trainl}
    npRes = sess.run([tf.reduce_min(tf.argmax(label, axis=1)),
                      tf.reduce_max(tf.argmax(label, axis=1))], feed_dict=fdict)

    print("4) ", npRes)
