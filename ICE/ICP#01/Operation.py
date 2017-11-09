# Library
from __future__ import print_function
import tensorflow as tf

# tf Graph input
x = tf.placeholder(tf.int32)
y = tf.placeholder(tf.int32)
z = tf.placeholder(tf.int32)

# Operations for square, add and multiply
squar = tf.square(x)
add = tf.add(squar, y)
mul = tf.multiply(add, z)

# Run tensorflow session to calculate
with tf.Session() as sess:
    # Run every operation with variable input
    print("Result: %i" % sess.run(mul, feed_dict={x: 2, y: 3, z:5}))