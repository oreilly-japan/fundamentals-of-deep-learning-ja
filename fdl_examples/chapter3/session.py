import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("data", one_hot=True)
minibatch_x, minibatch_y = mnist.train.next_batch(32)

x = tf.placeholder(tf.float32, name="x", shape=[None, 784])
W = tf.Variable(tf.random_uniform([784, 10], -1, 1), name="W")
b = tf.Variable(tf.zeros([10]), name="biases")

output = tf.matmul(x, W) + b

init_op = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init_op)
feed_dict = {x : minibatch_x}
sess.run(output, feed_dict=feed_dict)
