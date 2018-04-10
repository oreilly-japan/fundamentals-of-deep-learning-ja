import tensorflow as tf


def my_network(input):
    W_1 = tf.Variable(tf.random_uniform([784, 100], -1, 1), name="W_1")
    b_1 = tf.Variable(tf.zeros([100]), name="biases_1")
    output_1 = tf.matmul(input, W_1) + b_1

    W_2 = tf.Variable(tf.random_uniform([100, 50], -1, 1), name="W_2")
    b_2 = tf.Variable(tf.zeros([50]), name="biases_2")
    output_2 = tf.matmul(output_1, W_2) + b_2

    W_3 = tf.Variable(tf.random_uniform([50, 10], -1, 1),name="W_3")
    b_3 = tf.Variable(tf.zeros([10]), name="biases_3")
    output_3 = tf.matmul(output_2, W_3) + b_3

    # printing names
    print("Printing names of weight parameters")
    print(W_1.name, W_2.name, W_3.name)
    print("Printing names of bias parameters")
    print(b_1.name, b_2.name, b_3.name)

    return output_3


i_1 = tf.placeholder(tf.float32, [1000, 784], name="i_1")
my_network(i_1)

i_2 = tf.placeholder(tf.float32, [1000, 784], name="i_2")
my_network(i_2)
