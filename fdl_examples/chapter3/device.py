import tensorflow as tf


with tf.device("/gpu:2"):
    a = tf.constant([1.0, 2.0, 3.0, 4.0], shape=[2, 2], name="a")
    b = tf.constant([1.0, 2.0], shape=[2, 1], name="b")
    c = tf.matmul(a, b)

sess = tf.Session(
    config=tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=True
    )
)

sess.run(c)


c = []

for d in ["/gpu:0", "/gpu:1"]:
    with tf.device(d):
        a = tf.constant([1.0, 2.0, 3.0, 4.0], shape=[2, 2], name="a")
        b = tf.constant([1.0, 2.0], shape=[2, 1], name="b")
        c.append(tf.matmul(a, b))

with tf.device("/cpu:0"):
    sum = tf.add_n(c)

sess = tf.Session(
    config=tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=True
    )
)
sess.run(sum)