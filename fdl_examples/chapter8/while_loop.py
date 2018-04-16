import tensorflow as tf


values = tf.random_normal([10])

index = tf.constant(0)
values_array = tf.TensorArray(tf.float32, 10)
cumsum_value = tf.constant(0.)
cumsum_array = tf.TensorArray(tf.float32, 10)
 
values_array = values_array.unstack(values)


def loop_body(index, values_array, cumsum_value, cumsum_array):
    current_value = values_array.read(index)
    cumsum_value += current_value
    cumsum_array = cumsum_array.write(index, cumsum_value)
    index += 1
 
    return (index, values_array, cumsum_value, cumsum_array)

_, _, _, final_cumsum = tf.while_loop(
    cond= lambda index, *_: index < 10,
    body= loop_body,
    loop_vars= (index, values_array, cumsum_value, cumsum_array)
)

cumsum_vector = final_cumsum.stack()

with tf.Session() as sess:
    print(sess.run(cumsum_vector))