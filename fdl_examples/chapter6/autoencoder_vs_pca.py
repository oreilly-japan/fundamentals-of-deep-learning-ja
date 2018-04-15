import argparse
from sklearn import decomposition
import tensorflow as tf
from matplotlib import pyplot as plt
from fdl_examples.datatools import input_data
from . import autoencoder_mnist as ae


def scatter(codes, labels):
    colors = [
        ('#27ae60', 'o'),
        ('#2980b9', 'o'),
        ('#8e44ad', 'o'),
        ('#f39c12', 'o'),
        ('#c0392b', 'o'),
        ('#27ae60', 'x'),
        ('#2980b9', 'x'),
        ('#8e44ad', 'x'),
        ('#c0392b', 'x'),
        ('#f39c12', 'x'),
    ]
    for num in range(10):
        plt.scatter([
                codes[:,0][i] for i in range(len(labels))
                if labels[i] == num
            ],
            [
                codes[:,1][i] for i in range(len(labels))
                if labels[i] == num
            ],
            7,
            label=str(num),
            color=colors[num][0],
            marker=colors[num][1]
        )
    plt.legend()
    plt.show()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Test various optimization strategies'
    )
    parser.add_argument('n_code', type=int)
    args = parser.parse_args()
    n_code = args.n_code
    log_dir = "mnist_autoencoder_hidden={}_logs/".format(n_code)
    ckpt = tf.train.get_checkpoint_state(log_dir)
    savepath = ckpt.model_checkpoint_path
    print("Use savepath: {}".format(savepath))
    print("\nPULLING UP MNIST DATA")
    mnist = input_data.read_data_sets("data/", one_hot=False)
    print(mnist.test.labels)

    # Apply PCA
    print("\nSTARTING PCA")
    pca = decomposition.PCA(n_components=n_code)
    pca.fit(mnist.train.images)
    print("\nGENERATING PCA CODES AND RECONSTRUCTION")
    pca_codes = pca.transform(mnist.test.images)

    with tf.Graph().as_default():
        with tf.variable_scope("autoencoder_model"):
            x = tf.placeholder("float", [None, 784])
            phase_train = tf.placeholder(tf.bool)
            code = ae.encoder(x, n_code, phase_train)
            output = ae.decoder(code, n_code, phase_train)
            cost, train_summary_op = ae.loss(output, x)
            global_step = tf.Variable(
                0,
                name='global_step',
                trainable=False
            )

            train_op = ae.training(cost, global_step)
            eval_op, in_im_op, out_im_op, val_summary_op = ae.evaluate(
                output,
                x
            )
            
            sess = tf.Session()
            saver = tf.train.Saver()
            
            saver.restore(sess, savepath)

            # Apply AutoEncoder
            print("\nSTARTING AUTOENCODER")
            ae_codes= sess.run(
                code,
                feed_dict={
                    x: mnist.test.images,
                    phase_train: True
                }
            )

            scatter(ae_codes, mnist.test.labels)
            scatter(pca_codes, mnist.test.labels)
