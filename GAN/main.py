import tensorflow as tf
import numpy as np
from gan import *
from imgprocess import *


from tensorflow.examples.tutorials.mnist import input_data

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'


mnist = input_data.read_data_sets('MNIST_data', one_hot=False)


def run_gan(sess,G_sample,G_train_step,G_loss,D_train_step,D_loss,
        show_every=1000,print_every=50,batch_size=128,num_epoch=10):

    max_iter=int(mnist.train.num_examples*num_epoch/batch_size)

    for it in range(max_iter):
        if it % show_every==0:
            samples=sess.run(G_sample)
            fig=show_images(samples[:16])
            plt.show()

        minibatch,minibatch_y=mnist.train.next_batch(batch_size)
        
        _,D_loss_curr=sess.run([D_train_step,D_loss],feed_dict={x:minibatch})
        _,G_loss_curr=sess.run([G_train_step,D_loss],feed_dict={x:minibatch})

        #if it % print_every==0:
            #print('Iter:{},D:{:.4},G:{:.4}'.format(it,D_loss_curr,G_loss_curr))

    print('Final images')
    samples=sess.run(G_sample)

    fig=show_images(samples[:16])
    plt.show()


def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)
    return session



tf.reset_default_graph()
batch_size=128
noise_dim=96

x=tf.placeholder(tf.float32,[None,784])
z=sample_noise(batch_size,noise_dim)
G_sample=generator(z)

with tf.variable_scope("") as scope:
    logits_real=discriminator(preprocess_img(x))
    scope.reuse_variables()
    logits_fake=discriminator(G_sample)

D_vars=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,'discriminator')
G_vars=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,'generator')

#D_extra_step=tf.get_collection(tf.GraphKeys.UPDATE_OPS,'discriminator')
#G_extra_step=tf.get_collection(tf.GraphKeys.UPDATE_OPS,'generator')

D_optimizer,G_optimizer=get_optimizer()
D_loss,G_loss=gan_loss(logits_real,logits_fake)

D_train_step=D_optimizer.minimize(D_loss,var_list=D_vars)
G_train_step=G_optimizer.minimize(G_loss,var_list=G_vars)


with get_session() as sess:
    sess.run(tf.global_variables_initializer())
    run_gan(sess,G_sample,G_train_step,G_loss,D_train_step,D_loss)

