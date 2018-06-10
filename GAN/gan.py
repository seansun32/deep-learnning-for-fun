import tensorflow as tf
import numpy as np



def leaky_relu(x,alpha=0.01):
    return tf.maximum(x,alpha*x)

def sample_noise(batch_size,dim):
    return tf.random_uniform([batch_size,dim],minval=-1,maxval=1)

def discriminator(x):
    with tf.variable_scope('discriminator'):
        fc1=tf.layers.dense(inputs=x,units=256,activation=leaky_relu)
        fc2=tf.layers.dense(inputs=fc1,units=256,activation=leaky_relu)
        logits=tf.layers.dense(fc2,1)

        return logits


def generator(z):
    with tf.variable_scope('generator'):
        fc1=tf.layers.dense(z,1024,activation=tf.nn.relu)
        fc2=tf.layers.dense(fc1,1024,activation=tf.nn.relu)
        img=tf.layers.dense(fc2,784,activation=tf.nn.tanh)

        return img

def gan_loss(logits_real,logits_fake):

    D_loss=None
    G_loss=None
    
    labels_real=tf.ones_like(logits_real)
    labels_fake=tf.zeros_like(logits_fake)

    real_image_loss=tf.nn.sigmoid_cross_entropy_with_logits(logits=logits_real,labels=labels_real)
    fake_image_loss=tf.nn.sigmoid_cross_entropy_with_logits(logits=logits_fake,labels=labels_fake)

    D_loss=real_image_loss+fake_image_loss
    D_loss=tf.reduce_mean(D_loss)

    G_loss=tf.nn.sigmoid_cross_entropy_with_logits(logits=logits_fake,labels=labels_real)
    G_loss=tf.reduce_mean(G_loss)
   
    return D_loss,G_loss


def get_optimizer(learning_rate=1e-3,beta1=0.5):
    
    D_solver=None
    G_solver=None

    D_solver=tf.train.AdamOptimizer(learning_rate=learning_rate,beta1=beta1)
    G_solver=tf.train.AdamOptimizer(learning_rate=learning_rate,beta1=beta1)

    return D_solver,G_solver



