import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import tensorflow as tf
import numpy as np




def preprocess_img(x):
    return 2*x-1.0

def deprocess_img(x):
    return (x+1.0)/2.0

def show_images(images):
    image=np.reshape(images,[images.shape[0],-1])
    sqrtn=int(np.ceil(np.sqrt(images.shape[0])))
    sqrtimg=int(np.ceil(np.sqrt(images.shape[1])))

    fig=plt.figure(figsize=(sqrtn,sqrtn))
    gs=gridspec.GridSpec(sqrtn,sqrtn)
    gs.update(wspace=0.05,hspace=0.05)

    for i,img in enumerate(images):
        ax=plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(img.reshape([sqrtimg,sqrtimg]))

    plt.show()
    return






