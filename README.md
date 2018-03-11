### Image Compression

Image compression using Variational Autoencoder and Generative Adversarial Networks.

* There are three files utils.py, VAE.py, train.py. Following are the
   descriptions of each.
   * utils.py: This file contains the `next_batch` function customized to use it for CIFAR10 dataset.
   * VAE.py: A class for variational autoencoder is defined which will be used for training and evaluation purpose.
   * train.py: This is the function used for training the autoencoder. In the line no.6, equate x to the numpy data matrix of CIFAR10.
* For instructions on calling the function, please look at comments present inside the function.
