from VAE import VariationalAutoencoder
from utils import cifar10_next_batch

# CIFAR10 samples
n_samples = 50000
x = 'cifar10.data'        # Numpy data matrix

def train(hidden_layer_sizes, latent_dim, learning_rate=0.01,
          batch_size=128, training_epochs=10, display_step=5):
"""
    Function for training the Autoencoder
    Args:
        inputs:
            hidden_layer_sizes: List containing the sizes of all hidden layers including the input
                                for example: [784, 500, 500] for MNIST dataset.
            latent_dim: Dimension of the latent space.
            learning_rate: learning rate for the SGD optimization.
            batch_size: batch size used for mini batch training.
            training_epochs: No. of training epochs.
            display_steps: Display for every (display_steps) no. of steps.
"""    
    vae = VariationalAutoencoder(hidden_layer_sizes,
                                 latent_dim,
                                 learning_rate,
                                 batch_size)
# Training Cycle
    for epoch in range(training_epochs):
        total_cost = 0.
        total_batches = int(n_samples/batch_size)
    # Loop over all batches
        for i in range(total_batches):
            batch_xs = cifar10_next_batch(batch_size, x)
            
        # Fit training using batch data
            cost = vae.fit(batch_xs)
        # Compute total loss
            total_loss += cost / n_samples * batch_size
            
    # Display
        if epoch % display_step == 0:
            print ("Epoch:", '%04d' % (epoch+1),
                   "cost=", "{:.9f}".format(total_cost))
    return vae