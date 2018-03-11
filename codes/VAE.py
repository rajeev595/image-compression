import numpy as np
import tensorflow as tf

class VariationalAutoencoder:
"""
    Variational Autoencoder (VAE)
    Args:
        inputs:
            hidden_layer_sizes: List containing the sizes of all hidden layers including the input
                                for example: [784, 500, 500] for MNIST dataset.
            latent_dim: Dimension of the latent space.
            learning_rate: learning rate for the SGD optimization.
            batch_size: batch size used for mini batch training.
        outputs:
            cost: loss obtained for the respective batch size.
"""
    def __init__(self, hidden_layer_sizes, latent_dim, learning_rate=0.001, batch_size=128):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.latent_dim = latent_dim
        self.learning_rate = learning_rate
        self.batch_size = batch_size
    # Tensor flow graph input
        self.x = tf.placeholder(tf.float32, [None, hidden_layer_sizes[0]])
    # The autoencoder model
        self.model()
    # Optimizer and loss function
        self.loss_optimizer()
    # Initialize the tensorflow computaional graph
        init = tf.global_variables_initializer()
    # Launch a session
        sess = tf.InteractiveSession()
        self.sess.run(init)
        
    def model(self):
    # Initialize the parameters of encoder
        weights, biases = self.initialize_enc_weights()
    # Encoder
        self.z_mean, self.z_log_sigma = self.encoder(self.x, weights, biases)
    # Sampling
        self.z = self.sampling(self.z_mean, self.z_log_sigma)
    # Initialize the parameters of decoder
        weights, biases = self.initialize_dec_weights()
    # Decoder
        self.recon = self.decoder(self.z, weights, biases)
        
    def initialize_enc_weights(self):
    # Encoder Weights
        weights = {}
        biases = {}
    # Hidden layer weights and biases
        for i in range(len(self.hidden_layer_sizes)-1):
            fan_in = self.hidden_layer_sizes[i]
            fan_out = self.hidden_layer_sizes[i+1]
            low = -np.sqrt(6.0/(fan_in + fan_out))
            high = np.sqrt(6.0/(fan_in + fan_out))
            weights[i] = tf.Variable(tf.random_uniform((fan_in, fan_out),
                                                       minval=low,
                                                       maxval=high,
                                                       dtype=tf.float32)
                                    )
            biases[i] = tf.Variable(tf.zeros([fan_out]), dtype=tf.float32)
    # mu and sigma layer weights
        fan_in = self.hidden_layer_sizes[-1]
        fan_out = self.latent_dim
        low = -np.sqrt(6.0/(fan_in + fan_out))
        high = np.sqrt(6.0/(fan_in + fan_out))
        weights['mu'] = tf.Variable(tf.random_uniform((fan_in, fan_out),
                                                      minval=low,
                                                      maxval=high,
                                                      dtype=tf.float32)
                                   )
        biases['mu'] = tf.Variable([fan_out], dtype=tf.float32)
        weights['sigma'] = tf.Variable(tf.random_uniform((fan_in, fan_out),
                                                         minval=low,
                                                         maxval=high,
                                                         dtype=tf.float32)
                                      )
        biases['sigma'] = tf.Variable([fan_out], dtype=tf.float32)
        
        return weights, biases
        
    def initialize_dec_weights(self):
    # Decoder Weights
        weights = {}
        biases = {}
    # mu and sigma layer weights
        fan_in = self.latent_dim
        fan_out = self.hidden_layer_sizes[-1]
        low = -np.sqrt(6.0/(fan_in + fan_out))
        high = np.sqrt(6.0/(fan_in + fan_out))
        weights['z'] = tf.Variable(tf.random_uniform((fan_in, fan_out),
                                                     minval=low,
                                                     maxval=high,
                                                     dtype=tf.float32)
                                  )
        biases['z'] = tf.Variable([fan_out], dtype=tf.float32)
    # Hidden layer weights and biases
        for i in range(len(self.hidden_layer_sizes)-1)[::-1]:
            fan_in = self.hidden_layer_sizes[i+1]
            fan_out = self.hidden_layer_sizes[i]
            low = -np.sqrt(6.0/(fan_in + fan_out))
            high = np.sqrt(6.0/(fan_in + fan_out))
            weights[i] = tf.Variable(tf.random_uniform((fan_in, fan_out),
                                                       minval=low,
                                                       maxval=high,
                                                       dtype=tf.float32)
                                    )
            biases[i] = tf.Variable(tf.zeros([fan_out]), dtype=tf.float32)

            return weights, biases
        
    def sampling(self, z_mean, z_log_sigma):
        epsilon = tf.random_normal((self.batch_size, self.latent_dim), 
                                   0, 1, dtype=tf.float32)
        z = tf.add(z_mean, tf.multiply(tf.exp(z_log_sigma), epsilon))
        return z
    
    def encoder(self, x, weights, biases):
        h = x
        for i in range(len(self.hidden_layer_sizes)-1):
            h = tf.add(tf.matmul(h, weights[i]), biases[i])
            h = tf.nn.relu(h)
        z_mean = tf.add(tf.matmul(h, weights['mu']), biases['mu'])
        z_log_sigma = tf.add(tf.matmul(h, weights['sigma']), biases['sigma'])
        
        return z_mean, z_log_sigma
    
    def decoder(self, z, weights, biases):
        h = tf.add(tf.matmul(z, weights['z']), biases['z'])
        for i in range(len(self.hidden_layer_sizes)-1)[::-1]:
            h = tf.add(tf.matmul(h, weights[i]), biases[i])
            h = tf.nn.relu(h)
        return h
    
    def loss_optimizer(self):
        recon_loss = tf.reduce_sum(tf.square(self.x - self.x_recon), axis=1)
        latent_loss = -0.5*tf.reduce_sum(1 + self.z_log_sigma
                                         - tf.square(self.z_mean)
                                         - tf.exp(self.z_log_sigma), axis=1)
        self.cost = tf.reduce_mean(recon_loss + latent_loss)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(cost)

    def fit(self, X):
        _, cost = self.sess.run(self.optimizer, self.cos, 
                                feed_dict = {self.x: X})
        return cost