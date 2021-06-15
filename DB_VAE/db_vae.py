import functools
import tensorflow as tf
import numpy as np
from Utils.dataset_utils import preprocess_batch
from Settings.settings import *


# Logger for DB_VAE.
logger = logging.getLogger("DB_VAE")
logger.setLevel(LOG_LEVEL)


def make_vae_encoder(n_outputs: int = 1):
    """Generate a encoder of the variational autoencoder for image
    classification.

    Parameters
    ----------
    n_outputs : int, optional
        The number of outputs, by default 1.

    Returns
    -------
    tf.keras.Sequential
        The generated encoder network.
    """
    # base number of filters of the convolutional layers
    n_filters = 12

    # Define layers.
    BatchNormalization = tf.keras.layers.BatchNormalization
    Flatten = tf.keras.layers.Flatten
    Conv2D = functools.partial(
        tf.keras.layers.Conv2D, padding="same", activation=tf.nn.relu)
    Dense = functools.partial(tf.keras.layers.Dense, activation=tf.nn.relu)

    # Define the encoder network.
    encoder = tf.keras.Sequential([
        Conv2D(filters=1*n_filters, kernel_size=5, strides=2),
        BatchNormalization(),
        Conv2D(filters=2*n_filters, kernel_size=5, strides=2),
        BatchNormalization(),
        Conv2D(filters=4*n_filters, kernel_size=3, strides=2),
        BatchNormalization(),
        Conv2D(filters=6*n_filters, kernel_size=3, strides=2),
        BatchNormalization(),

        Flatten(),
        Dense(512),
        Dense(n_outputs, activation=None),
    ], name="VAE_encoder")
    return encoder


def make_vae_decoder():
    """Generate a decoder of the variational autoencoder for image
    classification.

    Returns
    -------
    tf.keras.Sequential
        The generated decoder network.
    """
    # base number of filters of the convolutional layers
    n_filters = 12

    # Define layers.
    # decompression of convoluted data by transposed convolution layer.
    Conv2DTranspose = functools.partial(
        tf.keras.layers.Conv2DTranspose, padding='same', activation='relu')
    Dense = functools.partial(tf.keras.layers.Dense, activation='relu')
    Reshape = tf.keras.layers.Reshape

    # Define the decoder network.
    decoder = tf.keras.Sequential([
        # Transform the sampled data to the output shape of the last
        # convolutional layer from the input.
        Dense(4*4*6*n_filters),
        Reshape(target_shape=(4, 4, 6*n_filters)),

        # Upscaling convolutional layers, symmetric to the encoder.
        Conv2DTranspose(filters=4*n_filters, kernel_size=3,  strides=2),
        Conv2DTranspose(filters=2*n_filters, kernel_size=3,  strides=2),
        Conv2DTranspose(filters=1*n_filters, kernel_size=5,  strides=2),

        # reconstruct images with 3 channels (RGB color model)
        Conv2DTranspose(filters=3, kernel_size=5,  strides=2)
    ], name="VAE_decoder")
    return decoder


def vae_loss_function(x: np.ndarray, x_reconstruction: np.ndarray,
                      z_mu: tf.Tensor, z_log_sigma_sq: tf.Tensor,
                      kl_weight: float = 0.0005):
    """Compute the loss of VAE defined by latent loss and reconstruction loss.

    Parameters
    ----------
    x : np.ndarray
        Input of the VAE.
    x_reconstruction : np.ndarray
        Output of the VAE, a reconstruction of the input based on decoding
        samples generated from the distribution parametrized by the latent
        variable.
    z_mu : tf.Tensor
        The mean parameter of the latent distribution.
    z_log_sigma_sq : tf.Tensor
        The log of the variance parameter of the latent distribution.
    kl_weight : float, optional
        Weight of the latent loss defined by KL-divergence, by default 0.0005.

    Returns
    -------
    np.ndarray
        The overall loss of the VAE.
    """
    # Latent loss is defined by the Kullback-Leibler (KL) divergence.
    # Proof: https://medium.com/retina-ai-health-inc/variational-inference-derivation-of-the-variational-autoencoder-vae-loss-function-a-true-story-3543a3dc67ee
    latent_loss = 0.5 * \
        tf.reduce_sum(tf.exp(z_log_sigma_sq) +
                      tf.square(z_mu) - 1 - z_log_sigma_sq, axis=1)

    # using mean to scalarize the loss, average over the height
    # width, and channel image dimensions. shape=(batch_size,)
    reconstruction_loss = tf.reduce_mean(
        tf.abs(x-x_reconstruction), axis=(1, 2, 3))

    vae_loss = kl_weight * latent_loss + reconstruction_loss
    return vae_loss


def debiasing_loss_function(x: np.ndarray,
                            x_reconstruction: np.ndarray,
                            y: np.ndarray,
                            y_logit: np.ndarray, z_mu: tf.Tensor,
                            z_log_sigma_sq: tf.Tensor):
    """Compute the overall loss of the VAE classification model.

    Parameters
    ----------
    x : np.ndarray
        Input of the VAE.
    x_reconstruction : np.ndarray
        Output of the VAE, a reconstruction of the input based on decoding
        samples generated from the distribution parametrized by the latent
        variable.
    y : np.ndarray
        Label of the input.
    y_logit : np.ndarray
        Encoder output of the logit of the classification.
    z_mu : tf.Tensor
        The mean of the normal distribution provided by the output of the
        encoder with shape (batch_size, self._latent_dim).
    z_log_sigma_sq : tf.Tensor
        The log of variance of the normal distribution provided by the
        output of the encoder with shape (batch_size, self._latent_dim).

    Returns
    -------
    tuple
        A tuple contains the total loss of the VAE and only the loss of the
        classification.
    """

    vae_loss = vae_loss_function(x, x_reconstruction, z_mu, z_log_sigma_sq)

    classification_loss = tf.nn.sigmoid_cross_entropy_with_logits(
        labels=y, logits=y_logit)

    # 1. for faces, 0. for non-faces
    face_indicator = tf.cast(tf.equal(y, 1), tf.float32)

    #  face_indicator * vae_loss removes the vae loss of
    total_loss = tf.reduce_mean(
        classification_loss + face_indicator * vae_loss)
    return total_loss, classification_loss


class DB_VAE(tf.keras.Model):
    """A debiasing variational autoencoder for image classification."""

    def __init__(self, latent_dim: int):
        """Initialize a VAE model.

        Parameters
        ----------
        latent_dim : int
            The dimension of the latent layer.
        """
        super().__init__()
        self._latent_dim = latent_dim
        # Means (mu) with shape (self._latent_dim, ),
        # log of variances (log_sigma_sq) with shape (self._latent_dim, ),
        # an extra one variable for image classification.
        encoder_out_dim = 2*self._latent_dim + 1

        # Make the encoder and the decoder of the VAE
        self._encoder = make_vae_encoder(encoder_out_dim)
        self._decoder = make_vae_decoder()

    def encode(self, x: np.ndarray):
        """Encode images <x> by using the VAE encoder.

        Parameters
        ----------
        x : np.ndarray
            Input image with shape (batch_size, height, width, 3).

        Returns
        -------
        tuple
            A tuple contains the logit of prediction result with shape
            (batch_size, 1), the mean parameter of the latent variable z in
            shape (batch_size, self._latent_dim) and the log of the variance
            parameter of the latent variable z with shape
            (batch_size, self._latent_dim).
        """
        encoder_output = self._encoder(x)
        # set first element of the output as prediction logit,
        # reshape it to (batch_size, 1)
        y_logit = tf.expand_dims(encoder_output[:, 0], -1)
        # mean of the latent variable z
        z_mu = encoder_output[:, 1:self._latent_dim+1]
        # log of variance of the latent variable z
        z_log_sigma_sq = encoder_output[:, 1:self._latent_dim+1]
        return y_logit, z_mu, z_log_sigma_sq

    def decode(self, z: np.ndarray):
        """Decode/Reconstruct the input <x> with the latent
        layer by using the VAE decoder.

        Parameters
        ----------
        z : ndarray
            Latent layer <z> with shape (batch_size, self._latent_dim)

        Returns
        -------
        ndarray
            Reconstructed input <x> with shape (batch_size, height, width, 3)
        """
        reconstruction = self._decoder(z)
        return reconstruction

    def sample_latent(self, z_mu: tf.Tensor, z_log_sigma_sq: tf.Tensor,
                      size: int = None):
        """Generate latent samples from the normal distribution with
        mean <z_mu> and log of variance <z_log_sigma_sq>. Assuming the latent
        variable is normally distributed.

        Parameters
        ----------
        z_mu : tf.Tensor
            The mean of the normal distribution provided by the output of the
            encoder with shape (batch_size, self._latent_dim).
        z_log_sigma_sq : tf.Tensor
            The log of variance of the normal distribution provided by the
            output of the encoder with shape (batch_size, self._latent_dim).
        size : int
            Size of the generated latent layer, used for generative model.
        Returns
        -------
        np.ndarray
            The latent sample generated with shape
            (batch_size, self._latent_dim).
        """
        if size:
            batch_size = size
            indices = np.random.choice(z_mu.shape[0],
                                       size=size, replace=False).tolist()
            z_mu = tf.gather(z_mu, indices)
            z_log_sigma_sq = tf.gather(z_log_sigma_sq, indices)
        else:
            batch_size = z_mu.shape[0]
        # generate samples from standard normal distribution
        # with the correct shape
        standard_normal_samples = tf.random.normal(
            shape=(batch_size, self._latent_dim))

        # transform standard normal samples to general normal samples with
        # mean z_mu and log of variance z_log_sigma_sq
        z = z_mu + tf.math.exp(0.5 * z_log_sigma_sq) * standard_normal_samples
        return z

    def call(self, x: np.ndarray):
        """Calls the model with input <x>.

        Parameters
        ----------
        x : np.ndarray
            Input image with shape (batch_size, height, width, 3).

        Returns
        -------
        tuple
            A tuple contains the logit of prediction result with shape
            (batch_size, 1), the mean parameter of the latent variable z in
            shape (batch_size, self._latent_dim), the log of the variance
            parameter of the latent variable z with shape
            (batch_size, self._latent_dim) and the reconstructed input <x> with
            shape (batch_size, height, width, 3).
        """
        # encode
        y_logit, z_mu, z_log_sigma_sq = self.encode(x)
        # generate latent layer
        z = self.sample_latent(z_mu, z_log_sigma_sq)
        # decode/reconstruct
        reconstruction = self.decode(z)
        return y_logit, z_mu, z_log_sigma_sq, reconstruction

    def predict(self, original_images: np.ndarray, predict_indices: list,
                unit_batch_size: int = 4096):
        """Predict/classify input <x> with only the VAE encoder.

        Parameters
        ----------
        original_images : np.ndarray
            Original input images (N, height, width, 3).
        unit_batch_size : int, optional
            Size of batch, for reducing the amount of computation
            in one single step, by default 4096.
        testing_indices : list
            A list of all testing indices.
        Returns
        -------
        np.ndarray
            The logit of prediction result with shape (batch_size, 1).
        """
        N = len(predict_indices)
        y_logit = np.array([], dtype=np.float32).reshape(0, 1)
        for start_ind in range(0, N, unit_batch_size):
            end_ind = min(start_ind+unit_batch_size, N)
            x_batch_indices = predict_indices[start_ind:end_ind]
            x_batch = preprocess_batch(x_batch_indices, original_images)[0]
            y_batch_logit = self.encode(x_batch)[0]
            y_logit = np.vstack((y_logit, y_batch_logit))
        return y_logit

    @tf.function
    def train_step(self, x: np.ndarray, y: np.ndarray,
                   optimizer: tf.keras.optimizers.Optimizer):
        """Training step of the VAE.

        Parameters
        ----------
        x : nd.array
            Input image with shape (batch_size, height, width, 3).
        y : nd.array
            Label of <x> with shape (batch_size, 1).
        optimizer : tf.keras.optimizers.Optimizer
            The training optimizer.
        Returns
        -------
        np.ndarray
            Total loss of the vae.
        """
        with tf.GradientTape() as tape:
            y_logit, z_mu, z_log_sigma_sq, x_reconstruction = self(x)
            total_loss = debiasing_loss_function(
                x, x_reconstruction, y, y_logit, z_mu, z_log_sigma_sq)[0]
        grads = tape.gradient(total_loss, self.trainable_variables)
        optimizer.apply_gradients(zip(grads, self.trainable_variables))
        return total_loss

    def _get_latent_mu(self, original_images: np.ndarray,
                       training_face_images_indices: list,
                       unit_batch_size: int = 4096):
        """Get all latent distribution mean parameter <z_mu> from the encoder
        output.

        Parameters
        ----------
        training_images : np.ndarray
            Training input images.
        training_face_images_indices : list
            A list of all indices of training faces.
        unit_batch_size: int, optional
            Size of batch, for reducing the amount of computation
            in one single step, by default 4096.

        Returns
        -------
        np.ndarray
            All latent distribution mean parameter <z_mu>.
        """
        N = len(training_face_images_indices)
        mu = np.zeros((N, self._latent_dim))
        for start_ind in range(0, N, unit_batch_size):
            end_ind = min(start_ind+unit_batch_size, N)
            img_indices = training_face_images_indices[start_ind:end_ind]
            batch = preprocess_batch(img_indices, original_images)[0]
            batch_mu = self.encode(batch)[1]
            mu[start_ind:end_ind] = batch_mu
        return mu

    def get_training_sample_probabilities(self, original_images: np.ndarray,
                                          training_face_images_indices: list,
                                          bins: int = 10,
                                          smoothing_factor: float = 0.001,
                                          unit_batch_size=4096):
        """Get the probablities of each instance in the training dataset to be
        chosen, debiasing the training inputs.

        Parameters
        ----------
        original_face_images : np.ndarray
            The original images.
        training_face_images_indices : list
            A list of all indices of training faces.
        bins : int, optional
            Bins of the histogram, by default 10.
        smoothing_factor : float, optional
            This parameter tunes the degree of debiasing: for smoothing_fac=0,
            the re-sampled training set will tend towards falling uniformly
            over the latent space, i.e., the most extreme debiasing,
            by default 0.001.
        unit_batch_size: int, optional
            Size of batch, for reducing the amount of computation
            in one single step, by default 4096.

        Returns
        -------
        np.ndarray
            The sampling probabilities for images within a batch
            based on how they distribute across the training data.
        """
        logger.info(
            f"Getting latent mu with unit batch size {unit_batch_size}")
        z_mu = self._get_latent_mu(original_images,
                                   training_face_images_indices,
                                   unit_batch_size)
        logger.info("Got latent mu.")
        training_sample_p = np.zeros(z_mu.shape[0])

        # Loop through each mean of the batch.
        for i in range(self._latent_dim):
            z_mu_distribution = z_mu[:, i]
            # Get the pdf of the distribution of z_mu.
            hist_density, bin_edges = np.histogram(
                z_mu_distribution, density=True, bins=bins)
            bin_edges[0] = -float('inf')
            bin_edges[-1] = float('inf')

            # number of mu's in each interval defined by bin_edges
            bin_idx = np.digitize(z_mu_distribution, bin_edges)

            # smoothing
            hist_smoothed_density = (
                hist_density + smoothing_factor) / np.sum(hist_density +
                                                          smoothing_factor)

            # invert the density function,
            # if the mu appears with low frequency in the batch,
            # we tend to sample more of the instance that produce this z_mu.
            p = 1.0/(hist_smoothed_density[bin_idx-1])

            # normalize to encode a probalility distribution
            p = p / np.sum(p)

            # take the maximum to see which
            training_sample_p = np.maximum(p, training_sample_p)
        # normalize to encode a probability distribution.
        training_sample_p /= np.sum(training_sample_p)
        return training_sample_p
