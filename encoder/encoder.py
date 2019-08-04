import tensorflow as tf
import numpy as np

from training.networks_stylegan import *
from training.loss import fp32
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Model
from dnnlib.tflib.autosummary import autosummary
from functools import partial


synthesis_kwargs = dict(output_transform=dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True), minibatch_size=8)

# ----------------------------------------------------------------------------
# Fully-connected encoder

def fc_encoder(
        image,  # Input image: [minibatch, channel, height, weight]
        labels,       #
        dlatent_size=512,  # Output shape
        mapping_layers=8,  # Number of mapping layers
        mapping_fmaps=50,  # Shape of intermediate latent features
        use_wscale=True,  # Enable equalized learning rate?
        mapping_lrmul=0.01,  # Learning rate multiplier for the mapping layers.
        mapping_nonlinearity='lrelu',  # Activation function: 'relu', 'lrelu'.
        label_size=0,             #
        **_kwargs):  # Ignore unrecognized keyword args.
    act, gain = {'relu': (tf.nn.relu, np.sqrt(2)), 'lrelu': (leaky_relu, np.sqrt(2))}[mapping_nonlinearity]
    image.set_shape([None, 3, 1024, 1024])
    image = tf.cast(image, 'float32')
    labels.set_shape([None, label_size])
    x = image
    lod_in = tf.cast(tf.get_variable('lod', initializer=np.float32(0.0), trainable=False), 'float32')
    for layer_idx in range(mapping_layers):
        with tf.variable_scope('Dense%d' % layer_idx):
            fmaps = dlatent_size if layer_idx == mapping_layers - 1 else mapping_fmaps
            x = dense(x, fmaps=fmaps, gain=gain, use_wscale=use_wscale, lrmul=mapping_lrmul)
            x = apply_bias(x, lrmul=mapping_lrmul)
            x = act(x)
    encoder_out = tf.identity(x, name='encoder_out')
    return encoder_out

# ----------------------------------------------------------------------------
# Inverse of style generator as encoder

def style_encoder(
        image,  # Input image: [minibatch, resolution, resolution, channel]
        dlatent_size=512,  # Disentangled latent (W) dimensionality.
        num_channels=3,  # Number of output color channels.
        resolution=1024,  # Output resolution.
        fmap_base=8192,  # Overall multiplier for the number of feature maps.
        fmap_decay=1.0,  # log2 feature map reduction when doubling the resolution.
        fmap_max=512,  # Maximum number of feature maps in any layer.
        use_styles=True,  # Enable style inputs?
        const_input_layer=True,  # First layer is a learned constant?
        use_noise=True,  # Enable noise inputs?
        randomize_noise=True,
        # True = randomize noise inputs every time (non-deterministic), False = read noise inputs from variables.
        nonlinearity='lrelu',  # Activation function: 'relu', 'lrelu'
        use_wscale=True,  # Enable equalized learning rate?
        use_pixel_norm=False,  # Enable pixelwise feature vector normalization?
        use_instance_norm=True,  # Enable instance normalization?
        dtype='float32',  # Data type to use for activations and outputs.
        fused_scale='auto',
        # True = fused convolution + scaling, False = separate ops, 'auto' = decide automatically.
        blur_filter=[1, 2, 1],  # Low-pass filter to apply when resampling activations. None = no filtering.
        structure='auto',
        # 'fixed' = no progressive growing, 'linear' = human-readable, 'recursive' = efficient, 'auto' = select automatically.
        is_template_graph=False,
        # True = template graph constructed by the Network class, False = actual evaluation.
        force_clean_graph=False,
        # True = construct a clean graph that looks nice in TensorBoard, False = default behavior.
        **_kwargs):  # Ignore unrecognized keyword args.

    pass

# ----------------------------------------------------------------------------
# Modelling encoder as a latent distribution conditional on images


def vae_encoder():
    pass

# ----------------------------------------------------------------------------
# Loss function for encoder.


def encoder_loss(G, E, D, E_opt, training_set, minibatch_size, reals, beta, labels=None, latent_broadcast=18):
    latents = E.get_output_for(reals, labels, is_training=True)
    # fakes = G.components.synthesis.run(tf.tile(latents[:, np.newaxis], [1, latent_broadcast, 1]),
    #                                    minibatch_size=minibatch_size)
    fakes = G.components.synthesis.get_output_for(tf.tile(latents[:, np.newaxis], [1, latent_broadcast, 1]), is_training=True)
    v_loss = vgg_loss(training_set, reals, fakes)
    w_loss = wp_loss(D, reals, fakes, labels)
    loss = v_loss + beta * w_loss
    return loss


def vgg_loss(training_set, reals, fakes, vgg_depth=9, mapping_fmaps=512):
    vgg16 = VGG16(include_top=False, input_shape=(training_set.shape[1], training_set.shape[2], training_set.shape[0]))
    vgg_model = Model(vgg16.input, vgg16.layers[vgg_depth].output)
    fake_img_features = vgg_model(fakes)
    real_img_features = vgg_model(reals)
    with tf.name_scope('VggLoss'):
        logits = dense(fake_img_features - real_img_features, fmaps=mapping_fmaps)
        loss = tf.losses.mean_squared_error(logits, tf.zeros(logits.shape))
    return loss


def wp_loss(D, reals, fakes, labels,
    wgan_lambda=10.0):  # Weight for the gradient penalty term.
    real_scores_out = fp32(D.get_output_for(reals, labels, is_training=True))
    fake_scores_out = fp32(D.get_output_for(fakes, labels, is_training=True))
    loss = fake_scores_out - real_scores_out
    grads = fp32(tf.gradients(fake_scores_out, [fakes])[0])
    grads_norms = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3]))
    loss += wgan_lambda * grads_norms
    return loss








