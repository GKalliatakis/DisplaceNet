# -*- coding: utf-8 -*-
'''Emotion Recognition in Context model for Keras

# Reference:
- [Emotion Recognition in Context](http://sunai.uoc.edu/emotic/pdf/EMOTIC_cvpr2017.pdf)
'''

from __future__ import division, print_function
import os

from keras.layers import Input
from keras.layers.core import Dense
from keras.models import Model
from keras.layers.core import Dropout
from keras.utils.data_utils import get_file
from keras import regularizers

from keras.applications.vgg16 import VGG16
from keras.layers.merge import concatenate
from applications.vgg16_places_365 import VGG16_Places365
from keras.optimizers import SGD

from utils.generic_utils import euclidean_distance_loss, rmse


WEIGHTS_PATH = 'https://github.com/GKalliatakis/ubiquitous-assets/releases/download/v0.7.0/emotic_vad_VGG16_weights_tf_dim_ordering_tf_kernels.h5'
WEIGHTS_PATH_NO_TOP = ''


def EMOTIC_VAD_VGG16(include_top=True,
                     weights='emotic'):
    """Instantiates the EMOTIC_VAD_VGG16 architecture.

    Optionally loads weights pre-trained
    on EMOTIC. Note that when using TensorFlow,
    for best performance you should set
    `image_data_format="channels_last"` in your Keras config
    at ~/.keras/keras.json.

    The model and the weights are compatible with both
    TensorFlow and Theano. The data format
    convention used by the model is the one
    specified in your Keras config file.

    # Arguments
        include_top: whether to include the 3 fully-connected
            layers at the top of the network.
        weights: one of `None` (random initialization),
                 'emotic' (pre-training on EMOTIC),
                 or the path to the weights file to be loaded.
        classes: optional number of discrete emotion classes to classify images into.
    # Returns
        A Keras model instance.
    # Raises
        ValueError: in case of invalid argument for `weights`
    """

    if not (weights in {'emotic', None} or os.path.exists(weights)):
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization), `emotic` '
                         '(pre-training on EMOTIC dataset), '
                         'or the path to the weights file to be loaded.')

    body_inputs = Input(shape=(224, 224, 3), name='INPUT')
    image_inputs = Input(shape=(224, 224, 3), name='INPUT')


    body_truncated_model = VGG16(include_top=False, weights='imagenet', input_tensor=body_inputs, pooling='avg')
    for layer in body_truncated_model.layers:
        layer.name = str("body-") + layer.name

    image_truncated_model = VGG16_Places365(include_top=False, weights='places', input_tensor=image_inputs, pooling='avg')
    for layer in image_truncated_model.layers:
        layer.name = str("image-") + layer.name

    # retrieve the ouputs
    body_plain_model_output = body_truncated_model.output
    image_plain_model_output = image_truncated_model.output

    merged = concatenate([body_plain_model_output, image_plain_model_output])

    x = Dense(256, activation='relu', name='FC1', kernel_regularizer=regularizers.l2(0.01), kernel_initializer='random_normal')(merged)

    x = Dropout(0.5, name='DROPOUT')(x)

    vad_cont_prediction = Dense(units=3, kernel_initializer='random_normal', name='VAD')(x)

    # At model instantiation, you specify the two inputs and the output.
    model = Model(inputs=[body_inputs, image_inputs], outputs=vad_cont_prediction, name='EMOTIC-VAD-regression-ResNet50')

    for layer in body_truncated_model.layers:
        layer.trainable = False

    for layer in image_truncated_model.layers:
        layer.trainable = False

    model.compile(optimizer=SGD(lr=1e-5, momentum=0.9),
                  loss=euclidean_distance_loss,
                  metrics=['mae', 'mse', rmse])

    # load weights
    if weights == 'emotic':
        if include_top:
            weights_path = get_file('emotic_vad_VGG16_weights_tf_dim_ordering_tf_kernels.h5',
                                    WEIGHTS_PATH,
                                    cache_subdir='AbuseNet')
        else:
            weights_path = get_file('emotic_vad_VGG16_weights_tf_dim_ordering_tf_kernels_notop.h5',
                                    WEIGHTS_PATH_NO_TOP,
                                    cache_subdir='AbuseNet')

        model.load_weights(weights_path)


    elif weights is not None:
        model.load_weights(weights)

    return model

