# -*- coding: utf-8 -*-
"""2 clss Human Rights Archive (HRA) VGG16 model for Keras

"""

from __future__ import division, print_function
import os

import warnings
import numpy as np

from keras import backend as K
from keras.utils.data_utils import get_file
from keras.layers import Input
from keras.layers.core import Dense
from keras.models import Model
from keras.layers.core import Dropout
from keras.layers import GlobalAveragePooling2D
from keras.applications.vgg16 import VGG16
from keras.applications.imagenet_utils import _obtain_input_shape
from keras.engine.topology import get_source_inputs
from keras.optimizers import SGD
from applications.hra_utils import _obtain_weights_path as owp

from applications.hra_utils import _obtain_train_mode

pre_trained_model = 'vgg16'

# ======================  ------------------------  ======================  ------------------------  ======================  ------------------------  ======================  ------------------------  ======================  ------------------------
CL_WEIGHTS_FEATURE_EXTRACTION_PATH, CL_FEATURE_EXTRACTION_FNAME = owp('cl', pre_trained_model, None, True)
DP_WEIGHTS_FEATURE_EXTRACTION_PATH, DP_FEATURE_EXTRACTION_FNAME = owp('cl', pre_trained_model, None, True)

# ======================  ------------------------  ======================  ------------------------  ======================  ------------------------  ======================  ------------------------  ======================  ------------------------
CL_WEIGHTS_PATH_ONE_CONV_LAYER, CL_PATH_ONE_CONV_LAYER_FNAME = owp('cl', pre_trained_model, 1, True)
CL_WEIGHTS_PATH_ONE_CONV_LAYER_NO_TOP, CL_PATH_ONE_CONV_LAYER_NO_TOP_FNAME = owp('cl', pre_trained_model, 1, False)

CL_WEIGHTS_PATH_TWO_CONV_LAYERS, CL_PATH_TWO_CONV_LAYERS_FNAME = owp('cl', pre_trained_model, 2, True)
CL_WEIGHTS_PATH_TWO_CONV_LAYERS_NO_TOP, CL_PATH_TWO_CONV_LAYERS_NO_TOP_FNAME = owp('cl', pre_trained_model, 2, False)

CL_WEIGHTS_PATH_THREE_CONV_LAYERS, CL_PATH_THREE_CONV_LAYERS_FNAME = owp('cl', pre_trained_model, 3, True)
CL_WEIGHTS_PATH_THREE_CONV_LAYERS_NO_TOP, CL_PATH_THREE_CONV_LAYERS_NO_TOP_FNAME = owp('cl', pre_trained_model, 3, False)

# ======================  ------------------------  ======================  ------------------------  ======================  ------------------------  ======================  ------------------------  ======================  ------------------------
DP_WEIGHTS_PATH_ONE_CONV_LAYER, DP_PATH_ONE_CONV_LAYER_FNAME = owp('dp', pre_trained_model, 1, True)
DP_WEIGHTS_PATH_ONE_CONV_LAYER_NO_TOP, DP_PATH_ONE_CONV_LAYER_NO_TOP_FNAME = owp('dp', pre_trained_model, 1, False)

DP_WEIGHTS_PATH_TWO_CONV_LAYERS, DP_PATH_TWO_CONV_LAYERS_FNAME = owp('dp', pre_trained_model, 2, True)
DP_WEIGHTS_PATH_TWO_CONV_LAYERS_NO_TOP, DP_PATH_TWO_CONV_LAYERS_NO_TOP_FNAME = owp('dp', pre_trained_model, 2, False)

DP_WEIGHTS_PATH_THREE_CONV_LAYERS, DP_PATH_THREE_CONV_LAYERS_FNAME = owp('dp', pre_trained_model, 3, True)
DP_WEIGHTS_PATH_THREE_CONV_LAYERS_NO_TOP, DP_PATH_THREE_CONV_LAYERS_NO_TOP_FNAME = owp('dp', pre_trained_model, 3, False)

# ======================  ------------------------  ======================  ------------------------  ======================  ------------------------  ======================  ------------------------  ======================  ------------------------

def HRA_VGG16(include_top=True, weights='HRA',
              input_tensor=None, input_shape=None,
              nb_of_conv_layers_to_fine_tune=None,
              first_phase_trained_weights = None,
              violation_class = 'cl',
              verbose=0):
    """Instantiates the VGG16 architecture fine-tuned (2 steps) on Human Rights Archive dataset.

    Optionally loads weights pre-trained on the 2 class version of Human Rights Archive Database.

    # Arguments
        include_top: whether to include the 3 fully-connected
            layers at the top of the network.
        weights: one of `None` (random initialization),
            'HRA' (pre-training on Human Rights Archive),
            or the path to the weights file to be loaded.
        input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
            to use as image input for the model.
        input_shape: optional shape tuple, only to be specified
            if `include_top` is False (otherwise the input shape
            has to be `(224, 224, 3)` (with `channels_last` data format)
            or `(3, 224, 224)` (with `channels_first` data format).
            It should have exactly 3 input channels,
            and width and height should be no smaller than 48.
            E.g. `(200, 200, 3)` would be one valid value.
        nb_of_conv_layers_to_fine_tune: integer to indicate the number of convolutional
            layers to fine-tune. One of `1` (2,499,360 trainable params), `2` (4,859,168 trainable params) or `3` (7,218,976 trainable params).
        first_phase_trained_weights: Weights of an already trained Keras model instance.
            Only relevant when using `fine_tuning` as train_mode after `feature_extraction` weights have been saved.
        violation_class: one of `cl` (HRA dataset with 2 classes - [i]'child_labour' and [ii]'no violation')
            or `dp` (HRA dataset with 2 classes - [i]'displaced_populations' and [ii]'no violation')
        verbose: Integer. 0, or 1. Verbosity mode. 0 = silent, 1 = model summary and weights info.

    # Returns
        A Keras model instance.

    # Raises
        ValueError: in case of invalid argument for `weights`, `violation_class`, `nb_of_conv_layers_to_fine_tune` or invalid input shape
        """
    if not (weights in {'HRA', None} or os.path.exists(weights)):
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization), `HRA` '
                         '(pre-training on Human Rights Archive two-class), '
                         'or the path to the weights file to be loaded.')


    if not (violation_class in {'cl', 'dp'}):
        raise ValueError("The `violation_class` argument should be either "
                         "`cl` (HRA dataset with 2 classes - [i]'child_labour' and [ii]'no violation') "
                         "'or `dp` (HRA dataset with 2 classes - [i]'displaced_populations' and [ii]'no violation')")

    if nb_of_conv_layers_to_fine_tune is None and include_top is False:
        raise ValueError('Setting the `include_top` argument as false '
                         'is only relevant when the `nb_of_conv_layers_to_fine_tune` argument is not None (feature extraction), '
                         'otherwise the returned model would be exactly the default '
                         'keras-applications model.')

    if weights == 'HRA' and first_phase_trained_weights is not None:
        raise ValueError('Setting the `first_phase_trained_weights` argument as the path to the weights file '
                         'obtained from utilising feature_extraction '
                         'is only relevant when the `weights` argument is `None`. '
                         'If the `weights` argument is `HRA`, it means the model has already been trained on HRA dataset '
                         'and there is no need to provide a path to the weights file (saved from feature_extraction) to be loaded.')

    if not (nb_of_conv_layers_to_fine_tune in {1, 2, 3, None}):
        raise ValueError('The `nb_of_conv_layers_to_fine_tune` argument should be either '
                         '`None` (indicates feature extraction mode), '
                         '`1`, `2` or `3`. '
                         'More than 3 conv. layers are not supported because the more parameters we are training , '
                         'the more we are at risk of overfitting.')

    cache_subdir = 'AbuseNet'

    mode = _obtain_train_mode(nb_of_conv_layers_to_fine_tune=nb_of_conv_layers_to_fine_tune)

    # Determine proper input shape
    input_shape = _obtain_input_shape(input_shape,
                                      default_size=224,
                                      min_size=48,
                                      data_format=K.image_data_format(),
                                      require_flatten=include_top,
                                      weights=weights)

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    # Ensure that the model takes into account any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input


    # create the base pre-trained model
    base_model = VGG16(weights='imagenet', include_top=False, input_tensor=img_input)
    x = base_model.output

    # Classification block - build a classifier model to put on top of the convolutional model
    if include_top:

        # add a global spatial pooling layer (which seems to have the best performance)
        x = GlobalAveragePooling2D(name='GAP')(x)

        # add a fully-connected layer
        x = Dense(256, activation='relu', name='FC1')(x)

        # When random init is enabled, we want to include Dropout,
        # otherwise when loading a pre-trained HRA model we want to omit that layer,
        # so the visualisations are done properly (there is an issue if it is included)
        if weights is None:
            x = Dropout(0.5,name='DROPOUT')(x)
        # and a logistic layer with the number of classes defined by the `classes` argument
        x = Dense(2, activation='softmax', name='PREDICTIONS')(x)

        model = Model(inputs=inputs, outputs=x, name='HRA-2CLASS-VGG16')

    else:
        model = Model(inputs=inputs, outputs=x, name='HRA-2CLASS-VGG16-NO-TOP')
        model.compile(optimizer=SGD(lr=0.0001, momentum=0.9),
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

        return model




    if mode == 'feature_extraction':

        print('[INFO] Feature extraction mode. \n')

        if verbose == 1:
            print(
                '[INFO] Number of trainable weights before freezing the conv. base of the original pre-trained convnet: '
                '' + str(len(model.trainable_weights)))

        for layer in base_model.layers:
            layer.trainable = False

        if verbose == 1:
            print(
                '[INFO] Number of trainable weights after freezing the conv. base of the original pre-trained convnet: '
                '' + str(len(model.trainable_weights)))

        model.compile(optimizer=SGD(lr=0.0001, momentum=0.9),
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])


    elif mode == 'fine_tuning':


            if nb_of_conv_layers_to_fine_tune == 1:
                # Uncomment for extra verbosity
                # print('[INFO] Fine-tuning of the last one (1) conv. layer. \n')

                if verbose == 1:
                    print(
                        '[INFO] Number of trainable weights before unfreezing the last conv. layer of the model with the retrained classifier: '
                        '' + str(len(model.trainable_weights)))

                for layer in model.layers[:17]:
                    layer.trainable = False
                for layer in model.layers[17:]:
                    layer.trainable = True

                if verbose == 1:
                    print(
                        '[INFO] Number of trainable weights after unfreezing the last conv. layer of the model with the retrained classifier: '
                        '' + str(len(model.trainable_weights)))

            elif nb_of_conv_layers_to_fine_tune == 2:
                # Uncomment for extra verbosity
                # print('[INFO] Fine-tuning of the last two (2) conv. layers. \n')
                if verbose == 1:
                    print(
                        '[INFO] Number of trainable weights before unfreezing the last two (2) conv. layers of the model with the retrained classifier: '
                        '' + str(len(model.trainable_weights)))

                for layer in model.layers[:16]:
                    layer.trainable = False
                for layer in model.layers[16:]:
                    layer.trainable = True

                if verbose == 1:
                    print(
                        '[INFO] Number of trainable weights after unfreezing the last two (2) conv. layers of the model with the retrained classifier: '
                        '' + str(len(model.trainable_weights)))

            elif nb_of_conv_layers_to_fine_tune == 3:
                # Uncomment for extra verbosity
                # print('[INFO] Fine-tuning of the last three (3) conv. layers. \n')
                if verbose == 1:
                    print(
                        '[INFO] Number of trainable weights before unfreezing the last three (3) conv. layers of the model with the retrained classifier: '
                        '' + str(len(model.trainable_weights)))

                for layer in model.layers[:15]:
                    layer.trainable = False
                for layer in model.layers[15:]:
                    layer.trainable = True

                if verbose == 1:
                    print(
                        '[INFO] Number of trainable weights after unfreezing the last three (3) conv. layers of the model with the retrained classifier: '
                        '' + str(len(model.trainable_weights)))

            model.compile(optimizer=SGD(lr=0.0001, momentum=0.9),
                          loss='categorical_crossentropy',
                          metrics=['accuracy'])

    if verbose == 1:
        model.summary()

    # load weights
    if weights == 'HRA':

        # Child labour
        if violation_class =='cl':
            if include_top:
                if mode == 'feature_extraction':
                    weights_path = get_file(CL_FEATURE_EXTRACTION_FNAME,
                                            CL_WEIGHTS_FEATURE_EXTRACTION_PATH,
                                            cache_subdir=cache_subdir)

                elif mode == 'fine_tuning':

                    if nb_of_conv_layers_to_fine_tune == 1:
                        weights_path = get_file(CL_PATH_ONE_CONV_LAYER_FNAME,
                                                CL_WEIGHTS_PATH_ONE_CONV_LAYER,
                                                cache_subdir=cache_subdir)
                    elif nb_of_conv_layers_to_fine_tune == 2:
                        weights_path = get_file(CL_PATH_TWO_CONV_LAYERS_FNAME,
                                                CL_WEIGHTS_PATH_TWO_CONV_LAYERS,
                                                cache_subdir=cache_subdir)
                    elif nb_of_conv_layers_to_fine_tune == 3:
                        weights_path = get_file(CL_PATH_THREE_CONV_LAYERS_FNAME,
                                                CL_WEIGHTS_PATH_THREE_CONV_LAYERS,
                                                cache_subdir=cache_subdir)

            # no top
            else:
                if nb_of_conv_layers_to_fine_tune == 1:
                    weights_path = get_file(CL_PATH_ONE_CONV_LAYER_NO_TOP_FNAME,
                                            CL_WEIGHTS_PATH_ONE_CONV_LAYER_NO_TOP,
                                            cache_subdir=cache_subdir)
                elif nb_of_conv_layers_to_fine_tune == 2:
                    weights_path = get_file(CL_PATH_TWO_CONV_LAYERS_NO_TOP_FNAME,
                                            CL_WEIGHTS_PATH_TWO_CONV_LAYERS_NO_TOP,
                                            cache_subdir=cache_subdir)
                elif nb_of_conv_layers_to_fine_tune == 3:
                    weights_path = get_file(CL_PATH_THREE_CONV_LAYERS_NO_TOP_FNAME,
                                            CL_WEIGHTS_PATH_THREE_CONV_LAYERS_NO_TOP,
                                            cache_subdir=cache_subdir)
        # Displaced populations
        elif violation_class == 'dp':
            if include_top:
                if mode == 'feature_extraction':
                    weights_path = get_file(DP_FEATURE_EXTRACTION_FNAME,
                                            DP_WEIGHTS_FEATURE_EXTRACTION_PATH,
                                            cache_subdir=cache_subdir)

                elif mode == 'fine_tuning':

                    if nb_of_conv_layers_to_fine_tune == 1:
                        weights_path = get_file(DP_PATH_ONE_CONV_LAYER_FNAME,
                                                DP_WEIGHTS_PATH_ONE_CONV_LAYER,
                                                cache_subdir=cache_subdir)
                    elif nb_of_conv_layers_to_fine_tune == 2:
                        weights_path = get_file(DP_PATH_TWO_CONV_LAYERS_FNAME,
                                                DP_WEIGHTS_PATH_TWO_CONV_LAYERS,
                                                cache_subdir=cache_subdir)
                    elif nb_of_conv_layers_to_fine_tune == 3:
                        weights_path = get_file(DP_PATH_THREE_CONV_LAYERS_FNAME,
                                                DP_WEIGHTS_PATH_THREE_CONV_LAYERS,
                                                cache_subdir=cache_subdir)

            # no top
            else:
                if nb_of_conv_layers_to_fine_tune == 1:
                    weights_path = get_file(DP_PATH_ONE_CONV_LAYER_NO_TOP_FNAME,
                                            DP_WEIGHTS_PATH_ONE_CONV_LAYER_NO_TOP,
                                            cache_subdir=cache_subdir)
                elif nb_of_conv_layers_to_fine_tune == 2:
                    weights_path = get_file(DP_PATH_TWO_CONV_LAYERS_NO_TOP_FNAME,
                                            DP_WEIGHTS_PATH_TWO_CONV_LAYERS_NO_TOP,
                                            cache_subdir=cache_subdir)
                elif nb_of_conv_layers_to_fine_tune == 3:
                    weights_path = get_file(DP_PATH_THREE_CONV_LAYERS_NO_TOP_FNAME,
                                            DP_WEIGHTS_PATH_THREE_CONV_LAYERS_NO_TOP,
                                            cache_subdir=cache_subdir)

        model.load_weights(weights_path)

    elif weights is not None:
        model.load_weights(weights)

    return model
