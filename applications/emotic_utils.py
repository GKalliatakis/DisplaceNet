"""Generic utilities continuous-emotion-recognition-in-VAD-space-based models.
"""

from __future__ import print_function
import numpy as np

import keras.backend as K
import itertools
from itertools import product
import tensorflow as tf
import keras.backend.tensorflow_backend as tfb
from keras.preprocessing import image
from applications.emotic_vgg16 import EMOTIC_VAD_VGG16
from applications.emotic_vgg19 import EMOTIC_VAD_VGG19
from applications.emotic_resnet50 import EMOTIC_VAD_ResNet50
from utils.generic_utils import imagenet_preprocess_input, places_preprocess_input



target_size = (224, 224)

CLASS_INDEX = None
CLASS_INDEX_PATH = 'https://github.com/GKalliatakis/Keras-EMOTIC-resources/releases/download/v1.0/emotic_class_index.json'


def _obtain_weights_CSVLogger_filenames(body_backbone_CNN, image_backbone_CNN):
    """Obtains the polished filenames for the weights and the CSVLogger of the model.

    # Arguments
        model_name: String to declare the name of the model

    # Returns
        Two strings that will serve as the filenames for the weights and the CSVLogger respectively.
    """

    prefix= 'trained_models/emotic_vad_'
    suffix= '_weights_tf_dim_ordering_tf_kernels.h5'
    weights_filename = prefix + body_backbone_CNN + suffix


    CSVLogger_filename = 'emotic_vad_'+body_backbone_CNN+'_training.csv'

    return weights_filename, CSVLogger_filename



# ----------------------------------------------------------------------------------------------------- #
#                                    Obtain number of classifiers
# ----------------------------------------------------------------------------------------------------- #

def _obtain_nb_classifiers(model_a_name = None, model_b_name = None, model_c_name = None):
    """Obtains the number of different classifiers based on given model names.
        Note that EMOTIC model has already combined body backbone CNN features (which in this case are the  `model_b_name` or `model_c_name`
        features, with `VGG16_Places365` features at training stage, but for simplicity reasons only the body backbone CNN name is adjustable.

    # Arguments
        model_b_name: One of `VGG16`, `VGG19`, `ResNet50` or `None`.
        model_c_name: One of `VGG16`, `VGG19`, `ResNet50` or `None`.

    # Returns
        The number of different classifiers alongside a polished file_name
    """

    if (model_b_name) is None and (model_c_name is not None):
        raise ValueError('The models names must be set in the correct order starting from model_a --> model_b --> model_c. ')


    if (model_b_name in {None}) and (model_c_name in {None}):
        nb_classifiers = 1
        file_name = model_a_name

        return nb_classifiers, file_name

    if (model_b_name in {'VGG16', 'VGG19', 'ResNet50'}) and (model_c_name in {None}):
        nb_classifiers = 2
        file_name = model_a_name + '_' + model_b_name
        return nb_classifiers, file_name

    if (model_b_name in {'VGG16', 'VGG19', 'ResNet50'})  and (model_c_name in {'VGG16', 'VGG19', 'ResNet50'}) :
        nb_classifiers = 3
        file_name = model_a_name + '_' + model_b_name + '_' + model_c_name
        return nb_classifiers, file_name


# ----------------------------------------------------------------------------------------------------- #
#                                    Prepare input images
# ----------------------------------------------------------------------------------------------------- #

def prepare_input_data(body_path,
                       image_path):
    """Prepares the raw images for the EMOTIC model.

    # Arguments
        body_path: Path to body only image file.
        image_path: Path to entire image file.

    # Returns
        The two processed images
    """

    body_img = image.load_img(body_path, target_size=(224, 224))
    x1 = image.img_to_array(body_img)
    x1 = np.expand_dims(x1, axis=0)
    x1 = imagenet_preprocess_input(x1)

    entire_img = image.load_img(image_path, target_size=(224, 224))
    x2 = image.img_to_array(entire_img)
    x2 = np.expand_dims(x2, axis=0)
    x2 = places_preprocess_input(x2)


    return x1, x2


# ----------------------------------------------------------------------------------------------------- #
#                            Obtain ensembling weights for different classifiers
# ----------------------------------------------------------------------------------------------------- #

def _obtain_ensembling_weights(nb_classifiers,
                               model_a_name,
                               model_b_name,
                               model_c_name):
    """Obtains the set of ensembling weights that will be used for conducting weighted average.

    # Arguments
        nb_classifiers: Integer, number of different classifiers that will be used for ensembling weights.
        model_a_name: One of `VGG16`, `VGG19`, `ResNet50` or `None`.
        model_b_name: One of `VGG16`, `VGG19`, `ResNet50` or `None`.
        model_c_name: One of `VGG16`, `VGG19`, `ResNet50` or `None`.

    # Returns
        The weights (float) for every model and every dimension.
    """

    if nb_classifiers == 2:

        if model_a_name == 'VGG16' and model_b_name == 'ResNet50':
            w_model_a = 0.55
            w_model_b = 0.45

            return w_model_a, w_model_b

        # ensure that giving models with different order will not effect the weights
        elif model_a_name == 'ResNet50' and model_b_name == 'VGG16':
            w_model_a = 0.45
            w_model_b = 0.55

            return w_model_a, w_model_b


        elif model_a_name == 'VGG16' and model_b_name == 'VGG19':
            w_model_a = 0.48
            w_model_b = 0.52

            return w_model_a, w_model_b

        # ensure that giving models with different order will not effect the weights
        elif model_a_name == 'VGG19' and model_b_name == 'VGG16':
            w_model_a = 0.52
            w_model_b = 0.48

            return w_model_a, w_model_b


        elif model_a_name == 'ResNet50' and model_b_name == 'VGG19':
            w_model_a = 0.40
            w_model_b = 0.60

            return w_model_a, w_model_b


        # ensure that giving models with different order will not effect the weights
        elif model_a_name == 'VGG19' and model_b_name == 'ResNet50':

            w_model_a = 0.60
            w_model_b = 0.40

            return w_model_a, w_model_b


    elif nb_classifiers == 3:

        if model_a_name == 'VGG16' and model_b_name == 'ResNet50' and model_c_name == 'VGG19':
            w_model_a = 0.35
            w_model_b = 0.28
            w_model_c = 0.37

            return w_model_a, w_model_b, w_model_c

        elif model_a_name == 'VGG16' and model_b_name == 'VGG19' and model_c_name == 'ResNet50':
            w_model_a = 0.35
            w_model_b = 0.37
            w_model_c = 0.28

            return w_model_a, w_model_b, w_model_c


        elif model_a_name == 'ResNet50' and model_b_name == 'VGG16' and model_c_name == 'VGG19':
            w_model_a = 0.28
            w_model_b = 0.35
            w_model_c = 0.37

            return w_model_a, w_model_b, w_model_c


        elif model_a_name == 'ResNet50' and model_b_name == 'VGG19' and model_c_name == 'VGG16':
            w_model_a = 0.28
            w_model_b = 0.37
            w_model_c = 0.35

            return w_model_a, w_model_b, w_model_c

        elif model_a_name == 'VGG19' and model_b_name == 'ResNet50' and model_c_name == 'VGG16':
            w_model_a = 0.37
            w_model_b = 0.28
            w_model_c = 0.35

            return w_model_a, w_model_b, w_model_c


        elif model_a_name == 'VGG19' and model_b_name == 'VGG16' and model_c_name == 'ResNet50':
            w_model_a = 0.37
            w_model_b = 0.35
            w_model_c = 0.28

            return w_model_a, w_model_b, w_model_c



# ----------------------------------------------------------------------------------------------------- #
#                      Obtain Keras model instances based on given model names
# ----------------------------------------------------------------------------------------------------- #

def _obtain_single_model_VAD(model_a_name):
    """Instantiates and returns 1 Keras model instance based on the given model name.

    # Arguments
        model_a_name: String to declare the name of the 1st model

    # Returns
        Single Keras model instance.
    """

    if model_a_name == 'VGG16':
        model_a = EMOTIC_VAD_VGG16(include_top=True, weights='emotic')

    elif model_a_name == 'VGG19':
        model_a = EMOTIC_VAD_VGG19(include_top=True, weights='emotic')

    elif model_a_name == 'ResNet50':
        model_a = EMOTIC_VAD_ResNet50(include_top=True, weights='emotic')

    return model_a


def _obtain_two_models_ensembling_VAD(model_a_name, model_b_name):
    """Instantiates and returns 2 Keras model instances based on the given model names.

    # Arguments
        model_a_name: String to declare the name of the 1st model
        model_b_name: String to declare the name of the 2nd model

    # Returns
        Two Keras model instances.
    """

    if model_a_name == 'VGG16':
        model_a = EMOTIC_VAD_VGG16(include_top=True, weights='emotic')

    elif model_a_name == 'VGG19':
        model_a = EMOTIC_VAD_VGG19(include_top=True, weights='emotic')

    elif model_a_name == 'ResNet50':
        model_a = EMOTIC_VAD_ResNet50(include_top=True, weights='emotic')


    if model_b_name == 'VGG16':
        model_b = EMOTIC_VAD_VGG16(include_top=True, weights='emotic')

    elif model_b_name == 'VGG19':
        model_b = EMOTIC_VAD_VGG19(include_top=True, weights='emotic')

    elif model_b_name == 'ResNet50':
        model_b = EMOTIC_VAD_ResNet50(include_top=True, weights='emotic')

    return model_a, model_b


def _obtain_three_models_ensembling_VAD(model_a_name, model_b_name, model_c_name):
    """Instantiates and returns 3 Keras model instances based on the given model names.

    # Arguments
        model_a_name: String to declare the name of the 1st model
        model_b_name: String to declare the name of the 2nd model
        model_c_name: String to declare the name of the 3rd model

    # Returns
        Three Keras model instances.
    """

    if model_a_name == 'VGG16':
        model_a = EMOTIC_VAD_VGG16(include_top=True, weights='emotic')

    elif model_a_name == 'VGG19':
        model_a = EMOTIC_VAD_VGG19(include_top=True, weights='emotic')

    elif model_a_name == 'ResNet50':
        model_a = EMOTIC_VAD_ResNet50(include_top=True, weights='emotic')


    if model_b_name == 'VGG16':
        model_b = EMOTIC_VAD_VGG16(include_top=True, weights='emotic')

    elif model_b_name == 'VGG19':
        model_b = EMOTIC_VAD_VGG19(include_top=True, weights='emotic')

    elif model_b_name == 'ResNet50':
        model_b = EMOTIC_VAD_ResNet50(include_top=True, weights='emotic')


    if model_c_name == 'VGG16':
        model_c = EMOTIC_VAD_VGG16(include_top=True, weights='emotic')

    elif model_c_name == 'VGG19':
        model_c = EMOTIC_VAD_VGG19(include_top=True, weights='emotic')

    elif model_c_name == 'ResNet50':
        model_c = EMOTIC_VAD_ResNet50(include_top=True, weights='emotic')

    return model_a, model_b, model_c



