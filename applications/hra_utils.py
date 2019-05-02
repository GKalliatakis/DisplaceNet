"""Utilities for HRA data preprocessing, prediction decoding and plotting.
"""

from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt

from keras.preprocessing import image
from keras.applications.inception_v3 import preprocess_input

from keras.utils import get_file
import json
from utils.generic_utils import imagenet_preprocess_input, places_preprocess_input


target_size = (224, 224)

CLASS_INDEX = None
CL_CLASS_INDEX_PATH = 'https://github.com/GKalliatakis/ubiquitous-assets/releases/download/v0.1.3/HRA_2classCL_index.json'
DP_CLASS_INDEX_PATH = 'https://github.com/GKalliatakis/ubiquitous-assets/releases/download/v0.1.3/HRA_2classDP_index.json'


def _obtain_train_mode(nb_of_conv_layers_to_fine_tune):
    """Obtains the train mode string based on the provided number of conv. layers that will be fine-tuned.

    # Arguments
        nb_of_conv_layers_to_fine_tune: integer to indicate the number of convolutional layersto fine-tune.
            One of `1`, `2` or `3`.
    # Returns
        A string that will serve as the train mode of the model.
    """

    if nb_of_conv_layers_to_fine_tune == None:
        return 'feature_extraction'
    elif nb_of_conv_layers_to_fine_tune in {1, 2, 3}:
        return 'fine_tuning'
    else:
        raise ValueError('The `nb_of_conv_layers_to_fine_tune` argument should be either '
                         '`None` (indicates feature extraction mode), '
                         '`1`, `2` or `3`. More than 3 conv. blocks are not included '
                         'because the more parameters we are training (unfreezing), the more we are at risk of overfitting.')



def _obtain_first_phase_trained_weights (violation_class,
                                         model_name):
    """Retrieves the weights of an already trained feature extraction model.
        Only relevant when using `fine_tuning` as train_mode after `feature_extraction` weights have been saved.

    # Arguments
        violation_class: violation_class: one of `cl` (HRA dataset with 2 classes - [i]'child_labour' and [ii]'no violation')
            or `dp` (HRA dataset with 2 classes - [i]'displaced_populations' and [ii]'no violation').
        model_name: String to declare the name of the model
    # Returns
        A string with the weights path.
    """

    if violation_class == 'cl':
        first_phase_trained_weights_filename = 'trained_models/' + 'cl_' + model_name + '_weights_feature_extraction_tf_dim_ordering_tf_kernels.h5'
    elif violation_class == 'dp':
        first_phase_trained_weights_filename = 'trained_models/' + 'dp_' + model_name + '_weights_feature_extraction_tf_dim_ordering_tf_kernels.h5'


    return first_phase_trained_weights_filename





def _obtain_weights_CSVLogger_filenames(violation_class,
                                        train_mode,
                                        model_name,
                                        nb_of_conv_layers_to_fine_tune):
    """Obtains the polished filenames for the weights and the CSVLogger of the model.

    # Arguments
        violation_class: violation_class: one of `cl` (HRA dataset with 2 classes - [i]'child_labour' and [ii]'no violation')
            or `dp` (HRA dataset with 2 classes - [i]'displaced_populations' and [ii]'no violation').
        train_mode: String to declare the train mode of the model (how many layers will be frozen during training).
            - `feature_extraction` taking the convolutional base of a previously-trained network,
                running the new data through it, and training a new classifier on top of the output.
            - `fine_tuning` unfreezing a few of the top layers of a frozen conv. base used for feature extraction,
                and jointly training both the newly added part of the model and these top layers.
        model_name: String to declare the name of the model
        nb_of_conv_layers_to_fine_tune: integer to indicate the number of convolutional
            layers to fine-tune. One of `1`, `2` or `3`.

    # Returns
        Two strings that will serve as the filenames for the weights and the CSVLogger respectively.
    """

    if violation_class == 'cl':
        if train_mode == 'feature_extraction':

            weights_filename = 'trained_models/' + 'cl_' + model_name + '_weights_feature_extraction_tf_dim_ordering_tf_kernels.h5'
            CSVLogger_filename = model_name + '_cl_feature_extraction.csv'

        else:

            if nb_of_conv_layers_to_fine_tune == 1:
                weights_filename = 'trained_models/' + 'cl_' + model_name + '_weights_one_layer_tf_dim_ordering_tf_kernels.h5'
                CSVLogger_filename = model_name + '_cl_fine_tuning_one_layer.csv'

            elif nb_of_conv_layers_to_fine_tune == 2:
                weights_filename = 'trained_models/' + 'cl_' + model_name + '_weights_two_layers_tf_dim_ordering_tf_kernels.h5'
                CSVLogger_filename = model_name + '_cl_fine_tuning_two_layers.csv'

            elif nb_of_conv_layers_to_fine_tune == 3:
                weights_filename = 'trained_models/' + 'cl_' + model_name + '_weights_three_layers_tf_dim_ordering_tf_kernels.h5'
                CSVLogger_filename = model_name + '_cl_fine_tuning_three_layers.csv'

            else:
                raise NotImplementedError(
                    'The `nb_of_conv_layers_to_fine_tune` argument should be either `1`, `2` or `3`. '
                    'More than 3 conv. blocks are not supported because the more parameters we are training, '
                    'the more we are at risk of overfitting.')

    elif violation_class == 'dp':
        if train_mode == 'feature_extraction':

            weights_filename = 'trained_models/' + 'dp_' + model_name + '_weights_feature_extraction_tf_dim_ordering_tf_kernels.h5'
            CSVLogger_filename = model_name + '_dp_feature_extraction.csv'

        else:

            if nb_of_conv_layers_to_fine_tune == 1:
                weights_filename = 'trained_models/' + 'dp_' + model_name + '_weights_one_layer_tf_dim_ordering_tf_kernels.h5'
                CSVLogger_filename = model_name + '_dp_fine_tuning_one_layer.csv'

            elif nb_of_conv_layers_to_fine_tune == 2:
                weights_filename = 'trained_models/' + 'dp_' + model_name + '_weights_two_layers_tf_dim_ordering_tf_kernels.h5'
                CSVLogger_filename = model_name + '_dp_fine_tuning_two_layers.csv'

            elif nb_of_conv_layers_to_fine_tune == 3:
                weights_filename = 'trained_models/' + 'dp_' + model_name + '_weights_three_layers_tf_dim_ordering_tf_kernels.h5'
                CSVLogger_filename = model_name + '_dp_cl_fine_tuning_three_layers.csv'

            else:
                raise NotImplementedError(
                    'The `nb_of_conv_layers_to_fine_tune` argument should be either `1`, `2` or `3`. '
                    'More than 3 conv. blocks are not supported because the more parameters we are training, '
                    'the more we are at risk of overfitting.')


    return weights_filename, CSVLogger_filename




def _obtain_weights_path(violation_class,
                         pre_trained_model,
                         nb_of_conv_layers_to_fine_tune,
                         include_top):
    """Obtains the polished filenames for the weights of a trained Keras model.

        # Arguments
            violation_class: violation_class: one of `cl` (HRA dataset with 2 classes - [i]'child_labour' and [ii]'no violation')
                or `dp` (HRA dataset with 2 classes - [i]'displaced_populations' and [ii]'no violation').
            pre_trained_model: 'One of `vgg16`, `vgg19`, `resnet50` or `vgg16_places365`.
            nb_of_conv_layers_to_fine_tune: integer to indicate the number of convolutional
                layers to fine-tune. One of `1`, `2` or `3`.
            include_top: whether to include the 3 fully-connected layers at the top of the network.

        # Returns
            Two strings that will serve as the original URL of the file (origin) and name of the file (fname) for loading the weights.
        """

    # This is the only URL that must be altered in case of changing the repo where weights files are stored.
    # The rest will be automatically inferred from the github_repo URL.
    # Note that the structure of the releases must comply with the following structure:
    # main_release_dir/
    #               v0.1.1(weights for feature extraction mode)/
    #               v0.1.2(weights for fine-tuning mode)/


    github_repo = 'https://github.com/GKalliatakis/ubiquitous-assets/releases'

    if nb_of_conv_layers_to_fine_tune == None:
        fname = violation_class + '_' + pre_trained_model + '_weights_feature_extraction_tf_dim_ordering_tf_kernels.h5'

    elif nb_of_conv_layers_to_fine_tune == 1:
        if include_top:
            fname = violation_class + '_' + pre_trained_model + '_weights_one_layer_tf_dim_ordering_tf_kernels.h5'
        else:
            fname = violation_class + '_' + pre_trained_model + '_weights_one_layer_tf_dim_ordering_tf_kernels_notop.h5'

    elif nb_of_conv_layers_to_fine_tune == 2:
        if include_top:
            fname = violation_class + '_' + pre_trained_model + '_weights_two_layers_tf_dim_ordering_tf_kernels.h5'
        else:
            fname = violation_class + '_' + pre_trained_model + '_weights_two_layers_tf_dim_ordering_tf_kernels_notop.h5'

    elif nb_of_conv_layers_to_fine_tune == 3:
        if include_top:
            fname = violation_class + '_' + pre_trained_model + '_weights_three_layers_tf_dim_ordering_tf_kernels.h5'
        else:
            fname = violation_class + '_' + pre_trained_model + '_weights_three_layers_tf_dim_ordering_tf_kernels_notop.h5'



    if nb_of_conv_layers_to_fine_tune == None:
        origin = github_repo + '/download/v0.1.1/' + fname
    else:
        origin = github_repo + '/download/v0.1.2/' + fname


    return origin, fname



def decode_predictions(violation_class,preds, top=2):
    """Decodes the prediction of a HRA-2CLASS model.

        # Arguments
            violation_class: violation_class: one of `cl` (HRA dataset with 2 classes - [i]'child_labour' and [ii]'no violation')
                or `dp` (HRA dataset with 2 classes - [i]'displaced_populations' and [ii]'no violation')
            preds: Numpy tensor encoding a batch of predictions.
            top: integer, how many top-guesses to return.

        # Returns
            A list of lists of top class prediction tuples `(class_name, class_description, score)`.
            One list of tuples per sample in batch input.

        # Raises
            ValueError: in case of invalid shape of the `pred` array must be 2D).
    """
    global CLASS_INDEX
    if len(preds.shape) != 2 or preds.shape[1] != 2:
        raise ValueError('`decode_predictions` expects '
                         'a batch of predictions '
                         '(i.e. a 2D array of shape (samples, 2)). '
                         'Found array with shape: ' + str(preds.shape))
    if CLASS_INDEX is None:
        if violation_class =='cl':
            fpath = get_file('HRA_2classCL_index.json',
                             CL_CLASS_INDEX_PATH,
                             cache_subdir='AbuseNet')
            CLASS_INDEX = json.load(open(fpath))
        elif violation_class =='dp':
            fpath = get_file('HRA_2classDP_index.json',
                             DP_CLASS_INDEX_PATH,
                             cache_subdir='AbuseNet')
            CLASS_INDEX = json.load(open(fpath))

    results = []
    for pred in preds:
        top_indices = pred.argsort()[-top:][::-1]
        result = [tuple(CLASS_INDEX[str(i)]) + (pred[i],) for i in top_indices]
        result.sort(key=lambda x: x[2], reverse=True)
        results.append(result)

    return results


def predict(violation_class, model, img, target_size):
  """Generates output predictions for a single PIL image.

    # Arguments
        violation_class: violation_class: one of `cl` (HRA dataset with 2 classes - [i]'child_labour' and [ii]'no violation')
            or `dp` (HRA dataset with 2 classes - [i]'displaced_populations' and [ii]'no violation')
        model: keras model
        img: PIL format image
        target_size: (w,h) tuple

    # Returns
        list of predicted labels and their probabilities
  """
  if img.size != target_size:
    img = img.resize(target_size)

  x = image.img_to_array(img)
  x = np.expand_dims(x, axis=0)
  x = preprocess_input(x)
  preds = model.predict(x)

  # print ('Raw preds: ',preds )

  return preds, decode_predictions(violation_class = violation_class, preds = preds, top=2)[0]


def predict_v2(violation_class, model, img, target_size):
  """Generates output predictions for a single PIL image.
    # Arguments
        violation_class: violation_class: one of `cl` (HRA dataset with 2 classes - [i]'child_labour' and [ii]'no violation')
            or `dp` (HRA dataset with 2 classes - [i]'displaced_populations' and [ii]'no violation')
        model: keras model
        img: PIL format image
        target_size: (w,h) tuple
    # Returns
        list of predicted labels and their probabilities
  """
  if img.size != target_size:
    img = img.resize(target_size)

  x = image.img_to_array(img)
  x = np.expand_dims(x, axis=0)
  x = preprocess_input(x)
  preds = model.predict(x)

  return decode_predictions(violation_class = violation_class, preds = preds, top=2)[0]


def duo_ensemble_predict(violation_class,
                         model_a, model_b,
                         img,
                         target_size
                         ):
  """Generates output predictions for a single PIL image for 2 different models,
    and then puts together those predictions by averaging them at inference time.

    # Arguments
        violation_class: violation_class: one of `cl` (HRA dataset with 2 classes - [i]'child_labour' and [ii]'no violation')
            or `dp` (HRA dataset with 2 classes - [i]'displaced_populations' and [ii]'no violation')
        model_a: 1st model
        model_b: 2nd model
        img: PIL format image
        target_size: (w,h) tuple

    # Returns
        list of predicted labels (which have been pooled accordingly) and their probabilities
  """
  if img.size != target_size:
    img = img.resize(target_size)

  x = image.img_to_array(img)
  x = np.expand_dims(x, axis=0)
  x = preprocess_input(x)

  preds_a = model_a.predict(x)
  preds_b = model_b.predict(x)
  final_preds = 0.50 * (preds_a + preds_b)

  return decode_predictions(violation_class = violation_class, preds = final_preds, top=2)[0]



def plot_preds(violation_class, image, preds):
  """Displays image and the top-n predicted probabilities in a bar graph.

  # Arguments
    violation_class: violation_class: one of `cl` (HRA dataset with 2 classes - [i]'child_labour' and [ii]'no violation')
        or `dp` (HRA dataset with 2 classes - [i]'displaced_populations' and [ii]'no violation')
    image: PIL image
    preds: list of predicted labels and their probabilities
  """

  if violation_class == 'cl':
      labels = ("Child Labour", "NO Child Labour")

  elif violation_class == 'dp':
      labels = ("Displaced Populations", "NO Displaced Populations")



  order = list(reversed(range(len(preds))))
  plt.imshow(image)
  plt.axis('off')

  # fig = plt.figure(figsize=(2, 2))
  #
  # fig.add_subplot(1, 1, 1)
  # plt.imshow(image)
  #
  # fig.add_subplot(2, 2, 2)
  # plt.barh(order, preds, alpha=0.55)


  plt.figure()
  plt.barh(order, preds, alpha=0.55)
  plt.yticks(order, labels)
  plt.xlabel('Probability')
  plt.xlim(0,1.01)
  plt.tight_layout()
  plt.show()


def prepare_input_data(img_path,
                       objects_or_places_flag):
    """Prepares the raw images for the EMOTIC model.

    # Arguments
        body_path: Path to body only image file.
        image_path: Path to entire image file.

    # Returns
        The two processed images
    """

    body_img = image.load_img(img_path, target_size=(224, 224))
    x1 = image.img_to_array(body_img)
    x1 = np.expand_dims(x1, axis=0)

    if objects_or_places_flag == 'objects':
        x1 = imagenet_preprocess_input(x1)

    elif objects_or_places_flag == 'places':
        x1 = places_preprocess_input(x1)


    return x1