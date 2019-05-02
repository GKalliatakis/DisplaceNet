# -*- coding: utf-8 -*-
'''

'''
from __future__ import print_function
import os

import numpy as np

from keras.preprocessing import image
from applications.hra_resnet50 import HRA_ResNet50
from applications.hra_vgg16 import HRA_VGG16
from applications.hra_vgg19 import HRA_VGG19
from applications.hra_vgg16_places365 import HRA_VGG16_Places365
from applications.hra_utils import plot_preds
from applications.hra_utils import prepare_input_data


def single_img_HRA_inference(img_path,
                             violation_class,
                             model_backend_name,
                             nb_of_conv_layers_to_fine_tune):

    """Performs single image inference.

    # Arguments
        img_path: Path to image file
        violation_class: one of `cl` (HRA dataset with 2 classes - [i]'child_labour' and [ii]'no violation')
            or `dp` (HRA dataset with 2 classes - [i]'displaced_populations' and [ii]'no violation')
        model_backend_name: One of `VGG16`, `VGG19`, `ResNet50` or `VGG16_Places365`.
        nb_of_conv_layers_to_fine_tune: integer to indicate the number of convolutional layers to fine-tune.
    # Returns
        Three integer values corresponding to `valence`, `arousal` and `dominance`.

    """
    (head, tail) = os.path.split(img_path)
    filename_only = os.path.splitext(tail)[0]

    if model_backend_name == 'VGG16':
        model = HRA_VGG16(weights='HRA',
                          violation_class=violation_class,
                          nb_of_conv_layers_to_fine_tune=nb_of_conv_layers_to_fine_tune)

        print("[INFO] Loading and preprocessing image...")

        x = prepare_input_data(img_path = img_path, objects_or_places_flag = 'objects')

    elif model_backend_name == 'VGG19':
        model = HRA_VGG19(weights='HRA',
                          violation_class=violation_class,
                          nb_of_conv_layers_to_fine_tune=nb_of_conv_layers_to_fine_tune)

        print("[INFO] Loading and preprocessing image...")

        x = prepare_input_data(img_path=img_path, objects_or_places_flag='objects')

    elif model_backend_name == 'ResNet50':
        model = HRA_ResNet50(weights='HRA',
                             violation_class=violation_class,
                             nb_of_conv_layers_to_fine_tune=nb_of_conv_layers_to_fine_tune)

        print("[INFO] Loading and preprocessing image...")

        x = prepare_input_data(img_path=img_path, objects_or_places_flag='objects')

    elif model_backend_name == 'VGG16_Places365':
        model = HRA_VGG16_Places365(weights='HRA',
                                    violation_class=violation_class,
                                    nb_of_conv_layers_to_fine_tune=nb_of_conv_layers_to_fine_tune)

        print("[INFO] Loading and preprocessing image...")

        x = prepare_input_data(img_path=img_path, objects_or_places_flag='places')



    # There seems to be relatively high confidence scores with the following method
    # preds = model.predict(x)[0]
    #
    # print ('Raw predictions: ', preds)
    #
    # top_preds = np.argsort(preds)[::-1][0:2]
    #
    # print('Sorted predictions: ', top_preds)
    #
    # if violation_class == 'cl':
    #     file_name = 'categories_HRA_2classCL.txt'
    # elif violation_class == 'dp':
    #     file_name = 'categories_HRA_2classDP.txt'
    #
    # classes = list()
    # with open(file_name) as class_file:
    #     for line in class_file:
    #         classes.append(line.strip().split(' ')[0][3:])
    # classes = tuple(classes)
    #
    #
    # print ('\n')
    # print('--PREDICTED HRA 2 CLASSES:')
    # # output the prediction
    # for i in range(0, 2):
    #     print(classes[top_preds[i]], '->', preds[top_preds[i]])





    img = image.load_img(img_path, target_size=(224, 224))

    from applications.hra_utils import predict as pd
    raw_preds, decoded_preds = pd(violation_class=violation_class,
                                  model=model,
                                  img=img,
                                  target_size=(224, 224))

    # print('Raw preds: ', raw_preds)
    # print ('Decoded preds: ', decoded_preds)

    # print (type(raw_preds))
    # print('Raw preds: ', raw_preds[0])
    # print(type(raw_preds[0]))

    top_1_predicted_probability = decoded_preds[0][2]

    # top_1_predicted = np.argmax(preds)
    top_1_predicted_label = decoded_preds[0][1]
    # print(top_1_predicted_label, '->' , top_1_predicted_probability)

    overlayed_text = str(top_1_predicted_label)+ ' (' + str(round(top_1_predicted_probability, 2)) + ')'

    return raw_preds, overlayed_text, top_1_predicted_label



def single_img_HRA_inference_return_only(img_path,
                                         violation_class,
                                         model_backend_name,
                                         nb_of_conv_layers_to_fine_tune):

    """Performs single image inference.

    # Arguments
        img_path: Path to image file
        violation_class: one of `cl` (HRA dataset with 2 classes - [i]'child_labour' and [ii]'no violation')
            or `dp` (HRA dataset with 2 classes - [i]'displaced_populations' and [ii]'no violation')
        model_backend_name: One of `VGG16`, `VGG19`, `ResNet50` or `VGG16_Places365`.
        nb_of_conv_layers_to_fine_tune: integer to indicate the number of convolutional layers to fine-tune.
    # Returns
        Three integer values corresponding to `valence`, `arousal` and `dominance`.

    """

    if model_backend_name == 'VGG16':
        model = HRA_VGG16(weights='HRA',
                          violation_class=violation_class,
                          nb_of_conv_layers_to_fine_tune=nb_of_conv_layers_to_fine_tune)

    elif model_backend_name == 'VGG19':
        model = HRA_VGG19(weights='HRA',
                          violation_class=violation_class,
                          nb_of_conv_layers_to_fine_tune=nb_of_conv_layers_to_fine_tune)

    elif model_backend_name == 'ResNet50':
        model = HRA_ResNet50(weights='HRA',
                             violation_class=violation_class,
                             nb_of_conv_layers_to_fine_tune=nb_of_conv_layers_to_fine_tune)

    elif model_backend_name == 'VGG16_Places365':
        model = HRA_VGG16_Places365(weights='HRA',
                                    violation_class=violation_class,
                                    nb_of_conv_layers_to_fine_tune=nb_of_conv_layers_to_fine_tune)

    ## Uncomment for extra verbosity
    # print('[INFO] HRA-2class model has been loaded')

    img = image.load_img(img_path, target_size=(224, 224))

    from applications.hra_utils import predict as pd
    raw_preds, decoded_preds = pd(violation_class=violation_class,
                                  model=model,
                                  img=img,
                                  target_size=(224, 224))

    return raw_preds

