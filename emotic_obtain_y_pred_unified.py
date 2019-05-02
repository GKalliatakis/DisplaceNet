# -*- coding: utf-8 -*-
""" This is meant to run on the server in order to record the estimated target values of either a single classifier or ensemble of classifiers
    for continuous emotion recognition in VAD space.
"""

from __future__ import print_function
import argparse
import pandas
import numpy as np
from utils.generic_utils import print_progress
from applications.emotic_utils import prepare_input_data, _obtain_nb_classifiers,_obtain_ensembling_weights, \
    _obtain_single_model_VAD, _obtain_two_models_ensembling_VAD, _obtain_three_models_ensembling_VAD


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_a", type = str,help = 'One of `VGG16`, `VGG19` or `ResNet50`')
    parser.add_argument("--model_b", type=str, default= None, help='One of `VGG16`, `VGG19`, `ResNet50` or `None`')
    parser.add_argument("--model_c", type=str, default= None, help='One of `VGG16`, `VGG19`, `ResNet50` or `None`')

    args = parser.parse_args()
    return args

args = get_args()

model_a_name = args.model_a
model_b_name = args.model_b
model_c_name = args.model_c

nb_classifiers, numpy_name = _obtain_nb_classifiers(model_a_name=model_a_name,
                                                    model_b_name=model_b_name,
                                                    model_c_name=model_c_name)

if nb_classifiers == 1:
    model_a = _obtain_single_model_VAD(model_a_name)

elif nb_classifiers == 2:
    w_model_a, w_model_b = _obtain_ensembling_weights(nb_classifiers=nb_classifiers,
                                                      model_a_name=model_a_name,
                                                      model_b_name=model_b_name,
                                                      model_c_name=model_c_name)

    model_a, model_b = _obtain_two_models_ensembling_VAD(model_a_name=model_a_name, model_b_name=model_b_name)

elif nb_classifiers == 3:
    w_model_a, w_model_b, w_model_c = _obtain_ensembling_weights(nb_classifiers=nb_classifiers,
                                                                 model_a_name=model_a_name,
                                                                 model_b_name=model_b_name,
                                                                 model_c_name=model_c_name)

    model_a, model_b, model_c = _obtain_three_models_ensembling_VAD(model_a_name=model_a_name,
                                                                    model_b_name=model_b_name,
                                                                    model_c_name=model_c_name)

# counter to iterate through all csv entries
field_number = 0

final_list = []

# server
csv_file = pandas.read_csv('/home/gkallia/git/emotic-VAD-classification/dataset/test.csv')
base_dir_of_cropped_imgs = '/home/gkallia/git/emotic-VAD-classification/dataset/raw_images/cropped_imgs/'
base_dir_of_entire_imgs = '/home/gkallia/git/emotic-VAD-classification/dataset/raw_images/entire_imgs/'

for entry in csv_file.filename:

    print_progress(iteration=field_number, total=7280, prefix='Progress:', suffix='Complete')

    person_img_path = base_dir_of_cropped_imgs + entry
    entire_img_path = base_dir_of_entire_imgs + entry

    x1, x2 = prepare_input_data(body_path = person_img_path,
                                image_path = entire_img_path)

    if nb_classifiers == 1:

        preds = model_a.predict([x1, x2])


    elif nb_classifiers == 2:
        # obtain predictions
        preds_model_a = model_a.predict([x1, x2])
        preds_model_b = model_b.predict([x1, x2])

        if w_model_a is None and w_model_b is None:
            # This new prediction array should be more accurate than any of the initial ones
            preds = 0.50 * (preds_model_a + preds_model_b)

        else:
            preds = w_model_a * preds_model_a + w_model_b * preds_model_b

    elif nb_classifiers == 3:
        # obtain predictions
        preds_model_a = model_a.predict([x1, x2])
        preds_model_b = model_b.predict([x1, x2])
        preds_model_c = model_c.predict([x1, x2])

        if w_model_a is None and w_model_b is None and w_model_c is None:
            # This new prediction array should be more accurate than any of the initial ones
            preds = 0.33 * (preds_model_a + preds_model_b +  preds_model_c)

        else:
            preds = w_model_a * preds_model_a + w_model_b * preds_model_b + w_model_c * preds_model_c

    final_list.append(preds[0])


    field_number += 1

final_numpy_name = 'y_predicted/' + numpy_name + '_y_predicted.npy'


np.save(final_numpy_name, final_list)


print('\n')
print('[INFO] NumPy array for estimated target values has been saved as `%s`' %final_numpy_name)



