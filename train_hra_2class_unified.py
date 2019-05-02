# -*- coding: utf-8 -*-
""" Top-level (abstract) script for training (fine-tuning) various CNNs on the HRA dataset with 2 classes.

    Example
    --------
    >>> python train_hra_2class_unified.py --violation_class cl --pre_trained_model vgg16 --nb_of_conv_layers_to_fine_tune 1 --nb_of_epochs 50

"""

from __future__ import print_function
import argparse
import os

from applications.hra_utils import _obtain_weights_CSVLogger_filenames,_obtain_train_mode, _obtain_first_phase_trained_weights
from wrappers.hra_transfer_cnn_manager import HRA_Transfer_CNN_Manager

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--violation_class", type = str, help = " One of `cl` ([i]'child_labour' & [ii]'no violation') "
                         " or `dp` ([i]'displaced_populations' & [ii]'no violation')")
    parser.add_argument("--pre_trained_model", type = str,help = 'One of `vgg16`, `vgg19`, `resnet50` or `vgg16_places365`')
    parser.add_argument("--nb_of_conv_layers_to_fine_tune", type = int, default=None, help = "Number of conv. layers to fine-tune")
    parser.add_argument("--nb_of_epochs", type = int, help = "Total number of iterations on the data")

    args = parser.parse_args()
    return args



# --------- Configure and pass a tensorflow session to Keras to restrict GPU memory fraction --------- #
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.50
set_session(tf.Session(config=config))


args = get_args()

# feature extraction case
if args.nb_of_conv_layers_to_fine_tune is None:
    first_phase_trained_weights = None

# fine-tune case
elif args.nb_of_conv_layers_to_fine_tune in {1, 2, 3}:
    first_phase_trained_weights = _obtain_first_phase_trained_weights(violation_class = args.violation_class, model_name= args.pre_trained_model)
    # check if the first_phase_trained_weights does exist
    if os.path.isfile(first_phase_trained_weights) is False:
        raise IOError("No such weights file: `" + first_phase_trained_weights + "`. ")

train_mode = _obtain_train_mode(nb_of_conv_layers_to_fine_tune=args.nb_of_conv_layers_to_fine_tune)

weights_filename, CSVLogger_filename = _obtain_weights_CSVLogger_filenames(violation_class=args.violation_class,
                                                                           train_mode=train_mode,
                                                                           model_name=args.pre_trained_model,
                                                                           nb_of_conv_layers_to_fine_tune=args.nb_of_conv_layers_to_fine_tune
                                                                           )

modelCheckpoint_quantity = 'val_loss'
earlyStopping_quantity = 'val_loss'



transfer_cnn_manager = HRA_Transfer_CNN_Manager(violation_class = args.violation_class,
                                                train_mode=train_mode,
                                                pre_trained_model = args.pre_trained_model,
                                                nb_of_conv_layers_to_fine_tune = args.nb_of_conv_layers_to_fine_tune,
                                                weights_to_file = weights_filename,
                                                first_phase_trained_weights = first_phase_trained_weights,
                                                nb_of_epochs = args.nb_of_epochs,
                                                modelCheckpoint_quantity = modelCheckpoint_quantity,
                                                earlyStopping_quantity = earlyStopping_quantity,
                                                CSVLogger_filename = CSVLogger_filename,
                                                )


transfer_cnn_manager.train()
