# -*- coding: utf-8 -*-
""" HRA_Transfer_CNN_Manager is a wrapper class which 'encapsulates' the functionalities needed for preparing (class instantiation)
    and training different CNNs (`train_model`) on the HRA dataset with 2 classes.
"""

from __future__ import print_function
import os
import sys
import math
import numpy as np
import os.path
import time
import datetime
import h5py
from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger
from keras.preprocessing.image import ImageDataGenerator

from applications.hra_vgg16 import HRA_VGG16
from applications.hra_vgg19 import HRA_VGG19
from applications.hra_resnet50 import HRA_ResNet50
from applications.hra_vgg16_places365 import HRA_VGG16_Places365

from utils.generic_utils import hms_string


class HRA_Transfer_CNN_Manager():
    """Loads the parameters needed for the training process on class instantiation
        & starts the training process.

        # Arguments
            violation_class: violation_class: one of `cl` (HRA dataset with 2 classes - [i]'child_labour' and [ii]'no violation')
                or `dp` (HRA dataset with 2 classes - [i]'displaced_populations' and [ii]'no violation')
            train_mode: String to declare the train mode of the model (how many layers will be frozen during training).
                - `feature_extraction` taking the convolutional base of a previously-trained network,
                    running the new data through it, and training a new classifier on top of the output.
                - `fine_tuning` unfreezing a few of the top layers of a frozen conv. base used for feature extraction,
                    and jointly training both the newly added part of the model and these top layers.
            pre_trained_model: One of `vgg16`, `vgg19`, `resnet50` or `vgg16_places365`.
            nb_of_conv_layers_to_fine_tune: integer to indicate the number of convolutional layers to fine-tune.
                One of `None` (indicates feature extraction mode), `1`, `2` or `3`.
            weights_to_file: File name or full path for saving the weights of the current training process.
            first_phase_trained_weights: Weights of an already trained feature extraction model.
                Only relevant when using `fine_tuning` as train_mode after `feature_extraction` weights have been saved.
            nb_of_epochs: Integer, total number of iterations on the data.
            modelCheckpoint_quantity: Quantity to monitor when saving the model after every epoch is enabled.
            earlyStopping_quantity: Quantity to monitor when stopping training when a monitored quantity has stopped improving is enabled.
            CSVLogger_filename: filename of the csv file, where the CSVLogger callback will stream epoch results to.

        # Raises
            ValueError: in case of invalid argument for `nb_of_conv_layers_to_fine_tune`
                or invalid argument for `first_phase_trained_weights`.
    """

    def __init__(self,
                 violation_class,
                 train_mode,
                 pre_trained_model,
                 nb_of_conv_layers_to_fine_tune,
                 weights_to_file,
                 first_phase_trained_weights,
                 nb_of_epochs,
                 modelCheckpoint_quantity,
                 earlyStopping_quantity,
                 CSVLogger_filename,
                 ):


        # extra check for the case when fine-tuning is selected without providing the correct first_phase_trained_weights.
        if  nb_of_conv_layers_to_fine_tune in {1, 2, 3} and first_phase_trained_weights is None:
            raise ValueError('The `first_phase_trained_weights` argument can be set to None only when '
                             '`nb_of_conv_layers_to_fine_tune` is None (feature extraction).'
                             'When `nb_of_conv_layers_to_fine_tune` is either 1 or 2, '
                             'the weights of an already trained feature extraction model must be saved prior to fine-tuning the model.')

        # Base directory for saving the trained models
        self.trained_models_dir = '/home/gkallia/git/Human-Rights-Violations-Conceptron/trained_models'
        self.feature_extraction_dir = os.path.join(self.trained_models_dir, 'feature_extraction/')
        self.fine_tuning_dir = os.path.join(self.trained_models_dir, 'fine_tuning/')
        self.logs_dir = os.path.join(self.trained_models_dir, 'logs/')

        if violation_class == 'cl':
            self.train_dir = os.path.join('/home/gkallia/git/Human-Rights-Violations-Conceptron/datasets/Two-class-HRV/ChildLabour', 'train')
            self.val_dir = os.path.join('/home/gkallia/git/Human-Rights-Violations-Conceptron/datasets/Two-class-HRV/ChildLabour', 'val')

        elif violation_class == 'dp':
            self.train_dir = os.path.join('/home/gkallia/git/Human-Rights-Violations-Conceptron/datasets/Two-class-HRV/DisplacedPopulations', 'train')
            self.val_dir = os.path.join('/home/gkallia/git/Human-Rights-Violations-Conceptron/datasets/Two-class-HRV/DisplacedPopulations', 'val')



        # Augmentation configuration with only rescaling.
        # Rescale is a value by which we will multiply the data before any other processing.
        # Our original images consist in RGB coefficients in the 0-255, but such values would
        # be too high for our models to process (given a typical learning rate),
        # so we target values between 0 and 1 instead by scaling with a 1/255. factor.
        datagen = ImageDataGenerator(rescale=1. / 255)

        img_width, img_height = 224, 224

        self.train_batch_size = 21
        self.val_batch_size = 10


        print('[INFO] Setting up image data generators...')

        self.train_generator = datagen.flow_from_directory(self.train_dir, target_size=(img_width, img_height),
                                                           class_mode='categorical',
                                                           shuffle=False,
                                                           batch_size=self.train_batch_size)

        self.val_generator = datagen.flow_from_directory(self.val_dir, target_size=(img_width, img_height),
                                                         class_mode='categorical',
                                                         shuffle=False,
                                                         batch_size=self.val_batch_size)


        num_classes = len(self.train_generator.class_indices)

        print('[INFO] Number of classes: ', num_classes)

        self.nb_train_samples = len(self.train_generator.filenames)
        # train_labels = self.train_generator.classes
        # self.train_labels = to_categorical(train_labels, num_classes=num_classes)
        # self.predict_size_train = int(math.ceil(self.nb_train_samples / self.train_batch_size))

        print ('[INFO] Number of train samples: ', self.nb_train_samples)

        # print('[INFO] Predict size train: ', self.predict_size_train)

        # save the class indices to use use later in predictions
        # np.save('class_indices.npy', self.train_generator.class_indices)


        self.nb_val_samples = len(self.val_generator.filenames)
        # val_labels = self.val_generator.classes
        # self.val_labels = to_categorical(val_labels, num_classes=num_classes)
        # self.predict_size_test = int(math.ceil(self.nb_val_samples / self.val_batch_size))

        print ('[INFO] Number of test samples: ', self.nb_val_samples)
        # print('[INFO] Predict size test: ', self.predict_size_test)

        self.steps_per_epoch = self.nb_train_samples // self.train_batch_size
        self.val_steps = self.nb_val_samples // self.val_batch_size



        # -------------------------------------------------------------------------------- #
        #                                Usage of callbacks
        # -------------------------------------------------------------------------------- #

        self.train_mode = train_mode
        self.pre_trained_model = pre_trained_model
        self.nb_of_conv_layers_to_fine_tune = nb_of_conv_layers_to_fine_tune
        self.weights_to_file = weights_to_file
        self.first_phase_trained_weights = first_phase_trained_weights
        self.nb_of_epochs = nb_of_epochs
        # self.modelCheckpoint_quantity = modelCheckpoint_quantity
        # self.earlyStopping_quantity = earlyStopping_quantity
        # self.CSVLogger_filename = CSVLogger_filename

        # self.steps_per_epoch = self.nb_train_samples // self.train_batch_size
        #
        #
        # self.val_steps = self.nb_val_samples // self.val_batch_size



        # CSVLogger
        model_log = 'trained_models/logs/' + CSVLogger_filename
        csv_logger = CSVLogger(model_log, append=True, separator=',')


        # ModelCheckpoint
        checkpointer = ModelCheckpoint(filepath=weights_to_file,
                                       monitor=modelCheckpoint_quantity,
                                       verbose=1,
                                       save_best_only=True,
                                       mode='auto',
                                       period=1,
                                       save_weights_only=True)

        early_stop = EarlyStopping(monitor=earlyStopping_quantity, patience=5, mode='auto')

        self.callbacks_list = [checkpointer, early_stop, csv_logger]


    def train(self):
        """Loads the selected model & starts the training process.
        """

        if self.pre_trained_model == 'vgg16':

            print('[INFO] Instantiating HRA-2CLASS-VGG16...')

            if self.train_mode == 'feature_extraction':

                model = HRA_VGG16(include_top=True, weights=None,
                                  input_tensor=None, input_shape=None,
                                  nb_of_conv_layers_to_fine_tune=self.nb_of_conv_layers_to_fine_tune,
                                  first_phase_trained_weights=None,
                                  verbose=1)
            else:

                if os.path.isfile(self.first_phase_trained_weights) is False:
                    raise IOError("No such weights file: `" + self.first_phase_trained_weights + "`. ")

                model = HRA_VGG16(include_top=True, weights=None,
                                  input_tensor=None, input_shape=None,
                                  nb_of_conv_layers_to_fine_tune=self.nb_of_conv_layers_to_fine_tune,
                                  first_phase_trained_weights=self.first_phase_trained_weights,
                                  verbose=1)

            print('[INFO] HRA-2CLASS-VGG16 model loaded')



        elif self.pre_trained_model == 'vgg19':

            print('[INFO] Instantiating HRA-2CLASS-VGG19...')

            if self.train_mode == 'feature_extraction':

                model = HRA_VGG19(include_top=True, weights=None,
                                  input_tensor=None, input_shape=None,
                                  nb_of_conv_layers_to_fine_tune=self.nb_of_conv_layers_to_fine_tune,
                                  first_phase_trained_weights=None,
                                  verbose=1)
            else:

                if os.path.isfile(self.first_phase_trained_weights) is False:
                    raise IOError("No such weights file: `" + self.first_phase_trained_weights + "`. ")

                model = HRA_VGG19(include_top=True, weights=None,
                                  input_tensor=None, input_shape=None,
                                  nb_of_conv_layers_to_fine_tune=self.nb_of_conv_layers_to_fine_tune,
                                  first_phase_trained_weights=self.first_phase_trained_weights,
                                  verbose=1)

            print('[INFO] HRA-2CLASS-VGG19 model loaded')




        elif self.pre_trained_model == 'resnet50':

            print('[INFO] Instantiating HRA-2CLASS-ResNet50...')

            if self.train_mode == 'feature_extraction':

                model = HRA_ResNet50(include_top=True, weights=None,
                                     input_tensor=None, input_shape=None,
                                     nb_of_conv_layers_to_fine_tune=self.nb_of_conv_layers_to_fine_tune,
                                     first_phase_trained_weights=None,
                                     verbose=1)
            else:

                if os.path.isfile(self.first_phase_trained_weights) is False:
                    raise IOError("No such weights file: `" + self.first_phase_trained_weights + "`. ")

                model = HRA_ResNet50(include_top=True, weights=None,
                                     input_tensor=None, input_shape=None,
                                     nb_of_conv_layers_to_fine_tune=self.nb_of_conv_layers_to_fine_tune,
                                     first_phase_trained_weights=self.first_phase_trained_weights,
                                     verbose=1)

            print('[INFO] HRA-2CLASS-ResNet50 model loaded')




        elif self.pre_trained_model == 'vgg16_places365':

            print('[INFO] Instantiating HRA-2CLASS-VGG16_Places365...')

            if self.train_mode == 'feature_extraction':

                model = HRA_VGG16_Places365(include_top=True, weights=None,
                                            input_tensor=None, input_shape=None,
                                            nb_of_conv_layers_to_fine_tune=self.nb_of_conv_layers_to_fine_tune,
                                            first_phase_trained_weights=None,
                                            verbose=1)
            else:

                if os.path.isfile(self.first_phase_trained_weights) is False:
                    raise IOError("No such weights file: `" + self.first_phase_trained_weights + "`. ")

                model = HRA_VGG16_Places365(include_top=True, weights=None,
                                            input_tensor=None, input_shape=None,
                                            nb_of_conv_layers_to_fine_tune=self.nb_of_conv_layers_to_fine_tune,
                                            first_phase_trained_weights=self.first_phase_trained_weights,
                                            verbose=1)

            print('[INFO] HRA-2CLASS-VGG16-Places365 model loaded')



        # Finally start fitting the dataset

        if self.train_mode == 'feature_extraction':
            print('[INFO] Start training the randomly initialised classifier on top of the pre-trained conv. base...')

            start_time = time.time()

            history = model.fit_generator(self.train_generator,
                                          epochs=self.nb_of_epochs,
                                          steps_per_epoch=self.steps_per_epoch,
                                          validation_data=self.val_generator,
                                          validation_steps=self.val_steps,
                                          callbacks=self.callbacks_list)

            end_time = time.time()
            print("[INFO] It took {} to train the randomly initialised classifier on top of the pre-trained conv. base".format(
                hms_string(end_time - start_time)))

            print('[INFO] Saved trained model as: %s ' % self.weights_to_file)

        else:
            print('[INFO] Start fine-tuning the model...')

            start_time = time.time()

            history = model.fit_generator(self.train_generator,
                                          epochs=self.nb_of_epochs,
                                          steps_per_epoch=self.steps_per_epoch,
                                          validation_data=self.val_generator,
                                          validation_steps=self.val_steps,
                                          callbacks=self.callbacks_list)

            end_time = time.time()
            print("[INFO] It took {} to fine-tune the top layers of the frozen conv. base".format(
                hms_string(end_time - start_time)))

            print('[INFO] Saved trained model as: %s ' % self.weights_to_file)

