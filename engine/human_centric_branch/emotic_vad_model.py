# -*- coding: utf-8 -*-
""" EMOTIC_VAD is the base class for instantiating various end-to-end models for continuous emotion recognition in VAD space
    using the EMOTIC dataset.

# Reference
- [Emotion Recognition in Context](http://sunai.uoc.edu/emotic/pdf/EMOTIC_cvpr2017.pdf)
- https://stackoverflow.com/questions/43452441/keras-all-layer-names-should-be-unique

"""


from __future__ import print_function
import numpy as np
import time

import h5py

from utils.generic_utils import hms_string
from functools import partial

from keras.engine.topology import get_source_inputs
from keras.models import Model
from keras.optimizers import SGD, RMSprop
from keras.applications.imagenet_utils import _obtain_input_shape
from keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization, Activation
from keras.layers.merge import concatenate
from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger
from keras.layers import Input
from keras import regularizers
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.applications.resnet50 import ResNet50
from applications.vgg16_places_365 import VGG16_Places365

from keras.losses import binary_crossentropy

from preprocessing.emotic.custom_generator import custom_generator, custom_generator_single_output
from utils.generic_utils import rmse, euclidean_distance_loss


class EMOTIC_VAD():
    """Loads the parameters needed for the training process on class instantiation
        & sets out the training process of various models (defined as different functions of the class) using the main `train` function.

    # Arguments
        hdf5_file: The HDF5 file containing the preprocessed images and their respective annotations.
        body_backbone_CNN: Truncated version of a CNN which takes as input the region of the image comprising
            the person who’s feelings are to be estimated. One of `VGG16`, `VGG19`, `ResNet50` or `VGG16_Places365`.
        image_backbone_CNN: Truncated version of a CNN which takes as input the the entire image.
            One of `VGG16`, `VGG19`, `ResNet50` or `VGG16_Places365`.
        nb_of_epochs: Integer, total number of iterations on the data.
        weights_to_file: File name or full path for saving the weights of the current training process.
        modelCheckpoint_quantity: Quantity to monitor when saving the model after every epoch is enabled.
        earlyStopping_quantity: Quantity to monitor when stopping training when a monitored quantity has stopped improving is enabled.
        CSVLogger_filename: filename of the csv file, where the CSVLogger callback will stream epoch results to.

    # Raises
        ValueError: in case of invalid argument for `body_backbone_CNN`
            or invalid argument for `image_backbone_CNN`.
    """


    def __init__(self,
                 hdf5_file,
                 body_backbone_CNN,
                 image_backbone_CNN,
                 nb_of_epochs,
                 weights_to_file,
                 modelCheckpoint_quantity,
                 earlyStopping_quantity,
                 CSVLogger_filename):


        if not (body_backbone_CNN in {'VGG16', 'VGG19', 'ResNet50', 'VGG16_Places365'}):
            raise ValueError('The `body_backbone_CNN` argument should be either '
                             '`VGG16`, `VGG19`, `ResNet50` or `VGG16_Places365`. ')

        if not (image_backbone_CNN in {'VGG16', 'VGG19', 'ResNet50', 'VGG16_Places365'}):
            raise ValueError('The `image_backbone_CNN` argument should be either '
                             '`VGG16`, `VGG19`, `ResNet50` or `VGG16_Places365`. ')

        self.body_backbone_CNN = body_backbone_CNN
        self.image_backbone_CNN = image_backbone_CNN

        # -------------------------------------------------------------------------------- #
        #                             Construct EMOTIC model
        # -------------------------------------------------------------------------------- #

        body_inputs = Input(shape=(224, 224, 3), name='INPUT')
        image_inputs = Input(shape=(224, 224, 3), name='INPUT')

        # Body module
        if 'VGG16' == body_backbone_CNN:
            self.body_truncated_model = VGG16(include_top=False, weights='imagenet', input_tensor=body_inputs, pooling='avg')

        elif 'VGG19' == body_backbone_CNN:
            self.body_truncated_model = VGG19(include_top=False, weights='imagenet', input_tensor=body_inputs, pooling='avg')

        elif 'ResNet50' == body_backbone_CNN:
            tmp_model = ResNet50(include_top=False, weights='imagenet', input_tensor=body_inputs, pooling='avg')
            self.body_truncated_model = Model(inputs=tmp_model.input, outputs=tmp_model.get_layer('activation_48').output)

        elif 'VGG16_Places365' == body_backbone_CNN:
            self.body_truncated_model = VGG16_Places365(include_top=False, weights='places', input_tensor=body_inputs, pooling='avg')

        for layer in self.body_truncated_model.layers:
            layer.name = str("body-") + layer.name


        print('[INFO] The plain, body `' + body_backbone_CNN + '` pre-trained convnet was successfully initialised.')

        # Image module
        if 'VGG16' == image_backbone_CNN:
            self.image_truncated_model = VGG16(include_top=False, weights='imagenet', input_tensor=image_inputs, pooling='avg')

        elif 'VGG19' == image_backbone_CNN:
            self.image_truncated_model = VGG19(include_top=False, weights='imagenet', input_tensor=image_inputs, pooling='avg')

        elif 'ResNet50' == image_backbone_CNN:
            tmp_model = ResNet50(include_top=False, weights='imagenet',input_tensor=image_inputs, pooling='avg')
            self.image_truncated_model = Model(inputs=tmp_model.input, outputs=tmp_model.get_layer('activation_48').output)

        elif 'VGG16_Places365' == image_backbone_CNN:
            self.image_truncated_model = VGG16_Places365(include_top=False, weights='places', input_tensor=image_inputs, pooling='avg')

        for layer in self.image_truncated_model.layers:
            layer.name = str("image-") + layer.name

        print('[INFO] The plain, image `' + image_backbone_CNN + '` pre-trained convnet was successfully initialised.')

        # retrieve the ouputs
        body_plain_model_output = self.body_truncated_model.output
        image_plain_model_output = self.image_truncated_model.output


        # In case ResNet50 is selected we need to use a global average pooling layer to follow the process used for the othe CNNs.
        if 'ResNet50' == body_backbone_CNN:
            body_plain_model_output = GlobalAveragePooling2D(name='GAP')(body_plain_model_output)

        if 'ResNet50' == image_backbone_CNN:
            image_plain_model_output = GlobalAveragePooling2D(name='GAP')(image_plain_model_output)

        merged = concatenate([body_plain_model_output, image_plain_model_output])

        x = Dense(256, activation='relu', name='FC1', kernel_regularizer=regularizers.l2(0.01), kernel_initializer='random_normal')(merged)

        x = Dropout(0.5, name='DROPOUT')(x)

        vad_cont_prediction = Dense(units=3, kernel_initializer='random_normal', name='VAD')(x)

        # At model instantiation, you specify the two inputs and the output.
        self.model = Model(inputs=[body_inputs, image_inputs], outputs=vad_cont_prediction, name='EMOTIC-VAD-regression')

        print('[INFO] Randomly initialised classifier was successfully added on top of the merged modules.')

        print('[INFO] Number of trainable weights before freezing the conv. bases of the respective original models: '
              '' + str(len(self.model.trainable_weights)))

        # first: train only the top layers (which were randomly initialized)
        # i.e. freeze all convolutional layers of the preliminary base model
        for layer in self.body_truncated_model.layers:
            layer.trainable = False

        for layer in self.image_truncated_model.layers:
            layer.trainable = False

        print('[INFO] Number of trainable weights after freezing the conv. bases of the respective original models: '
              '' + str(len(self.model.trainable_weights)))

        # # reference https://github.com/keras-team/keras/issues/4735#issuecomment-267472549
        # self.class_weight = { 'VALENCE': {0: 36.00, 1: 36.00, 2: 12.00, 3: 5.14, 4: 2.25, 5: 1.00, 6: 1.89, 7: 2.57, 8: 12.00, 9: 36.00},
        #                       'AROUSAL': {0: 23.00, 1: 11.50, 2: 4.60, 3: 1.00, 4: 2.09, 5: 1.64, 6: 1.14, 7: 2.09, 8: 3.83, 9: 4.60},
        #                       'DOMINANCE': {0: 34.00, 1: 17.00, 2: 11.33, 3: 6.80, 4: 5.66, 5: 1.70, 6: 1.00, 7: 2.42, 8: 3.40, 9: 6.80}
        #                     }


        self.model.compile(optimizer=SGD(lr=1e-5, momentum=0.9),
                           # loss='mse',
                           loss = euclidean_distance_loss,
                           metrics=['mae','mse', rmse])

        # print ('[INFO] Metrics names: ',self.model.metrics_names )

        print('[INFO] End-to-end `EMOTIC-VAD-regression` model has been successfully compiled.')

        # -------------------------------------------------------------------------------- #
        #                                    Configurations
        # -------------------------------------------------------------------------------- #


        nb_train_samples = 23706
        nb_val_samples = 3332
        nb_test_samples = 7280

        train_generator_batch_size = 54
        val_generator_batch_size = 49
        test_generator_batch_size = 52

        self.steps_per_epoch = nb_train_samples // train_generator_batch_size
        self.validation_steps = nb_val_samples // val_generator_batch_size
        self.test_steps = nb_test_samples // test_generator_batch_size


        # -------------------------------------------------------------------------------- #
        #                                Read the HDF5 file
        # -------------------------------------------------------------------------------- #
        # open the hdf5 file
        hdf5_file = h5py.File(hdf5_file, "r")

        self.nb_train_data = hdf5_file["x_image_train"].shape[0]

        self.nb_val_data = hdf5_file["x_image_val"].shape[0]

        self.nb_test_data = hdf5_file["x_image_test"].shape[0]



        # -------------------------------------------------------------------------------- #
        #                         Instantiate the custom generators
        # -------------------------------------------------------------------------------- #

        print('[INFO] Setting up custom generators...')

        self.train_generator = custom_generator_single_output(hdf5_file=hdf5_file,
                                                              nb_data=self.nb_train_data,
                                                              batch_size=train_generator_batch_size,
                                                              mode='train')

        self.val_generator = custom_generator_single_output(hdf5_file=hdf5_file,
                                                            nb_data=self.nb_val_data,
                                                            batch_size=val_generator_batch_size,
                                                            mode='val')

        self.test_generator = custom_generator_single_output(hdf5_file=hdf5_file,
                                                             nb_data=self.nb_test_data,
                                                             batch_size=test_generator_batch_size,
                                                             mode='test')




        # -------------------------------------------------------------------------------- #
        #                                Usage of callbacks
        # -------------------------------------------------------------------------------- #

        self.weights_to_file = weights_to_file
        self.nb_of_epochs = nb_of_epochs

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
        """Trains the EMOTIC model for a given number of epochs (iterations on a dataset).

        """

        self.model.summary()

        print('[INFO] Start training the end-to-end EMOTIC model...')

        start_time = time.time()

        history = self.model.fit_generator(self.train_generator,
                                           epochs=self.nb_of_epochs,
                                           steps_per_epoch=self.steps_per_epoch,
                                           validation_data=self.val_generator,
                                           validation_steps=self.validation_steps,
                                           callbacks=self.callbacks_list,
                                           # class_weight= self.class_weight
                                           )

        end_time = time.time()
        print("[INFO] It took {} to train the end-to-end EMOTIC model".format(hms_string(end_time - start_time)))

        print('[INFO] Saved trained model as: %s ' % self.weights_to_file)


