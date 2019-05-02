from __future__ import print_function
from preprocessing.load_data_from_numpy import load_data_from_numpy,load_data_from_numpy_single_output
from math import ceil
from utils.generic_utils import progress

import numpy as np
import pandas
import cv2
import h5py
import matplotlib.pyplot as plt


class Controller():

    def __init__(self,
                 hdf5_file,
                 train_csv_file_path,
                 val_csv_file_path,
                 test_csv_file_path,
                 cropped_imgs_dir,
                 entire_imgs_dir,
                 main_numpy_dir
                 ):
        """HDF5 controller base class.
                It can be used either to create a single HDF5 file
                containing a large number of images and their respective annotations (from EMOTIC Dataset)
                or to load a previously saved HDF5 file.

                In order to create a single HDF5 file, images (tensors) and their annotations must be available
                in numpy arrays in the following structure:

                main_numpy_dir/
                            train/
                                x_train.npy
                                emotions_train.npy
                                valence_train.npy
                                arousal_train.npy
                                dominance_train.npy
                                age_train.npy

                            val/
                                x_val.npy
                                emotions_val.npy
                                valence_val.npy
                                arousal_val.npy
                                dominance_val.npy
                                age_val.npy

                            test/
                                x_test.npy
                                emotions_test.npy
                                valence_test.npy
                                arousal_test.npy
                                dominance_test.npy
                                age_test.npy


                Also, `base_img_dir` must contain the raw images in the following structure:

                base_img_dir/
                            train/
                                images/
                                    xxxxxxxx.jpg
                                    xxxxxxxx.jpg
                                    ...


                            val/
                                images/
                                    xxxxxxxx.jpg
                                    xxxxxxxx.jpg
                                    ...

                            test/
                                images/
                                    xxxxxxxx.jpg
                                    xxxxxxxx.jpg
                                    ...

                Note that in order to end up with that structure you will need either download the images from
                - https://github.com/GKalliatakis/Keras-EMOTIC/releases/download/0.1/train.zip
                - https://github.com/GKalliatakis/Keras-EMOTIC/releases/download/0.1/val.zip
                - https://github.com/GKalliatakis/Keras-EMOTIC/releases/download/0.1/test.zip
                or recreate them using the `export_annotations_diff_modes.py` found in `EMOTIC_database` project.

        """

        self.hdf5_file = hdf5_file
        self.train_csv_file = pandas.read_csv(train_csv_file_path)
        self.val_csv_file = pandas.read_csv(val_csv_file_path)
        self.test_csv_file = pandas.read_csv(test_csv_file_path)
        #
        # (self.x_entire_train, self.x_cropped_train, self.valence_entire_train, self.valence_cropped_train, self.arousal_entire_train,
        #  self.arousal_cropped_train, self.dominance_entire_train, self.dominance_cropped_train), \
        # (self.x_entire_val, self.x_cropped_val, self.valence_entire_val, self.valence_cropped_val, self.arousal_entire_val, self.arousal_cropped_val,
        #  self.dominance_entire_val, self.dominance_cropped_val), \
        # (self.x_entire_test, self.x_cropped_test, self.valence_entire_test, self.valence_cropped_test, self.arousal_entire_test,
        #  self.arousal_cropped_test, self.dominance_entire_test, self.dominance_cropped_test) = load_data_from_numpy(main_numpy_dir=main_numpy_dir,verbose=1)

        (self.x_image_train, self.x_body_train, self.y_image_train, self.y_body_train), \
        (self.x_image_val, self.x_body_val, self.y_image_val, self.y_body_val), \
        (self.x_image_test, self.x_body_test, self.y_image_test, self.y_body_test) = load_data_from_numpy_single_output(main_numpy_dir=main_numpy_dir,verbose=1)

        self.cropped_imgs_dir = cropped_imgs_dir
        self.entire_imgs_dir = entire_imgs_dir





    def create_hdf5_VAD_classification(self, dataset, input_size):
        """ Saves a large number of images and their respective annotations (from EMOTIC Dataset) in a single HDF5 file.
            # Reference
                - http://machinelearninguru.com/deep_learning/data_preparation/hdf5/hdf5.html
                - http://sunai.uoc.edu/emotic/

            # Arguments
                dataset: name of the dataset that the HDF5 file will be created for.
                input_size: the default input size for the model (ref https://keras.io/applications/).
                    All models have input size of 224x224
                    except Xception,InceptionV3 and InceptionResNetV2 which have input size of 299x299.
        """

        if not (dataset in {'EMOTIC'}):
            raise ValueError('The `dataset` argument can be set to `EMOTIC` only. '
                             'More datasets will be added in future releases.')


        if dataset == 'EMOTIC':
            nb_train_samples = 23706
            nb_val_samples = 3332
            nb_test_samples = 7280

        train_shape = (nb_train_samples, input_size, input_size, 3)
        val_shape = (nb_val_samples, input_size, input_size, 3)
        test_shape = (nb_test_samples, input_size, input_size, 3)

        print('[INFO] Open the hdf5 file `'+ str(self.hdf5_file) +'` and start creating arrays.')

        # open a hdf5 file and create arrays
        hdf5_file = h5py.File(self.hdf5_file, mode='w')
        hdf5_file.create_dataset("x_entire_train", train_shape, np.uint8)
        hdf5_file.create_dataset("x_entire_val", val_shape, np.uint8)
        hdf5_file.create_dataset("x_entire_test", test_shape, np.uint8)

        hdf5_file.create_dataset("x_cropped_train", train_shape, np.uint8)
        hdf5_file.create_dataset("x_cropped_val", val_shape, np.uint8)
        hdf5_file.create_dataset("x_cropped_test", test_shape, np.uint8)

        # train arrays
        hdf5_file.create_dataset("valence_entire_train", (nb_train_samples, 10), np.float64)
        hdf5_file["valence_entire_train"][...] = self.valence_entire_train

        hdf5_file.create_dataset("arousal_entire_train", (nb_train_samples, 10), np.float64)
        hdf5_file["arousal_entire_train"][...] = self.arousal_entire_train

        hdf5_file.create_dataset("dominance_entire_train", (nb_train_samples, 10), np.float64)
        hdf5_file["dominance_entire_train"][...] = self.dominance_entire_train


        hdf5_file.create_dataset("valence_cropped_train", (nb_train_samples, 10), np.float64)
        hdf5_file["valence_cropped_train"][...] = self.valence_cropped_train

        hdf5_file.create_dataset("arousal_cropped_train", (nb_train_samples, 10), np.float64)
        hdf5_file["arousal_cropped_train"][...] = self.arousal_cropped_train

        hdf5_file.create_dataset("dominance_cropped_train", (nb_train_samples, 10), np.float64)
        hdf5_file["dominance_cropped_train"][...] = self.dominance_cropped_train



        # val arrays
        hdf5_file.create_dataset("valence_entire_val", (nb_val_samples, 10), np.float64)
        hdf5_file["valence_entire_val"][...] = self.valence_entire_val

        hdf5_file.create_dataset("arousal_entire_val", (nb_val_samples, 10), np.float64)
        hdf5_file["arousal_entire_val"][...] = self.arousal_entire_val

        hdf5_file.create_dataset("dominance_entire_val", (nb_val_samples, 10), np.float64)
        hdf5_file["dominance_entire_val"][...] = self.dominance_entire_val

        hdf5_file.create_dataset("valence_cropped_val", (nb_val_samples, 10), np.float64)
        hdf5_file["valence_cropped_val"][...] = self.valence_cropped_val

        hdf5_file.create_dataset("arousal_cropped_val", (nb_val_samples, 10), np.float64)
        hdf5_file["arousal_cropped_val"][...] = self.arousal_cropped_val

        hdf5_file.create_dataset("dominance_cropped_val", (nb_val_samples, 10), np.float64)
        hdf5_file["dominance_cropped_val"][...] = self.dominance_cropped_val


        # test arrays
        hdf5_file.create_dataset("valence_entire_test", (nb_test_samples, 10), np.float64)
        hdf5_file["valence_entire_test"][...] = self.valence_entire_test

        hdf5_file.create_dataset("arousal_entire_test", (nb_test_samples, 10), np.float64)
        hdf5_file["arousal_entire_test"][...] = self.arousal_entire_test

        hdf5_file.create_dataset("dominance_entire_test", (nb_test_samples, 10), np.float64)
        hdf5_file["dominance_entire_test"][...] = self.dominance_entire_test

        hdf5_file.create_dataset("valence_cropped_test", (nb_test_samples, 10), np.float64)
        hdf5_file["valence_cropped_test"][...] = self.valence_cropped_test

        hdf5_file.create_dataset("arousal_cropped_test", (nb_test_samples, 10), np.float64)
        hdf5_file["arousal_cropped_test"][...] = self.arousal_cropped_test

        hdf5_file.create_dataset("dominance_cropped_test", (nb_test_samples, 10), np.float64)
        hdf5_file["dominance_cropped_test"][...] = self.dominance_cropped_test

        print('[INFO] Arrays have been created.')


        field_number = 0
        print('[INFO] Start reading cropped images from train set.')
        # loop over cropped images train addresses
        for img_name in self.train_csv_file.filename:
            progress(field_number, nb_train_samples)

            img_name = self.cropped_imgs_dir + img_name

            img = cv2.imread(img_name)
            img = cv2.resize(img, (input_size, input_size), interpolation=cv2.INTER_CUBIC)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # save the image and calculate the mean so far
            hdf5_file["x_cropped_train"][field_number, ...] = img[None]

            field_number += 1
            if field_number > nb_train_samples - 1:
                break


        print('[INFO] Start reading entire images from train set.')
        field_number = 0
        # loop over entire images train addresses
        for img_name in self.train_csv_file.filename:
            progress(field_number, nb_train_samples)

            img_name = self.entire_imgs_dir + img_name

            img = cv2.imread(img_name)
            img = cv2.resize(img, (input_size, input_size), interpolation=cv2.INTER_CUBIC)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # save the image and calculate the mean so far
            hdf5_file["x_entire_train"][field_number, ...] = img[None]

            field_number += 1
            if field_number > nb_train_samples - 1:
                break



        print('[INFO] Start reading cropped images from validation set.')
        field_number = 0
        # loop over val addresses
        for img_name in self.val_csv_file.filename:
            progress(field_number, nb_val_samples)

            img_name = self.cropped_imgs_dir + img_name

            img = cv2.imread(img_name)
            img = cv2.resize(img, (input_size, input_size), interpolation=cv2.INTER_CUBIC)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # save the image and calculate the mean so far
            hdf5_file["x_cropped_val"][field_number, ...] = img[None]

            field_number += 1
            if field_number > nb_val_samples - 1:
                break


        print('[INFO] Start reading entire images from validation set.')
        field_number = 0
        # loop over val addresses
        for img_name in self.val_csv_file.filename:
            progress(field_number, nb_val_samples)

            img_name = self.entire_imgs_dir + img_name

            img = cv2.imread(img_name)
            img = cv2.resize(img, (input_size, input_size), interpolation=cv2.INTER_CUBIC)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # save the image and calculate the mean so far
            hdf5_file["x_entire_val"][field_number, ...] = img[None]

            field_number += 1
            if field_number > nb_val_samples - 1:
                break


        print('[INFO] Start reading cropped images from test set.')
        field_number = 0
        # loop over val addresses
        for img_name in self.test_csv_file.filename:
            progress(field_number, nb_test_samples)

            img_name = self.cropped_imgs_dir + img_name

            img = cv2.imread(img_name)
            img = cv2.resize(img, (input_size, input_size), interpolation=cv2.INTER_CUBIC)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # save the image and calculate the mean so far
            hdf5_file["x_cropped_test"][field_number, ...] = img[None]

            field_number += 1
            if field_number > nb_test_samples - 1:
                break


        print('[INFO] Start reading entire images from test set.')
        field_number = 0
        # loop over val addresses
        for img_name in self.test_csv_file.filename:
            progress(field_number, nb_test_samples)

            img_name = self.entire_imgs_dir + img_name

            img = cv2.imread(img_name)
            img = cv2.resize(img, (input_size, input_size), interpolation=cv2.INTER_CUBIC)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # save the image and calculate the mean so far
            hdf5_file["x_entire_test"][field_number, ...] = img[None]

            field_number += 1
            if field_number > nb_test_samples - 1:
                break


        hdf5_file.close()




    def create_hdf5_VAD_regression(self, dataset, input_size):
        """ Saves a large number of images and their respective annotations (from EMOTIC Dataset) in a single HDF5 file.
            # Reference
                - http://machinelearninguru.com/deep_learning/data_preparation/hdf5/hdf5.html
                - http://sunai.uoc.edu/emotic/

            # Arguments
                dataset: name of the dataset that the HDF5 file will be created for.
                input_size: the default input size for the model (ref https://keras.io/applications/).
                    All models have input size of 224x224
                    except Xception,InceptionV3 and InceptionResNetV2 which have input size of 299x299.
        """

        if not (dataset in {'EMOTIC'}):
            raise ValueError('The `dataset` argument can be set to `EMOTIC` only. '
                             'More datasets will be added in future releases.')


        if dataset == 'EMOTIC':
            nb_train_samples = 23706
            nb_val_samples = 3332
            nb_test_samples = 7280

        train_shape = (nb_train_samples, input_size, input_size, 3)
        val_shape = (nb_val_samples, input_size, input_size, 3)
        test_shape = (nb_test_samples, input_size, input_size, 3)

        print('[INFO] Open the hdf5 file `'+ str(self.hdf5_file) +'` and start creating arrays.')

        # open a hdf5 file and create arrays
        hdf5_file = h5py.File(self.hdf5_file, mode='w')
        hdf5_file.create_dataset("x_image_train", train_shape, np.uint8)
        hdf5_file.create_dataset("x_image_val", val_shape, np.uint8)
        hdf5_file.create_dataset("x_image_test", test_shape, np.uint8)

        hdf5_file.create_dataset("x_body_train", train_shape, np.uint8)
        hdf5_file.create_dataset("x_body_val", val_shape, np.uint8)
        hdf5_file.create_dataset("x_body_test", test_shape, np.uint8)

        # train arrays
        hdf5_file.create_dataset("y_image_train", (nb_train_samples, 3), np.uint8)
        hdf5_file["y_image_train"][...] = self.y_image_train

        hdf5_file.create_dataset("y_body_train", (nb_train_samples, 3), np.uint8)
        hdf5_file["y_body_train"][...] = self.y_body_train


        # val arrays
        hdf5_file.create_dataset("y_image_val", (nb_val_samples, 3), np.uint8)
        hdf5_file["y_image_val"][...] = self.y_image_val

        hdf5_file.create_dataset("y_body_val", (nb_val_samples, 3), np.uint8)
        hdf5_file["y_body_val"][...] = self.y_body_val


        # test arrays
        hdf5_file.create_dataset("y_image_test", (nb_test_samples, 3), np.uint8)
        hdf5_file["y_image_test"][...] = self.y_image_test

        hdf5_file.create_dataset("y_body_test", (nb_test_samples, 3), np.uint8)
        hdf5_file["y_body_test"][...] = self.y_body_test

        print('[INFO] Arrays have been created.')


        field_number = 0
        print('[INFO] Start reading body images from train set.')
        # loop over cropped images train addresses
        for img_name in self.train_csv_file.filename:
            progress(field_number, nb_train_samples)

            img_name = self.cropped_imgs_dir + img_name

            img = cv2.imread(img_name)
            img = cv2.resize(img, (input_size, input_size), interpolation=cv2.INTER_CUBIC)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # save the image and calculate the mean so far
            hdf5_file["x_body_train"][field_number, ...] = img[None]

            field_number += 1
            if field_number > nb_train_samples - 1:
                break


        print('[INFO] Start reading entire images from train set.')
        field_number = 0
        # loop over entire images train addresses
        for img_name in self.train_csv_file.filename:
            progress(field_number, nb_train_samples)

            img_name = self.entire_imgs_dir + img_name

            img = cv2.imread(img_name)
            img = cv2.resize(img, (input_size, input_size), interpolation=cv2.INTER_CUBIC)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # save the image and calculate the mean so far
            hdf5_file["x_image_train"][field_number, ...] = img[None]

            field_number += 1
            if field_number > nb_train_samples - 1:
                break



        print('[INFO] Start reading body images from validation set.')
        field_number = 0
        # loop over val addresses
        for img_name in self.val_csv_file.filename:
            progress(field_number, nb_val_samples)

            img_name = self.cropped_imgs_dir + img_name

            img = cv2.imread(img_name)
            img = cv2.resize(img, (input_size, input_size), interpolation=cv2.INTER_CUBIC)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # save the image and calculate the mean so far
            hdf5_file["x_body_val"][field_number, ...] = img[None]

            field_number += 1
            if field_number > nb_val_samples - 1:
                break


        print('[INFO] Start reading entire images from validation set.')
        field_number = 0
        # loop over val addresses
        for img_name in self.val_csv_file.filename:
            progress(field_number, nb_val_samples)

            img_name = self.entire_imgs_dir + img_name

            img = cv2.imread(img_name)
            img = cv2.resize(img, (input_size, input_size), interpolation=cv2.INTER_CUBIC)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # save the image and calculate the mean so far
            hdf5_file["x_image_val"][field_number, ...] = img[None]

            field_number += 1
            if field_number > nb_val_samples - 1:
                break


        print('[INFO] Start reading body images from test set.')
        field_number = 0
        # loop over val addresses
        for img_name in self.test_csv_file.filename:
            progress(field_number, nb_test_samples)

            img_name = self.cropped_imgs_dir + img_name

            img = cv2.imread(img_name)
            img = cv2.resize(img, (input_size, input_size), interpolation=cv2.INTER_CUBIC)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # save the image and calculate the mean so far
            hdf5_file["x_body_test"][field_number, ...] = img[None]

            field_number += 1
            if field_number > nb_test_samples - 1:
                break


        print('[INFO] Start reading entire images from test set.')
        field_number = 0
        # loop over val addresses
        for img_name in self.test_csv_file.filename:
            progress(field_number, nb_test_samples)

            img_name = self.entire_imgs_dir + img_name

            img = cv2.imread(img_name)
            img = cv2.resize(img, (input_size, input_size), interpolation=cv2.INTER_CUBIC)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # save the image and calculate the mean so far
            hdf5_file["x_image_test"][field_number, ...] = img[None]

            field_number += 1
            if field_number > nb_test_samples - 1:
                break


        hdf5_file.close()