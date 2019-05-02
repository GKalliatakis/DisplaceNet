"""Python utility which reads image names (locations on disk) from a csv file,
    loads them using the basic set of tools for image data provided by Keras (keras/preprocessing/image.py)
    and saves them in a numpy array.
"""

import numpy as np
import pandas
from utils.generic_utils import progress
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input


# reference https://stackoverflow.com/questions/12984426/python-pil-ioerror-image-file-truncated-with-big-images
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


def save_img_to_numpy(base_img_dir,
                      base_csv_dir,
                      input_size,
                      mode='train',
                      to_file = 'numpy_annotations/x_train'
                      ):
    """ Saves images loaded from a CSV to numpy array.

        # Arguments
            base_img_dir: the directory where the raw images are stored.
            In our setup, we:
            - created train/ val/ and test/ subfolders inside EMOTIC_database/

            base_csv_dir: the directory where the CSV files are stored.
            input_size: the default input size for the model (ref https://keras.io/applications/).
                All models have input size of 224x224 except Xception,InceptionV3 and InceptionResNetV2 which have input size of 299x299.
            mode: one of `train` (train set), `val` (validation set)
                or `test` (test set).
            to_file: the name or path of the numpy array where the images will be saved.
        """


    # Load CSV File With Pandas
    csv_name = base_csv_dir + mode + '.csv'
    csv_file = pandas.read_csv(csv_name)

    if mode == 'train':
        nb_samples = 23706
    elif mode == 'val':
        nb_samples = 3332
    elif mode == 'test':
        nb_samples = 7280

    field_number = 0

    # pre-allocating the data array, and then loading the data directly into it
    # ref: https://hjweide.github.io/efficient-image-loading
    data = np.empty((nb_samples, input_size, input_size, 3), dtype=np.uint8)

    for img_name in csv_file.filename:
        progress(field_number, nb_samples)

        img_name = base_img_dir + img_name
        img = image.load_img(img_name, target_size=(input_size, input_size))

        x = image.img_to_array(img) # this is a Numpy array with shape (input_size, input_size, 3)
        x = np.expand_dims(x, axis=0) # this is a Numpy array with shape (1, input_size, input_size, 3)
        x = preprocess_input(x)

        data[field_number, ...] = x

        field_number += 1
        if field_number > nb_samples - 1:
            break

    np.save(to_file, data)

    return data


if __name__ == '__main__':

    # x_train= save_img_to_numpy(base_img_dir ='/home/sandbox/Desktop/EMOTIC_database/',
    #                            base_csv_dir = '/home/sandbox/Desktop/Keras-EMOTIC/dataset/',
    #                            input_size = 299,
    #                            mode='test',
    #                            to_file = 'x_test')



    x_train = save_img_to_numpy(base_img_dir='/home/sandbox/Desktop/EMOTIC_resources/v0.3_divisible/entire_multiple_imgs/',
                                base_csv_dir='/home/sandbox/Desktop/EMOTIC_resources/v0.3_divisible/',
                                input_size=224,
                                mode='val',
                                to_file='x_entire_val')
