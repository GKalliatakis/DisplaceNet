"""Python utility which reads entries (different annotations) from a csv file,
    converts them to integer/list of integers(for multilabel-multiclass settings) and saves them in a numpy array.
"""

import pandas
import numpy as np
import math
from keras.utils import np_utils


def multi_output(csv_path,
                 nb_samples,
                 entry_type,
                 to_file):
    """Reads entries (different annotations) from a csv file,
        converts them to integer/list of integers(for multilabel-multiclass settings) and saves them in a numpy array.

        # Arguments
            csv_path: String to declare the full path of the csv file .
            nb_samples: Number of entries in the csv file to iterate.
            entry_type: The header name of the column to process. One of `emotions`, `valence`, `arousal`, `dominance` and `age`.
            to_file: File name of the numpy array that will hold the converted entries.
        """

    csv_file = pandas.read_csv(csv_path)

    # counter to iterate through all csv entries
    field_number = 0

    final_list = []

    if entry_type == 'valence':
        for entry in csv_file.valence:

            # convert the retrieved csv entry (whose type is always str) into integer/list of integers
            # int_list = map(int, entry.split(','))
            #
            # int_list = np_utils.to_categorical(int_list)

            # append the converted integer/list of integers to the final list
            # we need to subtrack 1 in order to transform the paper's 1-10 scale to pythonic way 0-9
            final_list.append(entry-1)


            field_number += 1
            if field_number > nb_samples - 1:
                break

    elif entry_type == 'arousal':
        for entry in csv_file.arousal:

            # convert the retrieved csv entry (whose type is always str) into integer/list of integers
            # int_list = map(int, entry.split(','))
            #
            # int_list = np_utils.to_categorical(int_list)

            # append the converted integer/list of integers to the final list
            # we need to subtrack 1 in order to transform the paper's 1-10 scale to pythonic way 0-9
            final_list.append(entry - 1)

            field_number += 1
            if field_number > nb_samples - 1:
                break


    elif entry_type == 'dominance':
        for entry in csv_file.dominance:

            # convert the retrieved csv entry (whose type is always str) into integer/list of integers
            # int_list = map(int, entry.split(','))
            #
            # int_list = np_utils.to_categorical(int_list)

            # append the converted integer/list of integers to the final list
            # we need to subtrack 1 in order to transform the paper's 1-10 scale to pythonic way 0-9
            final_list.append(entry - 1)

            field_number += 1
            if field_number > nb_samples - 1:
                break

    # expand dimensions from (xxxx,) to(xxxx, 1)
    # final_list = np.expand_dims(final_list, axis=1)
    np.save(to_file, final_list)


def three_neurons_single_output(csv_path,
                                nb_samples,
                                entry_type,
                                to_file):
    """Reads entries (different annotations) from a csv file,
        converts them to integer/list of integers(for multilabel-multiclass settings) and saves them in a numpy array.

        # Arguments
            csv_path: String to declare the full path of the csv file .
            nb_samples: Number of entries in the csv file to iterate.
            entry_type: The header name of the column to process. One of `emotions`, `valence`, `arousal`, `dominance` and `age`.
            to_file: File name of the numpy array that will hold the converted entries.
        """

    csv_file = pandas.read_csv(csv_path)

    # counter to iterate through all csv entries
    field_number = 0

    final_list = []

    if entry_type == 'VAD':
        for entry in csv_file.VAD:

            # convert the retrieved csv entry (whose type is always str) into integer/list of integers
            int_list = map(int, entry.split(','))

            # int_list = np_utils.to_categorical(int_list)

            # append the converted integer/list of integers to the final list
            final_list.append(int_list)


            field_number += 1
            if field_number > nb_samples - 1:
                break



    # expand dimensions from (xxxx,) to(xxxx, 1)
    # final_list = np.expand_dims(final_list, axis=1)
    np.save(to_file, final_list)




if __name__ == '__main__':
    #
    # # for multi-output model
    #
    # mode = 'val'
    # entry_type = 'valence'
    #
    #
    #
    # to_file = entry_type + '_'+mode
    # csv_path= '/home/sandbox/Desktop/EMOTIC_resources/VAD-classification/labels/'+ mode +'.csv'
    #
    #
    # if mode == 'train':
    #     nb_samples = 23706
    # elif mode == 'val':
    #     nb_samples = 3332
    # elif mode == 'test':
    #     nb_samples = 7280
    #
    #
    #
    #
    # multi_output(csv_path =csv_path,
    #              nb_samples = nb_samples,
    #              entry_type = entry_type,
    #              to_file= to_file)
    #
    #
    # x = np.load(str(to_file)+'.npy')
    #
    # print x.shape
    #
    # # print x[2]
    #
    # from keras.utils import np_utils
    #
    # x = np_utils.to_categorical(x)
    #
    #
    # np.save(to_file, x)
    #
    # x = np.load(str(to_file) + '.npy')
    #
    # print x.shape



    # for three neurons single output model

    mode = 'test'
    entry_type = 'VAD'

    to_file = 'y_' + mode +'.npy'

    csv_path= '/home/sandbox/Desktop/EMOTIC_resources/VAD-classification/CSV/'+ mode +'.csv'


    if mode == 'train':
        nb_samples = 23706
    elif mode == 'val':
        nb_samples = 3332
    elif mode == 'test':
        nb_samples = 7280


    three_neurons_single_output(csv_path=csv_path,
                                nb_samples=nb_samples,
                                entry_type=entry_type,
                                to_file=to_file)


    x = np.load(str(to_file))

    print x.shape

    print x[0]