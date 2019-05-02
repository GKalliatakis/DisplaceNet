"""Python utilities required to load data (image & their annotations) stored in numpy arrays.
    Functions `load_numpy_arrays_single_output` & `load_numpy_arrays_emotions_age_only` are deprecated.
    Use either the main function `load_data_from_numpy` to load all the applicable arrays
    or the supporting `load_annotations_only_from_numpy` instead.
"""

from __future__ import print_function
import numpy as np

def load_data_from_numpy(main_numpy_dir,
                         verbose = 1):

    print ('[INFO] Loading data from numpy arrays...')

    x_entire_train = np.load(main_numpy_dir + 'X_train/x_entire_train.npy')
    x_cropped_train = np.load(main_numpy_dir + 'X_train/x_cropped_train.npy')

    valence_entire_train = np.load(main_numpy_dir + 'Y_train/valence_train.npy')
    valence_cropped_train = np.load(main_numpy_dir + 'Y_train/valence_train.npy')

    arousal_entire_train = np.load(main_numpy_dir + 'Y_train/arousal_train.npy')
    arousal_cropped_train = np.load(main_numpy_dir + 'Y_train/arousal_train.npy')

    dominance_entire_train = np.load(main_numpy_dir + 'Y_train/dominance_train.npy')
    dominance_cropped_train = np.load(main_numpy_dir + 'Y_train/dominance_train.npy')



    x_entire_val = np.load(main_numpy_dir + 'X_train/x_entire_val.npy')
    x_cropped_val = np.load(main_numpy_dir + 'X_train/x_cropped_val.npy')



    valence_entire_val = np.load(main_numpy_dir + 'Y_train/valence_val.npy')
    valence_cropped_val = np.load(main_numpy_dir + 'Y_train/valence_val.npy')

    arousal_entire_val = np.load(main_numpy_dir + 'Y_train/arousal_val.npy')
    arousal_cropped_val = np.load(main_numpy_dir + 'Y_train/arousal_val.npy')

    dominance_entire_val = np.load(main_numpy_dir + 'Y_train/dominance_val.npy')
    dominance_cropped_val = np.load(main_numpy_dir + 'Y_train/dominance_val.npy')



    x_entire_test = np.load(main_numpy_dir + 'X_train/x_entire_test.npy')
    x_cropped_test = np.load(main_numpy_dir + 'X_train/x_cropped_test.npy')

    valence_entire_test = np.load(main_numpy_dir + 'Y_train/valence_test.npy')
    valence_cropped_test = np.load(main_numpy_dir + 'Y_train/valence_test.npy')

    arousal_entire_test = np.load(main_numpy_dir + 'Y_train/arousal_test.npy')
    arousal_cropped_test = np.load(main_numpy_dir + 'Y_train/arousal_test.npy')

    dominance_entire_test = np.load(main_numpy_dir + 'Y_train/dominance_test.npy')
    dominance_cropped_test = np.load(main_numpy_dir + 'Y_train/dominance_test.npy')



    print('[INFO] Data have been successfully loaded')
    print('---------------------------------------------------------------------------------------------------')
    if verbose == 1:
        print('x_entire_train shape:', x_entire_train.shape)
        print('x_cropped_train shape:', x_cropped_train.shape)
        print('valence_entire_train shape:', valence_entire_train.shape)
        print('valence_cropped_train shape:', valence_cropped_train.shape)
        print('arousal_entire_train shape:', arousal_entire_train.shape)
        print('arousal_cropped_train shape:', arousal_cropped_train.shape)
        print('dominance_entire_train shape:', dominance_entire_train.shape)
        print('dominance_cropped_train shape:', dominance_cropped_train.shape)

        print ('---------------------------------------------------------------------------------------------------')

        print('x_entire_val shape:', x_entire_val.shape)
        print('x_cropped_val shape:', x_cropped_val.shape)
        print('valence_entire_val shape:', valence_entire_val.shape)
        print('valence_cropped_val shape:', valence_cropped_val.shape)
        print('arousal_entire_val shape:', arousal_entire_val.shape)
        print('arousal_cropped_val shape:', arousal_cropped_val.shape)
        print('dominance_entire_val shape:', dominance_entire_val.shape)
        print('dominance_cropped_val shape:', dominance_cropped_val.shape)

        print ('---------------------------------------------------------------------------------------------------')

        print('x_entire_test shape:', x_entire_test.shape)
        print('x_cropped_test shape:', x_cropped_test.shape)
        print('valence_entire_test shape:', valence_entire_test.shape)
        print('valence_cropped_test shape:', valence_cropped_test.shape)
        print('arousal_entire_test shape:', arousal_entire_test.shape)
        print('arousal_cropped_test shape:', arousal_cropped_test.shape)
        print('dominance_entire_test shape:', dominance_entire_test.shape)
        print('dominance_cropped_test shape:', dominance_cropped_test.shape)

        print ('---------------------------------------------------------------------------------------------------')

    return (x_entire_train, x_cropped_train,valence_entire_train,valence_cropped_train,arousal_entire_train,arousal_cropped_train,dominance_entire_train,dominance_cropped_train), \
           (x_entire_val, x_cropped_val, valence_entire_val,valence_cropped_val,arousal_entire_val,arousal_cropped_val,dominance_entire_val,dominance_cropped_val), \
           (x_entire_test, x_cropped_test, valence_entire_test,valence_cropped_test,arousal_entire_test,arousal_cropped_test,dominance_entire_test,dominance_cropped_test)



def load_data_from_numpy_single_output(main_numpy_dir,
                                       verbose=1):

    print ('[INFO] Loading data from numpy arrays...')

    x_image_train = np.load(main_numpy_dir + 'X_train/x_image_train.npy')
    x_body_train = np.load(main_numpy_dir + 'X_train/x_body_train.npy')

    y_image_train = np.load(main_numpy_dir + 'Y_train/y_train.npy')
    y_body_train = np.load(main_numpy_dir + 'Y_train/y_train.npy')


    x_image_val = np.load(main_numpy_dir + 'X_train/x_image_val.npy')
    x_body_val = np.load(main_numpy_dir + 'X_train/x_body_val.npy')

    y_image_val = np.load(main_numpy_dir + 'Y_train/y_val.npy')
    y_body_val = np.load(main_numpy_dir + 'Y_train/y_val.npy')


    x_image_test = np.load(main_numpy_dir + 'X_train/x_image_test.npy')
    x_body_test = np.load(main_numpy_dir + 'X_train/x_body_test.npy')

    y_image_test = np.load(main_numpy_dir + 'Y_train/y_test.npy')
    y_body_test = np.load(main_numpy_dir + 'Y_train/y_test.npy')



    print('[INFO] Data have been successfully loaded')
    print('---------------------------------------------------------------------------------------------------')
    if verbose == 1:
        print('x_image_train shape:', x_image_train.shape)
        print('x_body_train shape:', x_body_train.shape)
        print('y_image_train shape:', y_image_train.shape)
        print('y_body_train shape:', y_body_train.shape)


        print ('---------------------------------------------------------------------------------------------------')

        print('x_image_val shape:', x_image_val.shape)
        print('x_body_val shape:', x_body_val.shape)
        print('y_image_val shape:', y_image_val.shape)
        print('y_body_val shape:', y_body_val.shape)


        print ('---------------------------------------------------------------------------------------------------')

        print('x_image_test shape:', x_image_test.shape)
        print('x_body_test shape:', x_body_test.shape)
        print('y_image_test shape:', y_image_test.shape)
        print('y_body_test shape:', y_body_test.shape)


        print ('---------------------------------------------------------------------------------------------------')

    return (x_image_train, x_body_train,y_image_train,y_body_train), \
           (x_image_val, x_body_val, y_image_val,y_body_val), \
           (x_image_test, x_body_test,y_image_test,y_body_test)







