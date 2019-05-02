# -*- coding: utf-8 -*-
"""Python utilities required by `AbuseNet`. """

from PIL import Image
from keras.callbacks import LearningRateScheduler
from keras import backend as K
import numpy as np
import matplotlib.pyplot as plt
import itertools
import sys
import tensorflow as tf
import keras.backend.tensorflow_backend as tfb


def crop(image_path, coords, saved_location):
    """ Crops an image.
    # Reference
    - https://www.blog.pythonlibrary.org/2017/10/03/how-to-crop-a-photo-with-python/

    # Arguments
        image_path: The path to the image to edit.
        coords: A tuple of x/y coordinates (x1, y1, x2, y2).
        saved_location: Path to save the cropped image.
    """
    image_obj = Image.open(image_path)
    cropped_image = image_obj.crop(coords)
    cropped_image.save(saved_location)

def crop_no_save(image_path, coords):
    """ Crops an image.
    # Reference
    - https://www.blog.pythonlibrary.org/2017/10/03/how-to-crop-a-photo-with-python/

    # Arguments
        image_path: The path to the image to edit.
        coords: A tuple of x/y coordinates (x1, y1, x2, y2).
        saved_location: Path to save the cropped image.
    """
    image_obj = Image.open(image_path)
    cropped_image = image_obj.crop(coords)

    return cropped_image



def progress(count, total):
    """ Command line progress bar.
    # Reference
    - https://gist.github.com/vladignatyev/06860ec2040cb497f0f3

    """
    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))

    # percents = round(100.0 * count / float(total), 1)

    tmp = str(count)+'/'+str(total)
    bar = '=' * filled_len + '.' * (bar_len - filled_len)

    print ('%s [%s]\r' % (tmp, bar))
    # sys.stdout.write('[%s] %s%s ...%s\r' % (bar, percents, '%', status))
    # sys.stdout.flush()


def hms_string(sec_elapsed):
    """ Formats the nb of seconds returned for a process.
    """
    h = int(sec_elapsed / (60 * 60))
    m = int((sec_elapsed % (60 * 60)) / 60)
    s = sec_elapsed % 60.
    return "{}:{:>02}:{:>05.2f}".format(h, m, s)



def preprocess_input(x, v2=True):
    x = x.astype('float32')
    x = x / 255.0
    if v2:
        x = x - 0.5
        x = x * 2.0
    return x



def step_decay_schedule(initial_lr=1e-3, decay_factor=0.75, step_size=10):
    '''Wrapper function to create a LearningRateScheduler with step decay schedule.
    # Reference
    - https://gist.github.com/jeremyjordan/86398d7c05c02396c24661baa4c88165
    '''
    def schedule(epoch):
        return initial_lr * (decay_factor ** np.floor(epoch / step_size))

    return LearningRateScheduler(schedule)



def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# Print iterations progress
# reference https://gist.github.com/aubricus/f91fb55dc6ba5557fbab06119420dd6a
def print_progress(iteration, total, prefix='', suffix='', decimals=1, bar_length=100):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        bar_length  - Optional  : character length of bar (Int)
    """
    str_format = "{0:." + str(decimals) + "f}"
    percents = str_format.format(100 * (iteration / float(total)))
    filled_length = int(round(bar_length * iteration / float(total)))
    bar = '█' * filled_length + '-' * (bar_length - filled_length)

    sys.stdout.write('\r%s |%s| %s%s %s' % (prefix, bar, percents, '%', suffix)),

    if iteration == total:
        sys.stdout.write('\n')
    sys.stdout.flush()


def places_preprocess_input(x, dim_ordering='default'):
    if dim_ordering == 'default':
        dim_ordering = K.image_dim_ordering()
    assert dim_ordering in {'tf', 'th'}

    if dim_ordering == 'th':
        x[:, 0, :, :] -= 104.006
        x[:, 1, :, :] -= 116.669
        x[:, 2, :, :] -= 122.679
        # 'RGB'->'BGR'
        x = x[:, ::-1, :, :]
    else:
        x[:, :, :, 0] -= 104.006
        x[:, :, :, 1] -= 116.669
        x[:, :, :, 2] -= 122.679
        # 'RGB'->'BGR'
        x = x[:, :, :, ::-1]
    return x


def imagenet_preprocess_input(x, dim_ordering='default'):
    if dim_ordering == 'default':
        dim_ordering = K.image_dim_ordering()
    assert dim_ordering in {'tf', 'th'}

    if dim_ordering == 'th':
        x[:, 0, :, :] -= 103.939
        x[:, 1, :, :] -= 116.779
        x[:, 2, :, :] -= 123.68
        # 'RGB'->'BGR'
        x = x[:, ::-1, :, :]
    else:
        x[:, :, :, 0] -= 103.939
        x[:, :, :, 1] -= 116.779
        x[:, :, :, 2] -= 123.68
        # 'RGB'->'BGR'
        x = x[:, :, :, ::-1]
    return x


def round_number(number):

    rounded_number = int(round(number))

    # make sure there are no values above 10 (which is the maximum value stated in the paper)
    if rounded_number > 10:

        rounded_number = 10

    return rounded_number



# -------------------------------------------------------------------------------- #
#                      Additional loss functions & metrics
# -------------------------------------------------------------------------------- #

def euclidean_distance_loss(y_true, y_pred):
    """
    Euclidean distance loss
    https://en.wikipedia.org/wiki/Euclidean_distance
    http://www.riptutorial.com/keras/example/32022/euclidean-distance-loss
    :param y_true: TensorFlow/Theano tensor
    :param y_pred: TensorFlow/Theano tensor of the same shape as y_true
    :return: float
    """
    return K.sqrt(K.sum(K.square(y_pred - y_true), axis=-1))


def rmse(y_true, y_pred):
    """
    Root mean squared error
    https://en.wikipedia.org/wiki/Euclidean_distance
    http://www.riptutorial.com/keras/example/32022/euclidean-distance-loss
    :param y_true: TensorFlow/Theano tensor
    :param y_pred: TensorFlow/Theano tensor of the same shape as y_true
    :return: float
    """
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))




class WeightedEuclideanDistance(object):
    """
    A weighted version of Euclidean distance loss for keras. This lets you apply a weight to unbalanced classes.
    # reference: implementation based on https://github.com/keras-team/keras/issues/2115#issuecomment-315571824

    Usage:
    The constructor expects a dictionary with same structure as `class_weight` param from model.fit
    """

    def __init__(self, weights):
        nb_cl = len(weights)
        self.weights = np.ones((nb_cl, nb_cl))
        for class_idx, class_weight in weights.items():
            self.weights[0][class_idx] = class_weight
            self.weights[class_idx][0] = class_weight
        self.__name__ = 'weighted_euclidean_distance'

    def __call__(self, y_true, y_pred):
        return self.weighted_euclidean_distance(y_true, y_pred)

    def weighted_euclidean_distance(self, y_true, y_pred):
        nb_cl = len(self.weights)
        final_mask = K.zeros_like(y_pred[..., 0])
        y_pred_max = K.max(y_pred, axis=-1)
        y_pred_max = K.expand_dims(y_pred_max, axis=-1)
        y_pred_max_mat = K.equal(y_pred, y_pred_max)
        for c_p, c_t in itertools.product(range(nb_cl), range(nb_cl)):
            w = K.cast(self.weights[c_t, c_p], K.floatx())
            y_p = K.cast(y_pred_max_mat[..., c_p], K.floatx())
            y_t = K.cast(y_pred_max_mat[..., c_t], K.floatx())
            final_mask += w * y_p * y_t
        return K.sqrt(K.sum(K.square(y_pred - y_true), axis=-1)) * final_mask


class WeightedBinaryCrossEntropy(object):
    """
        A weighted version of Euclidean distance loss for keras. This lets you apply a weight to unbalanced classes.
        # reference: implementation based on https://github.com/keras-team/keras/issues/2115#issuecomment-315571824

        Usage:
        The constructor expects a dictionary with same structure as `class_weight` param from model.fit
    """

    def __init__(self, weights):
        nb_cl = len(weights)
        self.weights = np.ones((nb_cl, nb_cl))
        for class_idx, class_weight in weights.items():
            self.weights[0][class_idx] = class_weight
            self.weights[class_idx][0] = class_weight
        self.__name__ = 'weighted_binary_crossentropy'

    def __call__(self, y_true, y_pred):
        return self.weighted_binary_crossentropy(y_true, y_pred)

    def weighted_binary_crossentropy(self, y_true, y_pred):
        nb_cl = len(self.weights)
        final_mask = K.zeros_like(y_pred[..., 0])
        y_pred_max = K.max(y_pred, axis=-1)
        y_pred_max = K.expand_dims(y_pred_max, axis=-1)
        y_pred_max_mat = K.equal(y_pred, y_pred_max)
        for c_p, c_t in itertools.product(range(nb_cl), range(nb_cl)):
            w = K.cast(self.weights[c_t, c_p], K.floatx())
            y_p = K.cast(y_pred_max_mat[..., c_p], K.floatx())
            y_t = K.cast(y_pred_max_mat[..., c_t], K.floatx())
            final_mask += w * y_p * y_t
        return K.mean(K.binary_crossentropy(y_true, y_pred), axis=-1) * final_mask



def weighted_binary_crossentropy2(target, output):
    """
    Weighted binary crossentropy between an output tensor
    and a target tensor. POS_WEIGHT is used as a multiplier
    for the positive targets.

    Combination of the following functions:
    * keras.losses.binary_crossentropy
    * keras.backend.tensorflow_backend.binary_crossentropy
    * tf.nn.weighted_cross_entropy_with_logits

    reference: https://stackoverflow.com/a/47313183/979377
    """
    # transform back to logits

    POS_WEIGHT = 10  # multiplier for positive targets, needs to be tuned

    _epsilon = tfb._to_tensor(tfb.epsilon(), output.dtype.base_dtype)
    output = tf.clip_by_value(output, _epsilon, 1 - _epsilon)
    output = tf.log(output / (1 - output))
    # compute weighted loss
    loss = tf.nn.weighted_cross_entropy_with_logits(targets=target,
                                                    logits=output,
                                                    pos_weight=POS_WEIGHT)
    return tf.reduce_mean(loss, axis=-1)



# reference: https://github.com/yu4u/age-gender-estimation/blob/3c3a0a2681c045264c2c294e548ffb1b84f24b9e/age_estimation/model.py#L8-L12
def vad_mean_absolute_error(y_true, y_pred):
    true_vad = K.sum(y_true * K.arange(1, 10, dtype="float32"), axis=-1)
    pred_vad = K.sum(y_pred * K.arange(1, 10, dtype="float32"), axis=-1)
    mae = K.mean(K.abs(true_vad - pred_vad))
    return mae