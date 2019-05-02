# -*- coding: utf-8 -*-
""" Training script for continuous emotion recognition in VAD space.

# Reference
- [Emotion Recognition in Context](http://sunai.uoc.edu/emotic/pdf/EMOTIC_cvpr2017.pdf)
- https://stackoverflow.com/questions/43452441/keras-all-layer-names-should-be-unique

"""

from __future__ import print_function
import argparse
from engine.human_centric_branch.emotic_vad_model import EMOTIC_VAD
from applications.emotic_utils import _obtain_weights_CSVLogger_filenames as regression_obtain_weights_CSVLogger_filenames




def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--body_backbone_CNN", type = str,help = 'One of `VGG16`, `VGG19`, `ResNet50`, `VGG16_Places365`')
    parser.add_argument("--image_backbone_CNN", type = str,help = 'One of `VGG16`, `VGG19`, `ResNet50`, `VGG16_Places365`')
    parser.add_argument("--modelCheckpoint_quantity", type=str, help='Quantity to monitor when `ModelCheckpoint` is enabled')
    parser.add_argument("--earlyStopping_quantity", type=str, help='Quantity to monitor when `EarlyStopping` is enabled')
    parser.add_argument("--nb_of_epochs", type=int, help="Total number of iterations on the data")

    args = parser.parse_args()
    return args



args = get_args()

hdf5_file = '/home/gkallia/git/emotic-VAD-classification/dataset/EMOTIC-VAD-regression.hdf5'


modelCheckpoint_quantity = args.modelCheckpoint_quantity
earlyStopping_quantity = args.earlyStopping_quantity

weights_filename, CSVLogger_filename = regression_obtain_weights_CSVLogger_filenames(body_backbone_CNN=args.body_backbone_CNN,
                                                                                     image_backbone_CNN=args.image_backbone_CNN)

emotic_model = EMOTIC_VAD(hdf5_file=hdf5_file,
                          body_backbone_CNN=args.body_backbone_CNN,
                          image_backbone_CNN=args.image_backbone_CNN,
                          nb_of_epochs=args.nb_of_epochs,
                          weights_to_file=weights_filename,
                          modelCheckpoint_quantity=modelCheckpoint_quantity,
                          earlyStopping_quantity=earlyStopping_quantity,
                          CSVLogger_filename=CSVLogger_filename)




emotic_model.train()