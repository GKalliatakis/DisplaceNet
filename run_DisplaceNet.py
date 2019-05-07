# -*- coding: utf-8 -*-
'''

'''
from __future__ import print_function
import argparse
from inference.displacenet_single_image_inference_unified import displaceNet_inference


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_path", type = str,help = 'path of the input image')
    parser.add_argument("--hra_model_backend_name", type = str,help = 'One of `VGG16`, `VGG19`, `ResNet50` or `VGG16_Places365`')
    parser.add_argument("--emotic_model_backend_name", type=str, help='One of `VGG16`, `VGG19` or `ResNet50`')
    parser.add_argument("--nb_of_conv_layers_to_fine_tune", type=int, help='indicates the number of convolutional layers that were fine-tuned during training')

    args = parser.parse_args()
    return args

args = get_args()


DisplaceNet_preds = displaceNet_inference(img_path=args.img_path,
                                          emotic_model_a_backend_name=args.emotic_model_backend_name,
                                          emotic_model_b_backend_name=None,
                                          emotic_model_c_backend_name=None,
                                          hra_model_backend_name=args.hra_model_backend_name,
                                          nb_of_fine_tuned_conv_layers=args.nb_of_conv_layers_to_fine_tune,
                                          violation_class='dp')

print (DisplaceNet_preds)
print (DisplaceNet_preds[0])