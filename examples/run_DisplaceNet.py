# -*- coding: utf-8 -*-
'''

'''
from __future__ import print_function
import os
from inference.displacenet_single_image_inference_unified import displaceNet_inference


img_path = '/home/sandbox/Desktop/HRA-2clas-full-test/DisplacedPopulations/displaced_populations/displaced_populations_0000.jpg'
violation_class = 'dp'
hra_model_backend_name = 'VGG16'
nb_of_conv_layers_to_fine_tune = 1

emotic_model_a_backend_name = 'VGG16'
emotic_model_b_backend_name = None
emotic_model_c_backend_name = None


DisplaceNet_preds = displaceNet_inference(img_path=img_path,
                                          emotic_model_a_backend_name=emotic_model_a_backend_name,
                                          emotic_model_b_backend_name=emotic_model_b_backend_name,
                                          emotic_model_c_backend_name=emotic_model_c_backend_name,
                                          hra_model_backend_name=hra_model_backend_name,
                                          nb_of_fine_tuned_conv_layers=nb_of_conv_layers_to_fine_tune,
                                          violation_class=violation_class)

print (DisplaceNet_preds)
print (DisplaceNet_preds[0])