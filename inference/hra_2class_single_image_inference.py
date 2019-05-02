# -*- coding: utf-8 -*-
'''


'''

from engine.displaced_people_branch.single_image_inference_hra_2class import single_img_HRA_inference
from applications.hra_utils import plot_preds
from keras.preprocessing import image
from matplotlib import pyplot as plt
from scipy.misc import imread


img_path = '/home/sandbox/Desktop/Testing Images/camping.jpg'
violation_class = 'dp'
model_backend_name = 'VGG16'
nb_of_conv_layers_to_fine_tune = 1

raw_preds, overlayed_text = single_img_HRA_inference(img_path=img_path,
                                     violation_class=violation_class,
                                     model_backend_name=model_backend_name,
                                     nb_of_conv_layers_to_fine_tune=nb_of_conv_layers_to_fine_tune)


img = image.load_img(img_path, target_size=(224, 224))
print (raw_preds[0])

# plot_preds(violation_class, img, raw_preds[0])


numpy_img_path = imread(img_path)
plt.figure(figsize=(10, 12))
plt.imshow(numpy_img_path)

current_axis = plt.gca()

# configure colours for bounding box and text
bounding_box_colour_rgbvar = (53, 42, 146)
bounding_box_colour_rgbvar2 = [x / 255.0 for x in bounding_box_colour_rgbvar]

text_colour_rgbvar = (214, 86, 100)
text_colour_rgbvar2 = [x / 255.0 for x in text_colour_rgbvar]

current_axis.text(0, 0, overlayed_text, size='x-large', color='white',
                  bbox={'facecolor': bounding_box_colour_rgbvar2, 'alpha': 1.0})

plt.axis('off')
plt.show()