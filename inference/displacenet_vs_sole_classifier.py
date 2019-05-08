from __future__ import print_function
from engine.displaced_people_branch.single_image_inference_hra_2class import single_img_HRA_inference


from keras.preprocessing import image
from scipy.misc import imread
from matplotlib import pyplot as plt

from inference.displacenet_single_image_inference_unified import displaceNet_inference


img_path = '/home/sandbox/Desktop/RF234809_Medium_res-LILY-AND-RUEBEN.jpg'
violation_class = 'dp'
hra_model_backend_name = 'VGG16'
nb_of_conv_layers_to_fine_tune = 1

raw_preds, sole_classifier_overlayed_text, top_1_predicted_label = single_img_HRA_inference(img_path=img_path,
                                                                                            violation_class=violation_class,
                                                                                            model_backend_name=hra_model_backend_name,
                                                                                            nb_of_conv_layers_to_fine_tune=nb_of_conv_layers_to_fine_tune)


img = image.load_img(img_path, target_size=(224, 224))
print ('Vanilla CNN prediction: ', raw_preds[0])


emotic_model_a_backend_name = 'VGG16'
emotic_model_b_backend_name = None
emotic_model_c_backend_name = None

final_preds = displaceNet_inference(img_path,
                                    emotic_model_a_backend_name,
                                    emotic_model_b_backend_name,
                                    emotic_model_c_backend_name,
                                    hra_model_backend_name,
                                    nb_of_conv_layers_to_fine_tune,
                                    violation_class)



print('DisplaceNet prediction: ', final_preds)

numpy_img_path = imread(img_path)
plt.figure(figsize=(10, 12))
plt.imshow(numpy_img_path)

current_axis = plt.gca()

# configure colours for bounding box and text
violation_bounding_box_colour_rgbvar = (255, 3, 62)
violation_bounding_box_colour_rgbvar2 = [x / 255.0 for x in violation_bounding_box_colour_rgbvar]

no_violation_bounding_box_colour_rgbvar = (34, 139, 34)
no_violation_bounding_box_colour_rgbvar2 = [x / 255.0 for x in no_violation_bounding_box_colour_rgbvar]

abusenet_overlayed_text = str(final_preds[0][0][1]) + ' (' + str(round(final_preds[0][0][2], 2)) + ')'

# print (abusenet_overlayed_text)

abusenet_overlayed_text = 'DisplaceNet: '+abusenet_overlayed_text
sole_classifier_overlayed_text = 'Vanilla CNN: '+ sole_classifier_overlayed_text


if violation_class == 'dp':
    if final_preds[0][0][1] == 'displaced_populations':
        current_axis.text(0, -28, abusenet_overlayed_text, size='x-large', color='white',
                          bbox={'facecolor': violation_bounding_box_colour_rgbvar2, 'alpha': 1.0})

        if top_1_predicted_label == 'displaced_populations':
            current_axis.text(0, -7, sole_classifier_overlayed_text, size='x-large', color='white',
                              bbox={'facecolor': violation_bounding_box_colour_rgbvar2, 'alpha': 1.0})
        else:
            current_axis.text(0, -7, sole_classifier_overlayed_text, size='x-large', color='white',
                              bbox={'facecolor': no_violation_bounding_box_colour_rgbvar2, 'alpha': 1.0})

    elif final_preds[0][0][1] == 'no_displaced_populations':
        current_axis.text(0, -45, abusenet_overlayed_text, size='x-large', color='white',
                          bbox={'facecolor': no_violation_bounding_box_colour_rgbvar2, 'alpha': 1.0})

        if top_1_predicted_label == 'displaced_populations':
            current_axis.text(0, -7, sole_classifier_overlayed_text, size='x-large', color='white',
                              bbox={'facecolor': violation_bounding_box_colour_rgbvar2, 'alpha': 1.0})
        else:
            current_axis.text(0, -7, sole_classifier_overlayed_text, size='x-large', color='white',
                              bbox={'facecolor': no_violation_bounding_box_colour_rgbvar2, 'alpha': 1.0})

else:
    if final_preds[0][0][1] == 'child_labour':
        current_axis.text(0, -38, abusenet_overlayed_text, size='x-large', color='white',
                          bbox={'facecolor': violation_bounding_box_colour_rgbvar2, 'alpha': 1.0})

        if top_1_predicted_label == 'child_labour':
            current_axis.text(0, -7, sole_classifier_overlayed_text, size='x-large', color='white',
                              bbox={'facecolor': violation_bounding_box_colour_rgbvar2, 'alpha': 1.0})
        else:
            current_axis.text(0, -7, sole_classifier_overlayed_text, size='x-large', color='white',
                              bbox={'facecolor': no_violation_bounding_box_colour_rgbvar2, 'alpha': 1.0})

    elif final_preds[0][0][1] == 'no_child_labour':
        current_axis.text(0, -38, abusenet_overlayed_text, size='x-large', color='white',
                          bbox={'facecolor': no_violation_bounding_box_colour_rgbvar2, 'alpha': 1.0})

        if top_1_predicted_label == 'child_labour':
            current_axis.text(0, -7, sole_classifier_overlayed_text, size='x-large', color='white',
                              bbox={'facecolor': violation_bounding_box_colour_rgbvar2, 'alpha': 1.0})
        else:
            current_axis.text(0, -7, sole_classifier_overlayed_text, size='x-large', color='white',
                              bbox={'facecolor': no_violation_bounding_box_colour_rgbvar2, 'alpha': 1.0})





plt.axis('off')
plt.show()
