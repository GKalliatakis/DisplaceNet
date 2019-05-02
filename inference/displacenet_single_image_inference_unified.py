# -*- coding: utf-8 -*-
'''

'''

from __future__ import print_function
from engine.human_centric_branch.global_emotional_traits_branch import single_img_VAD_inference_return_only
from engine.displaced_people_branch.single_image_inference_hra_2class import single_img_HRA_inference_return_only
from utils.inference_utils import _obtain_emotional_traits_calibrated_predictions
from applications.hra_utils import decode_predictions

from keras.preprocessing import image
from scipy.misc import imread
from matplotlib import pyplot as plt

def displaceNet_inference(img_path,
                          emotic_model_a_backend_name,
                          emotic_model_b_backend_name,
                          emotic_model_c_backend_name,
                          hra_model_backend_name,
                          nb_of_fine_tuned_conv_layers,
                          violation_class):






    # obtain global emotional traits as VAD values
    global_valence, global_dominance = single_img_VAD_inference_return_only(img_path=img_path,
                                                                            object_detector_backend='RetinaNet',
                                                                            model_a_backend_name=emotic_model_a_backend_name,
                                                                            model_b_backend_name=emotic_model_b_backend_name,
                                                                            model_c_backend_name=emotic_model_c_backend_name,
                                                                            )

    raw_HRA_preds = single_img_HRA_inference_return_only(img_path=img_path,
                                                         violation_class=violation_class,
                                                         model_backend_name=hra_model_backend_name,
                                                         nb_of_conv_layers_to_fine_tune=nb_of_fine_tuned_conv_layers)

    # plain_preds = decode_predictions(violation_class=violation_class,
    #                                  preds=raw_HRA_preds,
    #                                  top=2)

    # # Uncomment for extra verbosity
    # plain_predicted_probability = plain_preds[0][0][2]
    #
    # plain_predicted_label = plain_preds[0][0][1]
    #
    # print ('\n')
    #
    # print('[INFO] Plain predictions: ',plain_predicted_label, '->', plain_predicted_probability)
    #
    # print ('Global dominance: ', global_dominance)

    # Case where no people were detected
    if global_dominance == 0:
        # print ('No people were detected!')


        final_preds = decode_predictions(violation_class=violation_class,
                                         preds=raw_HRA_preds,
                                         top=2)



    else:

        calibrated_preds = _obtain_emotional_traits_calibrated_predictions(emotional_trait=global_dominance,
                                                                           raw_preds=raw_HRA_preds)


        final_preds = decode_predictions(violation_class=violation_class,
                                         preds=calibrated_preds,
                                         top=2)




        # # Uncomment for extra verbosity
        # calibrated_predicted_probability = final_preds[0][0][2]
        #
        # calibrated_predicted_label = final_preds[0][0][1]
        #
        # print('[INFO] Calibrated predictions: ', calibrated_predicted_label, '->', calibrated_predicted_probability)


    return final_preds



if __name__ == "__main__":


    img_path = '/home/sandbox/Desktop/Testing Images/human_right_viol_1.jpg'
    violation_class = 'cl'
    hra_model_backend_name = 'VGG16'
    nb_of_fine_tuned_conv_layers = 1

    emotic_model_a_backend_name = 'VGG16'
    emotic_model_b_backend_name = None
    emotic_model_c_backend_name = None


    final_preds = displaceNet_inference(img_path,
                                        emotic_model_a_backend_name,
                                        emotic_model_b_backend_name,
                                        emotic_model_c_backend_name,
                                        hra_model_backend_name,
                                        nb_of_fine_tuned_conv_layers,
                                        violation_class)

    print (final_preds)

    img = image.load_img(img_path, target_size=(224, 224))

    # plot_preds(violation_class, img, raw_preds[0])


    numpy_img_path = imread(img_path)
    plt.figure(figsize=(10, 12))
    plt.imshow(numpy_img_path)

    current_axis = plt.gca()

    # configure colours for bounding box and text
    violation_bounding_box_colour_rgbvar = (255, 3, 62)
    violation_bounding_box_colour_rgbvar2 = [x / 255.0 for x in violation_bounding_box_colour_rgbvar]

    no_violation_bounding_box_colour_rgbvar = (34, 139, 34)
    no_violation_bounding_box_colour_rgbvar2 = [x / 255.0 for x in no_violation_bounding_box_colour_rgbvar]


    overlayed_text = str(final_preds[0][0][1]) + ' (' + str(round(final_preds[0][0][2], 2)) + ')'

    if violation_class == 'dp':
        if final_preds[0][0][1] == 'displaced_populations':
            current_axis.text(0, 0, overlayed_text, size='x-large', color='white',
                          bbox={'facecolor': violation_bounding_box_colour_rgbvar2, 'alpha': 1.0})

        elif final_preds[0][0][1] == 'no_displaced_populations':
            current_axis.text(0, 0, overlayed_text, size='x-large', color='white',
                              bbox={'facecolor': no_violation_bounding_box_colour_rgbvar2, 'alpha': 1.0})

    else:
        if final_preds[0][0][1] == 'child_labour':
            current_axis.text(0, 0, overlayed_text, size='x-large', color='white',
                              bbox={'facecolor': violation_bounding_box_colour_rgbvar2, 'alpha': 1.0})

        elif final_preds[0][0][1] == 'no_child_labour':
            current_axis.text(0, 0, overlayed_text, size='x-large', color='white',
                              bbox={'facecolor': no_violation_bounding_box_colour_rgbvar2, 'alpha': 1.0})






    plt.axis('off')
    plt.show()


    # img_path = '/home/sandbox/Desktop/Testing Images/camping.jpg'
    # violation_class = 'dp'
    # hra_model_backend_name = 'VGG16'
    # nb_of_fine_tuned_conv_layers = 1
    #
    # emotic_model_a_backend_name = 'VGG16'
    # emotic_model_b_backend_name = 'VGG19'
    # emotic_model_c_backend_name = 'ResNet50'
    #
    #
    # final_preds = displaceNet_inference(img_path,
    #                    emotic_model_a_backend_name,
    #                    emotic_model_b_backend_name,
    #                    emotic_model_c_backend_name,
    #                    hra_model_backend_name,
    #                    nb_of_fine_tuned_conv_layers,
    #                    violation_class)
    #
    # print (final_preds)