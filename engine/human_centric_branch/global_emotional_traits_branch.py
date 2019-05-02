# -*- coding: utf-8 -*-
'''
Use three emotional dimensions - valence, arousal and dominance - to describe human perceptions of physical environments.

Interpretations of pleasure: Positive versus negative affective states (e.g. excitement, relaxation, love, and
tranquility versus cruelty, humiliation, disinterest, and boredom)

Interpretations of arousal: Level of mental alertness and physical activity. (e.g. sleep, inactivity, boredom, and
relaxation at the lower end versus wakefulness, bodily tension, strenuous
exercise, and concentration at the higher end).

Interpretations of dominance: Ranges from feelings of total lack control or influence on events and surroundings to
the opposite extreme of feeling influential and in control

'''
from __future__ import print_function
import os
import warnings

from engine.object_detection_branch.retina_net.single_img_inference import RetinaNet_single_img_detection
from engine.object_detection_branch.ssd_detector import single_shot_detector

from applications.emotic_utils import _obtain_single_model_VAD,prepare_input_data, _obtain_nb_classifiers, _obtain_ensembling_weights,\
    _obtain_two_models_ensembling_VAD,_obtain_three_models_ensembling_VAD

from scipy.misc import imread
from matplotlib import pyplot as plt
from utils.generic_utils import crop, round_number




def single_img_VAD_inference(img_path,
                             object_detector_backend,
                             model_a_backend_name,
                             model_b_backend_name = None,
                             model_c_backend_name = None):
    """Performs single image inference.
        It also saves the original image (`img_path`) with the overlaid recognised humans bounding boxes and their VAD values.

    # Arguments
        img_path: Path to image file
        object_detector_backend: Backend with which the objects will be detected. One of `SSD` or `RetinaNet`.
            the person who’s feelings are to be estimated.
        model_backend_name: One of `VGG16`, `VGG19` or `ResNet50`.
            Note that EMOTIC model has already combined `model_backend_name` features with `VGG16_Places365` features at training stage,
            but for simplicity reasons only the body backbone CNN name is adjustable.

    # Returns
        Three integer values corresponding to `valence`, `arousal` and `dominance`.

    # Raises
        ImportError: if PIL is not available.
        ValueError: if interpolation method is not supported.
    """

    if not (object_detector_backend in {'SSD', 'RetinaNet'}):
        raise ValueError('The `object_detector_backend_name` argument should be either '
                         '`SSD` for Single-Shot MultiBox Detector or `RetinaNet` for RetinaNet dense detector. ')

    (head, tail) = os.path.split(img_path)
    filename_only = os.path.splitext(tail)[0]

    nb_classifiers, classifiers_names = _obtain_nb_classifiers(model_a_name=model_a_backend_name,
                                                               model_b_name=model_b_backend_name,
                                                               model_c_name=model_c_backend_name)

    save_as = 'results/'+filename_only + '_' + classifiers_names + '.png'

    if nb_classifiers == 1:
        model_a = _obtain_single_model_VAD(model_a_backend_name)

    elif nb_classifiers == 2:
        w_model_a, w_model_b = _obtain_ensembling_weights(nb_classifiers=nb_classifiers,
                                                          model_a_name=model_a_backend_name,
                                                          model_b_name=model_b_backend_name,
                                                          model_c_name=model_c_backend_name)

        model_a, model_b = _obtain_two_models_ensembling_VAD(model_a_name=model_a_backend_name,
                                                             model_b_name=model_b_backend_name)

    elif nb_classifiers == 3:
        w_model_a, w_model_b, w_model_c = _obtain_ensembling_weights(nb_classifiers=nb_classifiers,
                                                                     model_a_name=model_a_backend_name,
                                                                     model_b_name=model_b_backend_name,
                                                                     model_c_name=model_c_backend_name)

        model_a, model_b, model_c = _obtain_three_models_ensembling_VAD(model_a_name=model_a_backend_name,
                                                                        model_b_name=model_b_backend_name,
                                                                        model_c_name=model_c_backend_name)


    numpy_img_path = imread(img_path)

    # ~Object detection branch~
    if object_detector_backend == 'SSD':
        coordinates, persons = single_shot_detector(img_path=img_path, imshow=False)

    elif object_detector_backend == 'RetinaNet':
        coordinates, persons = RetinaNet_single_img_detection(img_path=img_path, imshow=False)

    # configure colours for bounding box and text
    bounding_box_colour_rgbvar = (53, 42, 146)
    bounding_box_colour_rgbvar2 = [x / 255.0 for x in bounding_box_colour_rgbvar]

    text_colour_rgbvar = (214, 86, 100)
    text_colour_rgbvar2 = [x / 255.0 for x in text_colour_rgbvar]

    if persons != 0:
        print('--IMAGE INFERENCE FOR |%d| PERSON(S) FOUND:' % persons)

        plt.figure(figsize=(10, 12))
        plt.imshow(numpy_img_path)

        current_axis = plt.gca()

        counter = 1
        valence_sum = 0
        arousal_sum = 0
        dominance_sum = 0

        for box in coordinates:

            # checks if the number of persons have been reached in order to stop the for loop.
            # if counter > persons:
            #     break

            if box[0] != 0:
                print('[INFO] Person #%d' % counter)

                crop(image_path=img_path, coords=box, saved_location='body_img.jpg')

                x1, x2 = prepare_input_data(body_path = 'body_img.jpg',
                                            image_path = img_path)

                if nb_classifiers == 1:

                    preds = model_a.predict([x1, x2])


                elif nb_classifiers == 2:
                    # obtain predictions
                    preds_model_a = model_a.predict([x1, x2])
                    preds_model_b = model_b.predict([x1, x2])

                    if w_model_a is None and w_model_b is None:
                        # This new prediction array should be more accurate than any of the initial ones
                        preds = 0.50 * (preds_model_a + preds_model_b)

                    else:
                        preds = w_model_a * preds_model_a + w_model_b * preds_model_b

                elif nb_classifiers == 3:
                    # obtain predictions
                    preds_model_a = model_a.predict([x1, x2])
                    preds_model_b = model_b.predict([x1, x2])
                    preds_model_c = model_c.predict([x1, x2])

                    if w_model_a is None and w_model_b is None and w_model_c is None:
                        # This new prediction array should be more accurate than any of the initial ones
                        preds = 0.33 * (preds_model_a + preds_model_b + preds_model_c)

                    else:
                        preds = w_model_a * preds_model_a + w_model_b * preds_model_b + w_model_c * preds_model_c

                # Uncomment to round predicted values
                # valence = round_number(preds[0][0])
                # arousal = round_number(preds[0][1])
                # dominance = round_number(preds[0][2])

                valence = preds[0][0]
                arousal = preds[0][1]
                dominance = preds[0][2]

                print('  Valence (V) --  how pleasant the emotions are: ', valence)
                print('  Arousal (A) --  unrest level of the person(s): ', arousal)
                print('Dominance (D) -- control level of the situation: ', dominance)

                valence_sum += valence
                arousal_sum += arousal
                dominance_sum += dominance

                # current_axis.add_patch(
                #     plt.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1],
                #                   color=text_colour_rgbvar2,
                #                   fill=False,
                #                   linewidth=3.5))




                counter += 1

        global_valence = valence_sum/persons
        global_arousal = arousal_sum/persons
        global_dominance = dominance_sum/persons

        print ('\n')
        print('--GLOBAL EMOTIONAL TRAITS:')

        print("  Valence (V) --  how pleasant the emotions are: %.2f"  % global_valence)
        print("  Arousal (A) --  unrest level of the person(s): %.2f"  % global_arousal)
        print("Dominance (D) --  control level of the situation: %.2f" % global_dominance)
        # print('  Valence (V) --  how pleasant the emotions are: ', global_valence)
        # print('  Arousal (A) --  unrest level of the person(s): ', global_arousal)
        # print('Dominance (D) -- control level of the situation: ', global_dominance)
        #
        # overlayed_text = 'Global emotional traits:' + '\n' '(V): ' + str(round(global_valence,2)) + '\n' '(A): ' + str(round(global_arousal,2)) + '\n' '(D): ' + \
        #                  str(round(global_dominance,2))

        overlayed_text = 'DOMINANCE: ' + \
                         str(round(global_dominance,2))

        current_axis.text(5, -10, overlayed_text, size='x-large', color='white',
                          bbox={'facecolor': bounding_box_colour_rgbvar2, 'alpha': 1.0})




        plt.axis('off')
        plt.savefig(save_as)
        plt.show()
        os.remove("body_img.jpg")

    else:
        warnings.warn('No global emotional traits were identified: '
                      'there was no person detected in the image.')

        global_valence = 0
        global_arousal = 0
        global_dominance = 0



    return global_valence, global_arousal, global_dominance





def single_img_VAD_inference_return_only(img_path,
                                         object_detector_backend,
                                         model_a_backend_name,
                                         model_b_backend_name=None,
                                         model_c_backend_name=None):
    """Performs single image inference.

    # Arguments
        img_path: Path to image file
        object_detector_backend: Backend with which the objects will be detected. One of `SSD` or `RetinaNet`.
            the person who’s feelings are to be estimated.
        model_backend_name: One of `VGG16`, `VGG19` or `ResNet50`.
            Note that EMOTIC model has already combined `model_backend_name` features with `VGG16_Places365` features at training stage,
            but for simplicity reasons only the body backbone CNN name is adjustable.

    # Returns
        Three integer values corresponding to `valence`, `arousal` and `dominance`.

    # Raises
        ImportError: if PIL is not available.
        ValueError: if interpolation method is not supported.
    """

    if not (object_detector_backend in {'SSD', 'RetinaNet'}):
        raise ValueError('The `object_detector_backend_name` argument should be either '
                         '`SSD` for Single-Shot MultiBox Detector or `RetinaNet` for RetinaNet dense detector. ')


    nb_classifiers, classifiers_names = _obtain_nb_classifiers(model_a_name=model_a_backend_name,
                                                               model_b_name=model_b_backend_name,
                                                               model_c_name=model_c_backend_name)


    if nb_classifiers == 1:
        model_a = _obtain_single_model_VAD(model_a_backend_name)

    elif nb_classifiers == 2:
        w_model_a, w_model_b = _obtain_ensembling_weights(nb_classifiers=nb_classifiers,
                                                          model_a_name=model_a_backend_name,
                                                          model_b_name=model_b_backend_name,
                                                          model_c_name=model_c_backend_name)

        model_a, model_b = _obtain_two_models_ensembling_VAD(model_a_name=model_a_backend_name,
                                                             model_b_name=model_b_backend_name)

    elif nb_classifiers == 3:
        w_model_a, w_model_b, w_model_c = _obtain_ensembling_weights(nb_classifiers=nb_classifiers,
                                                                     model_a_name=model_a_backend_name,
                                                                     model_b_name=model_b_backend_name,
                                                                     model_c_name=model_c_backend_name)

        model_a, model_b, model_c = _obtain_three_models_ensembling_VAD(model_a_name=model_a_backend_name,
                                                                        model_b_name=model_b_backend_name,
                                                                        model_c_name=model_c_backend_name)
    # Uncomment for extra verbosity
    # print('[INFO] EMOTIC VAD models have been loaded')

    # numpy_img_path = imread(img_path)

    # ~Object detection branch~
    if object_detector_backend == 'SSD':
        coordinates, persons = single_shot_detector(img_path=img_path, imshow=False)

    elif object_detector_backend == 'RetinaNet':
        coordinates, persons = RetinaNet_single_img_detection(img_path=img_path, imshow=False)

    # Uncomment for extra verbosity
    # print('[INFO] Objects in image have been detected')


    if persons != 0:
        # Uncomment for extra verbosity
        # print('[INFO] Carrying out continuous emotion recognition in VAD space for %d person(s) found: ' % persons)

        counter = 1
        dominance_sum = 0
        valence_sum = 0

        for box in coordinates:

            # checks if the number of persons have been reached in order to stop the for loop.
            # if counter > persons:
            #     break

            if box[0] != 0:
                # Uncomment for extra verbosity
                # print('[INFO] Person #%d' % counter)

                crop(image_path=img_path, coords=box, saved_location='body_img.jpg')

                x1, x2 = prepare_input_data(body_path = 'body_img.jpg',
                                            image_path = img_path)

                if nb_classifiers == 1:

                    preds = model_a.predict([x1, x2])


                elif nb_classifiers == 2:
                    # obtain predictions
                    preds_model_a = model_a.predict([x1, x2])
                    preds_model_b = model_b.predict([x1, x2])

                    if w_model_a is None and w_model_b is None:
                        # This new prediction array should be more accurate than any of the initial ones
                        preds = 0.50 * (preds_model_a + preds_model_b)

                    else:
                        preds = w_model_a * preds_model_a + w_model_b * preds_model_b

                elif nb_classifiers == 3:
                    # obtain predictions
                    preds_model_a = model_a.predict([x1, x2])
                    preds_model_b = model_b.predict([x1, x2])
                    preds_model_c = model_c.predict([x1, x2])

                    if w_model_a is None and w_model_b is None and w_model_c is None:
                        # This new prediction array should be more accurate than any of the initial ones
                        preds = 0.33 * (preds_model_a + preds_model_b + preds_model_c)

                    else:
                        preds = w_model_a * preds_model_a + w_model_b * preds_model_b + w_model_c * preds_model_c

                # Uncomment to round predicted values
                # valence = round_number(preds[0][0])
                # arousal = round_number(preds[0][1])
                # dominance = round_number(preds[0][2])

                valence = preds[0][0]
                # arousal = preds[0][1]
                dominance = preds[0][2]

                # Uncomment for extra verbosity
                # print('  Valence (V): ', valence)
                # print('  Arousal (A): ', arousal)
                # print('Dominance (D): ', dominance)

                valence_sum += valence
                # arousal_sum += arousal
                dominance_sum += dominance



                counter += 1

        global_valence = valence_sum/persons
        # global_arousal = arousal_sum/persons
        global_dominance = dominance_sum/persons

        # Uncomment for extra verbosity
        # print ('\n')
        # print('[INFO] Global emotional traits::')
        # print('  Valence (V) --  how pleasant the emotions are: ', global_valence)
        # print('  Arousal (A) --  unrest level of the person(s): ', global_arousal)
        # print('Dominance (D) -- control level of the situation: ', global_dominance)
        # print('\n')

        os.remove("body_img.jpg")

    else:
        print("[WARNING] No global emotional traits were identified -- no `people` found in input image `", img_path, '`')

        global_valence = 0
        # global_arousal = 0
        global_dominance = 0



    return global_valence, global_dominance


def single_img_VAD_inference_with_bounding_boxes(img_path,
                                                 object_detector_backend,
                                                 model_a_backend_name,
                                                 model_b_backend_name=None,
                                                 model_c_backend_name=None):
    """Performs single image inference.
        It also saves the original image (`img_path`) with the overlaid recognised humans bounding boxes and their VAD values.

    # Arguments
        img_path: Path to image file
        object_detector_backend: Backend with which the objects will be detected. One of `SSD` or `RetinaNet`.
            the person who’s feelings are to be estimated.
        model_backend_name: One of `VGG16`, `VGG19` or `ResNet50`.
            Note that EMOTIC model has already combined `model_backend_name` features with `VGG16_Places365` features at training stage,
            but for simplicity reasons only the body backbone CNN name is adjustable.

    # Returns
        Three integer values corresponding to `valence`, `arousal` and `dominance`.

    # Raises
        ImportError: if PIL is not available.
        ValueError: if interpolation method is not supported.
    """

    if not (object_detector_backend in {'SSD', 'RetinaNet'}):
        raise ValueError('The `object_detector_backend_name` argument should be either '
                         '`SSD` for Single-Shot MultiBox Detector or `RetinaNet` for RetinaNet dense detector. ')

    (head, tail) = os.path.split(img_path)
    filename_only = os.path.splitext(tail)[0]

    nb_classifiers, classifiers_names = _obtain_nb_classifiers(model_a_name=model_a_backend_name,
                                                               model_b_name=model_b_backend_name,
                                                               model_c_name=model_c_backend_name)

    save_as = 'results/'+filename_only + '_' + classifiers_names + '.png'

    if nb_classifiers == 1:
        model_a = _obtain_single_model_VAD(model_a_backend_name)

    elif nb_classifiers == 2:
        w_model_a, w_model_b = _obtain_ensembling_weights(nb_classifiers=nb_classifiers,
                                                          model_a_name=model_a_backend_name,
                                                          model_b_name=model_b_backend_name,
                                                          model_c_name=model_c_backend_name)

        model_a, model_b = _obtain_two_models_ensembling_VAD(model_a_name=model_a_backend_name,
                                                             model_b_name=model_b_backend_name)

    elif nb_classifiers == 3:
        w_model_a, w_model_b, w_model_c = _obtain_ensembling_weights(nb_classifiers=nb_classifiers,
                                                                     model_a_name=model_a_backend_name,
                                                                     model_b_name=model_b_backend_name,
                                                                     model_c_name=model_c_backend_name)

        model_a, model_b, model_c = _obtain_three_models_ensembling_VAD(model_a_name=model_a_backend_name,
                                                                        model_b_name=model_b_backend_name,
                                                                        model_c_name=model_c_backend_name)


    numpy_img_path = imread(img_path)

    # ~Object detection branch~
    if object_detector_backend == 'SSD':
        coordinates, persons = single_shot_detector(img_path=img_path, imshow=False)

    elif object_detector_backend == 'RetinaNet':
        coordinates, persons = RetinaNet_single_img_detection(img_path=img_path, imshow=False)

    # configure colours for bounding box and text
    bounding_box_colour_rgbvar = (53, 42, 146)
    bounding_box_colour_rgbvar2 = [x / 255.0 for x in bounding_box_colour_rgbvar]

    text_colour_rgbvar = (214, 86, 100)
    text_colour_rgbvar2 = [x / 255.0 for x in text_colour_rgbvar]

    if persons != 0:
        print('--IMAGE INFERENCE FOR |%d| PERSON(S) FOUND:' % persons)

        plt.figure(figsize=(10, 12))
        plt.imshow(numpy_img_path)

        current_axis = plt.gca()

        counter = 1
        valence_sum = 0
        arousal_sum = 0
        dominance_sum = 0

        for box in coordinates:

            # checks if the number of persons have been reached in order to stop the for loop.
            # if counter > persons:
            #     break

            if box[0] != 0:
                print('[INFO] Person #%d' % counter)

                crop(image_path=img_path, coords=box, saved_location='body_img.jpg')

                x1, x2 = prepare_input_data(body_path = 'body_img.jpg',
                                            image_path = img_path)

                if nb_classifiers == 1:

                    preds = model_a.predict([x1, x2])


                elif nb_classifiers == 2:
                    # obtain predictions
                    preds_model_a = model_a.predict([x1, x2])
                    preds_model_b = model_b.predict([x1, x2])

                    if w_model_a is None and w_model_b is None:
                        # This new prediction array should be more accurate than any of the initial ones
                        preds = 0.50 * (preds_model_a + preds_model_b)

                    else:
                        preds = w_model_a * preds_model_a + w_model_b * preds_model_b

                elif nb_classifiers == 3:
                    # obtain predictions
                    preds_model_a = model_a.predict([x1, x2])
                    preds_model_b = model_b.predict([x1, x2])
                    preds_model_c = model_c.predict([x1, x2])

                    if w_model_a is None and w_model_b is None and w_model_c is None:
                        # This new prediction array should be more accurate than any of the initial ones
                        preds = 0.33 * (preds_model_a + preds_model_b + preds_model_c)

                    else:
                        preds = w_model_a * preds_model_a + w_model_b * preds_model_b + w_model_c * preds_model_c

                # Uncomment to round predicted values
                # valence = round_number(preds[0][0])
                # arousal = round_number(preds[0][1])
                # dominance = round_number(preds[0][2])

                valence = preds[0][0]
                arousal = preds[0][1]
                dominance = preds[0][2]

                print('  Valence (V) --  how pleasant the emotions are: ', valence)
                print('  Arousal (A) --  unrest level of the person(s): ', arousal)
                print('Dominance (D) -- control level of the situation: ', dominance)

                valence_sum += valence
                arousal_sum += arousal
                dominance_sum += dominance

                current_axis.add_patch(
                    plt.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1],
                                  color=text_colour_rgbvar2,
                                  fill=False,
                                  linewidth=3.5))

                people_VAD_overlayed_text = '(V): ' + str(round(valence, 2)) + '\n' '(A): ' \
                                            + str(round(arousal, 2)) + '\n' '(D): ' \
                                            + str(round(dominance, 2))

                current_axis.text(box[0]+5, box[1]-10, people_VAD_overlayed_text, size='x-large', color='white',
                                  bbox={'facecolor': bounding_box_colour_rgbvar2, 'alpha': 1.0})




                counter += 1

        global_valence = valence_sum/persons
        global_arousal = arousal_sum/persons
        global_dominance = dominance_sum/persons

        print ('\n')
        print('--GLOBAL EMOTIONAL TRAITS:')

        print("  Valence (V) --  how pleasant the emotions are: %.2f"  % global_valence)
        print("  Arousal (A) --  unrest level of the person(s): %.2f"  % global_arousal)
        print("Dominance (D) --  control level of the situation: %.2f" % global_dominance)
        # print('  Valence (V) --  how pleasant the emotions are: ', global_valence)
        # print('  Arousal (A) --  unrest level of the person(s): ', global_arousal)
        # print('Dominance (D) -- control level of the situation: ', global_dominance)

        overlayed_text = '(V): ' + str(round(global_valence,2)) + '\n' '(A): ' + str(round(global_arousal,2)) + '\n' '(D): ' + \
                         str(round(global_dominance,2))


        # current_axis.text(0, 0, overlayed_text, size='x-large', color='white',
        #                   bbox={'facecolor': bounding_box_colour_rgbvar2, 'alpha': 1.0})

        plt.axis('off')
        plt.savefig(save_as)
        plt.show()
        os.remove("body_img.jpg")

    else:
        warnings.warn('No global emotional traits were identified: '
                      'there was no person detected in the image.')

        global_valence = 0
        global_arousal = 0
        global_dominance = 0



    return global_valence, global_arousal, global_dominance





if __name__ == "__main__":

    img_path = '/home/sandbox/Desktop/Two-class-HRV/ChildLabour/test/no_child_labour/no_child_labour_0015.jpg'
    model_a_backend_name = 'VGG19'
    model_b_backend_name = 'VGG16'
    model_c_backend_name = 'ResNet50'

    valence, arousal, dominance = single_img_VAD_inference(img_path = img_path,
                                                           object_detector_backend='RetinaNet',
                                                           model_a_backend_name = model_a_backend_name,
                                                           model_b_backend_name=model_b_backend_name,
                                                           model_c_backend_name=model_c_backend_name,
                                                           )