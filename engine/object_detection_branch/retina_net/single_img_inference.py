from __future__ import print_function

import matplotlib.pyplot as plt
# import miscellaneous modules
import numpy as np

import cv2
from engine.object_detection_branch.retina_net.keras_retinanet import models
from keras.utils.data_utils import get_file

from engine.object_detection_branch.retina_net.keras_retinanet.utils.visualization import draw_box, draw_caption
from engine.object_detection_branch.retina_net.keras_retinanet.utils.colors import label_color

from engine.object_detection_branch.retina_net.keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image

MODEL_PATH = 'https://github.com/GKalliatakis/Keras-EMOTIC-resources/releases/download/v1.0.2/resnet50_coco_best_v2.1.0.h5'


def RetinaNet_single_img_detection(img_path,
                                   imshow = False):

    # load the downloaded/trained model
    model_path = get_file('resnet50_coco_best_v2.h5',
                          MODEL_PATH,
                          cache_subdir='EMOTIC/object_detectors')

    # load RetinaNet model
    model = models.load_model(model_path, backbone_name='resnet50')

    # load label to names mapping
    labels_to_names = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane',
                       5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light',
                       10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench',
                       14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow',
                       20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack',
                       25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee',
                       30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat',
                       35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle',
                       40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl',
                       46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli',
                       51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake', 56: 'chair',
                       57: 'couch', 58: 'potted plant', 59: 'bed', 60: 'dining table', 61: 'toilet', 62: 'tv',
                       63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave',
                       69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase',
                       76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'}

    # load image
    image = read_image_bgr(img_path)


    if imshow:
        draw = image.copy()
        draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)

    # preprocess image for network
    image = preprocess_image(image)
    image, scale = resize_image(image)

    # process image
    boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))

    # correct for image scale
    boxes /= scale


    persons_counter = 0
    # run a for loop to define the number of detected persons
    for box, score, label in zip(boxes[0], scores[0], labels[0]):
        # scores are sorted so we can break
        if score < 0.5:
            break

        # decode predicted labels
        decoded_label = "{}".format(labels_to_names[label])

        if decoded_label == 'person':
            # print('[INFO] `person` was detected')
            persons_counter += 1

            # b = box.astype(int)

        # TODO must handle the cases where no `person` object class was detected
        # else:
        #     print ('[INFO] No `person` was detected')


    final_array = np.empty([persons_counter, 4])
    counter = 0

    for box, score, label in zip(boxes[0], scores[0], labels[0]):

        if counter > persons_counter:
            break

        # scores are sorted so we can break
        if score < 0.5:
            break

        # decode predicted labels
        decoded_label = "{}".format(labels_to_names[label])

        if decoded_label == 'person':

            b = box.astype(int)
            final_array[counter][0] = b[0]
            final_array[counter][1] = b[1]
            final_array[counter][2] = b[2]
            final_array[counter][3] = b[3]

            counter += 1

            if imshow:
                color = label_color(label)
                draw_box(draw, b, color=color)
                caption = "{} {:.3f}".format(labels_to_names[label], score)
                draw_caption(draw, b, caption)



    if imshow:
        plt.figure(figsize=(20, 12))
        plt.axis('off')
        plt.imshow(draw)
        plt.show()


    return final_array, persons_counter


