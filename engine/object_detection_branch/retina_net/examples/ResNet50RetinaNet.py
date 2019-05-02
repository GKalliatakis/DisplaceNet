from __future__ import print_function

# import miscellaneous modules
import matplotlib.pyplot as plt
import numpy as np
import os
import time

import cv2
# import keras
# set tf backend to allow memory to grow, instead of claiming everything
# import keras_retinanet
from examples.object_detectors.retina_net.keras_retinanet import models

from engine.object_detectors.retina_net.keras_retinanet.utils import draw_box, draw_caption
from engine.object_detectors.retina_net.keras_retinanet.utils import label_color
from engine.object_detectors.retina_net.keras_retinanet.utils import read_image_bgr, preprocess_image, resize_image

# adjust this to point to your downloaded/trained model
# models can be downloaded here: https://github.com/fizyr/keras-retinanet/releases
model_path = os.path.join('..', 'snapshots', 'resnet50_coco_best_v2.1.0.h5')

# load retinanet model
model = models.load_model(model_path, backbone_name='resnet50')

# if the model is not converted to an inference model, use the line below
# see: https://github.com/fizyr/keras-retinanet#converting-a-training-model-to-inference-model
#model = models.load_model(model_path, backbone_name='resnet50', convert_model=True)

#print(model.summary())

# load label to names mapping for visualization purposes
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


# Run detection on example

# load image
image = read_image_bgr('human_right_viol_2.jpg')

# copy to draw on
draw = image.copy()
draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)

# preprocess image for network
image = preprocess_image(image)
image, scale = resize_image(image)

# process image
start = time.time()
boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))
print("processing time: ", time.time() - start)






# correct for image scale
boxes /= scale

counter = 0


persons_counter = 0
final_array = np.empty([len(boxes[0]), 4])

# visualize detections
for box, score, label in zip(boxes[0], scores[0], labels[0]):
    # scores are sorted so we can break
    if score < 0.5:
        break

    decoded_label = "{}".format(labels_to_names[label])

    if decoded_label == 'person':
        persons_counter = persons_counter + 1

    color = label_color(label)

    b = box.astype(int)
    draw_box(draw, b, color=color)

    final_array[counter][0] = b[0]
    final_array[counter][1] = b[1]
    final_array[counter][2] = b[2]
    final_array[counter][3] = b[3]


    caption = "{} {:.3f}".format(labels_to_names[label], score)
    draw_caption(draw, b, caption)

    counter += 1

print ('Persons found: ', persons_counter)

plt.figure(figsize=(15, 15))
plt.axis('off')
plt.imshow(draw)
plt.show()