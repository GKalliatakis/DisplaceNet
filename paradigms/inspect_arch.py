from __future__ import division, print_function
import os

from keras.layers import Input
from keras.layers.core import Dense
from keras.models import Model
from keras.layers.core import Dropout
from keras.layers import GlobalAveragePooling2D
from keras.utils.data_utils import get_file
from keras import regularizers
from keras.utils import plot_model


from keras.applications.resnet50 import ResNet50
from keras.layers.merge import concatenate
from applications.vgg16_places_365 import VGG16_Places365
from keras.optimizers import SGD

from utils.generic_utils import euclidean_distance_loss, rmse


body_inputs = Input(shape=(224, 224, 3), name='INPUT')
image_inputs = Input(shape=(224, 224, 3), name='INPUT')

# Body module
tmp_model = ResNet50(include_top=False, weights='imagenet', input_tensor=body_inputs, pooling='avg')
# tmp_model.summary()

plot_model(tmp_model, to_file='original_model.png',show_shapes = True)

for i, layer in enumerate(tmp_model.layers):
    print(i, layer.name)

new_model = Model(inputs=tmp_model.input, outputs=tmp_model.get_layer(index=169).output)

for i, layer in enumerate(new_model.layers):
    print(i, layer.name)

# tmp_model.summary()

plot_model(new_model, to_file='intermediate_layer.png',show_shapes = True)




