## Applications

Applications is the _Keras-like-applications_ module of DisplaceNet.
It provides model definitions and fine-tuned weights for a number of popular archictures, such as VGG16, VGG19, ResNet50 and VGG16-places365.



### Usage

All architectures are compatible with both TensorFlow and Theano, and upon instantiation the models will be built according to the
image dimension ordering set in your Keras configuration file at ~/.keras/keras.json.
For instance, if you have set image_dim_ordering=tf, then any model loaded from this repository will get built according to
the TensorFlow dimension ordering convention, "Width-Height-Depth".

Pre-trained weights can be automatically loaded upon instantiation (weights='HRA' argument in model constructor for
models trained on Human Rights Archive two-class dataset or weights='EMOTIC' for models trained on EMOTIC dataset).
Weights in all cases are automatically downloaded. The input size used was 224x224 for all models.


### Available fine-tuned models
**Models for image classification with weights trained on HRA subset:**
- [VGG16](https://github.com/GKalliatakis/AbuseNet/blob/master/applications/hra_vgg16.py)
- [VGG19](https://github.com/GKalliatakis/AbuseNet/blob/master/applications/hra_vgg19.py)
- [ResNet50](https://github.com/GKalliatakis/AbuseNet/blob/master/applications/hra_resnet50.py)
- [VGG16-places365](https://github.com/GKalliatakis/AbuseNet/blob/master/applications/hra_vgg16_places365.py)


**Models for continuous emotion recognition in Valence-Arousal-Dominance space with weights trained on EMOTIC:**
- [VGG16](https://github.com/GKalliatakis/AbuseNet/blob/master/applications/emotic_vgg16__vgg16_places365.py)
- [VGG19](https://github.com/GKalliatakis/AbuseNet/blob/master/applications/emotic_vgg19__vgg16_places365.py)
- [ResNet50](https://github.com/GKalliatakis/AbuseNet/blob/master/applications/emotic_resnet50__vgg16_places365.py)


```
