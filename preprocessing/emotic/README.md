## Preprocessing (chronological order of completion)

### annotations_browser.py
Annotations browser base class which reads metadata from a .mat file [MATLAB formatted data] and saves them to a CSV file.

### save_raw_imgs.py
Reads image names (locations on disk) from a csv file,
then loads them using the basic set of tools for image data provided by Keras (keras/preprocessing/image.py)
and finally saves them in a numpy array.

### csv_to_numpy.py
Reads entries (different annotations) from a csv file, converts them to integer/list of integers(for multilabel-multiclass settings)
and saves them in a numpy array.

### load_data_from_numpy.py
Utilities required to load data (image & their annotations) stored in numpy arrays.
The main function is `load_data_from_numpy` which loads all the applicable arrays.
The supporting function is `load_annotations_only_from_numpy` which loads only the annotations without the image tensors.


### hdf5_controller.py
Contains the HDF5 controller base class which can be used either to create a single HDF5 file
containing a large number of images and their respective annotations (EMOTIC Dataset) or to load a previously saved HDF5 file.

In order to create a single HDF5 file, `load_data_from_numpy.py` requires images (tensors) and
their annotations in numpy arrays to comply with the following structure:

                main_numpy_dir/
                            train/
                                x_train.npy
                                emotions_train.npy
                                valence_train.npy
                                arousal_train.npy
                                dominance_train.npy
                                age_train.npy

                            val/
                                x_val.npy
                                emotions_val.npy
                                valence_val.npy
                                arousal_val.npy
                                dominance_val.npy
                                age_val.npy

                            test/
                                x_test.npy
                                emotions_test.npy
                                valence_test.npy
                                arousal_test.npy
                                dominance_test.npy
                                age_test.npy

Note that in order to obtain the numpy arrays you will need either to download them (images as tensors are not included due to file size) from:
- https://github.com/GKalliatakis/Keras-EMOTIC/releases/download/0.3/train.zip
- https://github.com/GKalliatakis/Keras-EMOTIC/releases/download/0.3/val.zip
- https://github.com/GKalliatakis/Keras-EMOTIC/releases/download/0.3/test.zip

or recreate them using the `csv_to_numpy.py` for the annotations and `save_raw_imgs.py` for the images.


Also, `base_img_dir` must contain the raw images in the following structure:

                base_img_dir/
                            train/
                                images/
                                    xxxxxxxx.jpg
                                    xxxxxxxx.jpg
                                    ...


                            val/
                                images/
                                    xxxxxxxx.jpg
                                    xxxxxxxx.jpg
                                    ...

                            test/
                                images/
                                    xxxxxxxx.jpg
                                    xxxxxxxx.jpg
                                    ...

Note that in order to end up with that structure you will need either to download the images from
- https://github.com/GKalliatakis/Keras-EMOTIC/releases/download/0.1/train.zip
- https://github.com/GKalliatakis/Keras-EMOTIC/releases/download/0.1/val.zip
- https://github.com/GKalliatakis/Keras-EMOTIC/releases/download/0.1/test.zip

or recreate them using the `crop_bounding_rectangles` function from `annotations_browser.py`
