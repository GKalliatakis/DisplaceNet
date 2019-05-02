"""Custom generator for pentuple-output Keras models.
"""

from math import ceil


def custom_generator(hdf5_file, nb_data, batch_size, mode):
    """ Generates batches of tensor image data in form of ==> (x1, y1) ,(x2, y2).
        # Reference
        - https://stackoverflow.com/questions/50333532/load-images-and-annotations-from-csv-and-use-fit-generator-with-multi-output-mod
        - http://machinelearninguru.com/deep_learning/data_preparation/hdf5/hdf5.html

        # Arguments
            hdf5_file: path or hdf5 object which contains the images and the annotations.
            nb_data: total number of samples saved in the array.
            batch_size: size of the batch to generate tensor image data for.
            module: one of `body` or `image`.

        # Returns
            A generator object.

    """

    batches_list = list(range(int(ceil(float(nb_data) / batch_size))))

    while True:

        # loop over batches
        for n, i in enumerate(batches_list):
            i_s = i * batch_size  # index of the first image in this batch
            i_e = min([(i + 1) * batch_size, nb_data])  # index of the last image in this batch

            if mode == 'train':
                body_x = hdf5_file["x_cropped_train"][i_s:i_e, ...]
                image_x = hdf5_file["x_entire_train"][i_s:i_e, ...]
                # valence_body_y = hdf5_file["valence_cropped_train"][i_s:i_e]
                valence_image_y = hdf5_file["valence_entire_train"][i_s:i_e]
                # arousal_body_y = hdf5_file["arousal_cropped_train"][i_s:i_e]
                arousal_image_y = hdf5_file["arousal_entire_train"][i_s:i_e]
                # dominance_body_y = hdf5_file["dominance_cropped_train"][i_s:i_e]
                dominance_image_y = hdf5_file["dominance_entire_train"][i_s:i_e]

            elif mode == 'val':
                body_x = hdf5_file["x_cropped_val"][i_s:i_e, ...]
                image_x = hdf5_file["x_entire_val"][i_s:i_e, ...]
                # valence_body_y = hdf5_file["valence_cropped_val"][i_s:i_e]
                valence_image_y = hdf5_file["valence_entire_val"][i_s:i_e]
                # arousal_body_y = hdf5_file["arousal_cropped_val"][i_s:i_e]
                arousal_image_y = hdf5_file["arousal_entire_val"][i_s:i_e]
                # dominance_body_y = hdf5_file["dominance_cropped_val"][i_s:i_e]
                dominance_image_y = hdf5_file["dominance_entire_val"][i_s:i_e]

            elif mode == 'test':
                body_x = hdf5_file["x_cropped_test"][i_s:i_e, ...]
                image_x = hdf5_file["x_entire_test"][i_s:i_e, ...]
                # valence_body_y = hdf5_file["valence_cropped_test"][i_s:i_e]
                valence_image_y = hdf5_file["valence_entire_test"][i_s:i_e]
                # arousal_body_y = hdf5_file["arousal_cropped_test"][i_s:i_e]
                arousal_image_y = hdf5_file["arousal_entire_test"][i_s:i_e]
                # dominance_body_y = hdf5_file["dominance_cropped_test"][i_s:i_e]
                dominance_image_y = hdf5_file["dominance_entire_test"][i_s:i_e]


        yield [body_x,image_x], [valence_image_y,arousal_image_y,dominance_image_y]

def custom_generator_single_output(hdf5_file, nb_data, batch_size, mode):
        """ Generates batches of tensor image data in form of ==> (x1, y1) ,(x2, y2).
            # Reference
            - https://stackoverflow.com/questions/50333532/load-images-and-annotations-from-csv-and-use-fit-generator-with-multi-output-mod
            - http://machinelearninguru.com/deep_learning/data_preparation/hdf5/hdf5.html

            # Arguments
                hdf5_file: path or hdf5 object which contains the images and the annotations.
                nb_data: total number of samples saved in the array.
                batch_size: size of the batch to generate tensor image data for.
                module: one of `body` or `image`.

            # Returns
                A generator object.

        """

        batches_list = list(range(int(ceil(float(nb_data) / batch_size))))

        while True:

            # loop over batches
            for n, i in enumerate(batches_list):
                i_s = i * batch_size  # index of the first image in this batch
                i_e = min([(i + 1) * batch_size, nb_data])  # index of the last image in this batch

                if mode == 'train':
                    body_x = hdf5_file["x_body_train"][i_s:i_e, ...]
                    image_x = hdf5_file["x_image_train"][i_s:i_e, ...]
                    y = hdf5_file["y_image_train"][i_s:i_e]

                elif mode == 'val':
                    body_x = hdf5_file["x_body_val"][i_s:i_e, ...]
                    image_x = hdf5_file["x_image_val"][i_s:i_e, ...]
                    y = hdf5_file["y_image_val"][i_s:i_e]

                elif mode == 'test':
                    body_x = hdf5_file["x_body_test"][i_s:i_e, ...]
                    image_x = hdf5_file["x_image_test"][i_s:i_e, ...]
                    y = hdf5_file["y_image_test"][i_s:i_e]

            yield [body_x, image_x], y