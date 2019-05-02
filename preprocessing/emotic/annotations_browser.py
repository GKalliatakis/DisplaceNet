from __future__ import print_function
import os
from scipy.io import loadmat
from utils.generic_utils import crop
import csv

from array import array

import shutil
import tqdm


from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


class AnnotationsBrowser():

    def __init__(self,
                 matfile = 'Annotations.mat',
                 EMOTIC_base_dir = '/home/sandbox/Desktop/EMOTIC_database/emotic',
                 mode = 'train'
                 ):
        """Annotations browser base class.

            Example
            --------
            >>> browser= AnnotationsBrowser(matfile='Annotations.mat',
            >>>                             EMOTIC_base_dir='/home/sandbox/Desktop/EMOTIC_database/emotic',
            >>>                             mode= 'train')
            >>> if browser.mode == 'train':
            >>>     nb_samples = 17077
            >>> elif browser.mode == 'val':
            >>>     nb_samples = 2088
            >>> elif browser.mode == 'test':
            >>>     nb_samples = 4389

            >>> dir_name = browser.mode +'/'

            >>> bounding_rectangles_dir =  dir_name + 'images'
            >>> emotion_categories_filename = dir_name + 'emotions.csv'

            >>> for field_number in range(0, nb_samples):
            >>>     browser.crop_bounding_rectangles(field_number=field_number, to_file=bounding_rectangles_dir)
            >>>     browser.emotion_categories(field_number,emotion_categories_filename)
        """

        self.matfile = matfile
        self.matdata = loadmat(matfile)
        self.emotic_base_dir = EMOTIC_base_dir
        self.mode = mode

        if mode == 'train':
            self.categories_field_name = 'annotations_categories'
            self.continuous_field_name = 'annotations_continuous'
        else:
            self.categories_field_name = 'combined_categories'
            self.continuous_field_name = 'combined_continuous'



    def copy_images(self,
                    field_number,
                    copy_dir):
        """Saves a copy for every image found in annotations at a given directory.
            # Arguments
                field_number: serial number of every image sample in the database.
                to_file: name of the csv file to save the exported data.
        """

        folder = self.matdata[self.mode][0]['folder'][field_number][0]
        filename = self.matdata[self.mode][0]['filename'][field_number][0]

        # print (filename)

        src_full_path = self.emotic_base_dir + '/' + folder + '/' + filename
        copy_full_path = str(copy_dir)+str(field_number)+'_'+str(filename)


        # print (src_full_path)
        # print (copy_full_path)

        shutil.copyfile(src_full_path, copy_full_path)



    def multi_copy_images(self,
                          field_number,
                          copy_dir):
        """Saves a separate image copy for every person under consideration
            (this is relevant for the entire image module of the CNN model) found in annotations.
            # Arguments
                field_number: serial number of every image sample in the database.
                to_file: name of the csv file to save the exported data.
        """


        nb_annotated_persons = len(self.matdata[self.mode][0]['person'][field_number][0])

        for x in range(0, nb_annotated_persons):

            folder = self.matdata[self.mode][0]['folder'][field_number][0]
            filename = self.matdata[self.mode][0]['filename'][field_number][0]

            full_path = self.emotic_base_dir +'/'+ folder+'/'+filename


            filename_only, file_extension_only = os.path.splitext(filename)

            # EMOTIC dataset contains up to 14 different persons annotated in the images
            if x == 0:
                to_file = str(field_number) + '_' + filename_only + '.jpg'
            elif x == 1:
                to_file = str(field_number) + '_' + filename_only + '_B.jpg'
            elif x == 2:
                to_file = str(field_number) + '_' + filename_only + '_C.jpg'
            elif x == 3:
                to_file = str(field_number) + '_' + filename_only + '_D.jpg'
            elif x == 4:
                to_file = str(field_number) + '_' + filename_only + '_E.jpg'
            elif x == 5:
                to_file = str(field_number) + '_' + filename_only + '_F.jpg'
            elif x == 6:
                to_file = str(field_number) + '_' + filename_only + '_G.jpg'
            elif x == 7:
                to_file = str(field_number) + '_' + filename_only + '_H.jpg'
            elif x == 8:
                to_file = str(field_number) + '_' + filename_only + '_I.jpg'
            elif x == 9:
                to_file = str(field_number) + '_' + filename_only + '_J.jpg'
            elif x == 10:
                to_file = str(field_number) + '_' + filename_only + '_K.jpg'
            elif x == 11:
                to_file = str(field_number) + '_' + filename_only + '_L.jpg'
            elif x == 12:
                to_file = str(field_number) + '_' + filename_only + '_M.jpg'
            elif x == 13:
                to_file = str(field_number) + '_' + filename_only + '_N.jpg'


            copy_full_path = copy_dir + to_file

            # print (copy_full_path)

            shutil.copyfile(full_path, copy_full_path)



    def crop_bounding_rectangles(self,
                                 field_number,
                                 to_file,
                                 ):
        """Crops and saves a separate image for every person under consideration
            based on the bounding rectangle of the corresponding 4 co-ordinates found in annotations.
            # Arguments
                field_number: serial number of every image sample in the database.
                to_file: name of the csv file to save the exported data.
        """

        nb_annotated_persons = len(self.matdata[self.mode][0]['person'][field_number][0])

        for x in range(0, nb_annotated_persons):

            folder = self.matdata[self.mode][0]['folder'][field_number][0]
            filename = self.matdata[self.mode][0]['filename'][field_number][0]

            full_path = self.emotic_base_dir +'/'+ folder+'/'+filename

            body_bbox_tuple = ()
            x1_body_bbox = self.matdata[self.mode][0]['person'][field_number]['body_bbox'][0][x][0][0]
            y1_body_bbox = self.matdata[self.mode][0]['person'][field_number]['body_bbox'][0][x][0][1]
            x2_body_bbox = self.matdata[self.mode][0]['person'][field_number]['body_bbox'][0][x][0][2]
            y2_body_bbox = self.matdata[self.mode][0]['person'][field_number]['body_bbox'][0][x][0][3]

            body_bbox_tuple = body_bbox_tuple + (x1_body_bbox,y1_body_bbox,x2_body_bbox,y2_body_bbox,)

            filename_only, file_extension_only = os.path.splitext(filename)

            # EMOTIC dataset contains up to 14 different persons annotated in the images
            if x == 0:
                to_file = to_file + '/' + str(field_number) + '_' + filename_only + '.jpg'
            elif x == 1:
                to_file = to_file + '/' + str(field_number) + '_' + filename_only + '_B.jpg'
            elif x == 2:
                to_file = to_file + '/' + str(field_number) + '_' + filename_only + '_C.jpg'
            elif x == 3:
                to_file = to_file + '/' + str(field_number) + '_' + filename_only + '_D.jpg'
            elif x == 4:
                to_file = to_file + '/' + str(field_number) + '_' + filename_only + '_E.jpg'
            elif x == 5:
                to_file = to_file + '/' + str(field_number) + '_' + filename_only + '_F.jpg'
            elif x == 6:
                to_file = to_file + '/' + str(field_number) + '_' + filename_only + '_G.jpg'
            elif x == 7:
                to_file = to_file + '/' + str(field_number) + '_' + filename_only + '_H.jpg'
            elif x == 8:
                to_file = to_file + '/' + str(field_number) + '_' + filename_only + '_I.jpg'
            elif x == 9:
                to_file = to_file + '/' + str(field_number) + '_' + filename_only + '_J.jpg'
            elif x == 10:
                to_file = to_file + '/' + str(field_number) + '_' + filename_only + '_K.jpg'
            elif x == 11:
                to_file = to_file + '/' + str(field_number) + '_' + filename_only + '_L.jpg'
            elif x == 12:
                to_file = to_file + '/' + str(field_number) + '_' + filename_only + '_M.jpg'
            elif x == 13:
                to_file = to_file + '/' + str(field_number) + '_' + filename_only + '_N.jpg'


            crop(full_path, body_bbox_tuple, to_file)


    def emotion_categories(self,
                           field_number,
                           to_file):
        """Exports and saves the one-hot encoded emotion categories for every person under consideration.
            # Arguments
                field_number: serial number of every image sample in the database.
                to_file: name of the csv file to save the exported data.
        """

        emotion_categories_list = []
        nb_annotated_persons = len(self.matdata[self.mode][0]['person'][field_number][0])


        # define universe of possible input values
        alphabet = ['Affection',
                    'Anger',
                    'Annoyance',
                    'Anticipation',
                    'Aversion',
                    'Confidence',
                    'Disapproval',
                    'Disconnection',
                    'Disquietment',
                    'Doubt/Confusion',
                    'Embarrassment',
                    'Engagement',
                    'Esteem',
                    'Excitement',
                    'Fatigue',
                    'Fear',
                    'Happiness',
                    'Pain',
                    'Peace',
                    'Pleasure',
                    'Sadness',
                    'Sensitivity',
                    'Suffering',
                    'Surprise',
                    'Sympathy',
                    'Yearning']

        # define a mapping of chars to integers
        char_to_int = dict((c, i) for i, c in enumerate(alphabet))
        int_to_char = dict((i, c) for i, c in enumerate(alphabet))


        # Loop over every `person` under consideration (`nb_annotated_persons`)
        for person in xrange(0, nb_annotated_persons):
            # nb_emotion_categories = len(
            #     self.matdata[self.mode][0]['person'][field_number][self.categories_field_name][0][person][0][0][0][0])

            if self.mode == 'train':
                nb_emotion_categories = len(
                    self.matdata[self.mode][0]['person'][field_number][self.categories_field_name][0][person][0][0][0][0])

            else:
                nb_emotion_categories = len(
                    self.matdata[self.mode][0]['person'][field_number][self.categories_field_name][0][person][0])


            multiple_persons_emotion_categories_list = []

            integer_encoded_list = []
            integer_encoded__multiple_list = []
            # Loop over the total number of emotion categories found for `person`
            for emotion in xrange(0, nb_emotion_categories):

                if self.mode == 'train':
                    current_emotion_category = \
                        self.matdata[self.mode][0]['person'][field_number][self.categories_field_name][0][person][0][0][0][0][emotion][0]
                else:
                    current_emotion_category = \
                        self.matdata[self.mode][0]['person'][field_number][self.categories_field_name][0][person][0][emotion][0]

                # emotion_as_int = [char_to_int[char] for char in current_emotion_category]

                if current_emotion_category == 'Affection':
                    emotion_as_int = 0

                elif current_emotion_category == 'Anger':
                    emotion_as_int = 1

                elif current_emotion_category == 'Annoyance':
                    emotion_as_int = 2

                elif current_emotion_category == 'Anticipation':
                    emotion_as_int = 3

                elif current_emotion_category == 'Aversion':
                    emotion_as_int = 4

                elif current_emotion_category == 'Confidence':
                    emotion_as_int = 5

                elif current_emotion_category == 'Disapproval':
                    emotion_as_int = 6

                elif current_emotion_category == 'Disconnection':
                    emotion_as_int = 7

                elif current_emotion_category == 'Disquietment':
                    emotion_as_int = 8

                elif current_emotion_category == 'Doubt/Confusion':
                    emotion_as_int = 9

                elif current_emotion_category == 'Embarrassment':
                    emotion_as_int = 10

                elif current_emotion_category == 'Engagement':
                    emotion_as_int = 11

                elif current_emotion_category == 'Esteem':
                    emotion_as_int = 12

                elif current_emotion_category == 'Excitement':
                    emotion_as_int = 13

                elif current_emotion_category == 'Fatigue':
                    emotion_as_int = 14

                elif current_emotion_category == 'Fear':
                    emotion_as_int = 15

                elif current_emotion_category == 'Happiness':
                    emotion_as_int = 16

                elif current_emotion_category == 'Pain':
                    emotion_as_int = 17

                elif current_emotion_category == 'Peace':
                    emotion_as_int = 18

                elif current_emotion_category == 'Pleasure':
                    emotion_as_int = 19

                elif current_emotion_category == 'Sadness':
                    emotion_as_int = 20

                elif current_emotion_category == 'Sensitivity':
                    emotion_as_int = 21

                elif current_emotion_category == 'Suffering':
                    emotion_as_int = 22

                elif current_emotion_category == 'Surprise':
                    emotion_as_int = 23

                elif current_emotion_category == 'Sympathy':
                    emotion_as_int = 24

                elif current_emotion_category == 'Yearning':
                    emotion_as_int = 25



                # Checks if the number of people under consideration is one (single person in the image) or
                # more than one (multiple persons in the image) and create their corresponding lists
                # that will hold the `current_emotion_category`
                if nb_annotated_persons == 1:

                    emotion_categories_list.append(int(emotion_as_int))

                else:

                    multiple_persons_emotion_categories_list.append(int(emotion_as_int))


            # # integer encode for single and multiple persons
            # integer_encoded = [char_to_int[char] for char in emotion_categories_list]
            # integer_encoded_multiple = [char_to_int[char] for char in multiple_persons_emotion_categories_list]
            #
            # if nb_annotated_persons == 1:
            #     integer_encoded_list.append(integer_encoded)
            # else:
            #     integer_encoded__multiple_list.append(integer_encoded_multiple)



            with open(to_file, 'a') as resultFile:
                wr = csv.writer(resultFile, dialect='excel')

                if nb_annotated_persons == 1:
                    print (emotion_categories_list)
                    wr.writerow(emotion_categories_list)
                else:
                    print (multiple_persons_emotion_categories_list)
                    wr.writerow(multiple_persons_emotion_categories_list)



    def continuous_dimensions(self,
                              field_number,
                              dimension,
                              to_file):
        """Exports and saves the continuous dimension annotations for every person under consideration.
            # Arguments
                field_number: serial number of every image sample in the database.
                dimension: one of `valence` (how positive or pleasant an emotion is),
                    `arousal` (measures the agitation level of the person) or
                    `dominance` (measures the control level of the situation by the person).
                to_file: name of the csv file to save the exported data.
        """

        if dimension == 'valence':
            int_dimension = 0
        elif dimension == 'arousal':
            int_dimension = 1
        elif dimension == 'dominance':
            int_dimension = 2

        dimension_list = []

        nb_annotated_persons = len(self.matdata[self.mode][0]['person'][field_number][0])

        print ('=== Processing field ' + str(field_number + 1) + ' ===')

        for person in xrange(0, nb_annotated_persons):
            multiple_persons_dimension_list = []

            current_valence_dimension = \
                self.matdata[self.mode][0]['person'][field_number][self.continuous_field_name][0][person][0][0][int_dimension][
                    0][0]

            # print current_valence_dimension



            if nb_annotated_persons == 1:
                dimension_list.append(current_valence_dimension)

                with open(to_file, 'a') as resultFile:
                    print (dimension_list)
                    wr = csv.writer(resultFile, dialect='excel')
                    wr.writerow(dimension_list)
            else:
                multiple_persons_dimension_list.append(current_valence_dimension)

                with open(to_file, 'a') as resultFile:
                    print (multiple_persons_dimension_list)
                    wr = csv.writer(resultFile, dialect='excel')
                    wr.writerow(multiple_persons_dimension_list)


    def age(self,
            field_number,
            to_file):
        """Exports and saves the annotated age categories for every person under consideration in train set
            # Arguments
                field_number: serial number of every image sample in the database.
                to_file: name of the csv file to save the exported data.
        """

        age_list = []

        nb_annotated_persons = len(self.matdata[self.mode][0]['person'][field_number][0])

        print ('=== Processing field ' + str(field_number + 1) + ' ===')

        for person in xrange(0, nb_annotated_persons):
            multiple_persons_age_list = []

            current_age = self.matdata[self.mode][0]['person'][field_number]['age'][0][person][0]

            print (current_age)

            if current_age == 'Kid':
                current_age_int = 0
            elif current_age == 'Teenager':
                current_age_int = 1
            elif current_age == 'Adult':
                current_age_int = 2


            if nb_annotated_persons == 1:
                age_list.append(current_age_int)

                with open(to_file, 'a') as resultFile:
                    wr = csv.writer(resultFile, dialect='excel')
                    wr.writerow(age_list)
            else:
                multiple_persons_age_list.append(current_age_int)

                with open(to_file, 'a') as resultFile:
                    wr = csv.writer(resultFile, dialect='excel')
                    wr.writerow(multiple_persons_age_list)



    def age_single_label_categorical(self,
                                     field_number,
                                     to_file):
        """Exports and saves the annotated age categories for every person under consideration in train set
            # Arguments
                field_number: serial number of every image sample in the database.
                to_file: name of the csv file to save the exported data.
        """

        age_list = []

        nb_annotated_persons = len(self.matdata[self.mode][0]['person'][field_number][0])

        print ('=== Processing field ' + str(field_number + 1) + ' ===')

        for person in xrange(0, nb_annotated_persons):
            multiple_persons_age_list = []

            current_age = self.matdata[self.mode][0]['person'][field_number]['age'][0][person][0]

            print (current_age)

            if current_age == 'Kid':
                current_age_int = 0
            elif current_age == 'Teenager':
                current_age_int = 1
            elif current_age == 'Adult':
                current_age_int = 2

            if nb_annotated_persons == 1:
                age_list.append(current_age_int)

                with open(to_file, 'a') as resultFile:
                    wr = csv.writer(resultFile, dialect='excel')
                    wr.writerow(age_list)
            else:
                multiple_persons_age_list.append(current_age_int)

                with open(to_file, 'a') as resultFile:
                    wr = csv.writer(resultFile, dialect='excel')
                    wr.writerow(multiple_persons_age_list)



