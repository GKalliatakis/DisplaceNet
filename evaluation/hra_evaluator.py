from __future__ import print_function
import os

from applications.hra_utils import prepare_input_data, predict_v2
from applications.hra_vgg16 import HRA_VGG16
from applications.hra_vgg19 import HRA_VGG19
from applications.hra_resnet50 import HRA_ResNet50
from applications.hra_vgg16_places365 import HRA_VGG16_Places365
from sklearn.metrics import accuracy_score, classification_report, precision_score, confusion_matrix, average_precision_score
from keras.preprocessing import image

class BaseEvaluator(object):
    """Perfofmance metrics base class.
    """


    def __init__(self,
                 model,
                 model_backend_name,
                 main_test_dir ='/home/sandbox/Desktop/Human_Rights_Archive_DB/test',
                 violation_class = 'cl'
                 ):

        self.model = model
        self.model_backend_name = model_backend_name
        self.main_test_dir = main_test_dir
        self.total_nb_of_test_images = sum([len(files) for r, d, files in os.walk(main_test_dir)])
        self.sorted_categories_names = sorted(os.listdir(main_test_dir))
        self.violation_class = violation_class
        self.y_true = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]



    def _obtain_y_pred(self,
                       prob_threshold=0.75):

        y_pred = []
        y_scores = []

        predicted_class_list = []
        actual_class_list = []
        coverage_count = 0

        for hra_class in self.sorted_categories_names:

            # variable that contains the main dir alongside the selected category
            tmp = os.path.join(self.main_test_dir, hra_class)
            img_names = sorted(os.listdir(tmp))

            for raw_img in img_names:
                # variable that contains the final image to be loaded
                print('Processing  [' + raw_img + ']')
                final_img = os.path.join(tmp, raw_img)

                img = image.load_img(final_img, target_size=(224, 224))

                # if self.model_backend_name == 'VGG16':
                #     x = prepare_input_data(img_path=final_img, objects_or_places_flag='objects')
                #
                # elif self.model_backend_name == 'VGG19':
                #     x = prepare_input_data(img_path=final_img, objects_or_places_flag='objects')
                #
                # elif self.model_backend_name == 'ResNet50':
                #     x = prepare_input_data(img_path=final_img, objects_or_places_flag='objects')
                #
                # elif self.model_backend_name == 'VGG16_Places365':
                #     x = prepare_input_data(img_path=final_img, objects_or_places_flag='places')


                preds = predict_v2(violation_class=self.violation_class,
                                   model=self.model,
                                   img=img,
                                   target_size=(224, 224))

                y_pred.append(int(preds[0][0]))
                y_scores.append(preds[0][2])

                top_1_predicted_probability = preds[0][2]

                # top_1_predicted = np.argmax(preds)
                top_1_predicted_label = preds[0][1]

                if top_1_predicted_probability >= prob_threshold:
                    coverage_count += 1

                print ('`' + hra_class + '/' + raw_img + '`  ===>  `' +
                       top_1_predicted_label + '`' + ' with ' + str(top_1_predicted_probability) + ' P')

                predicted_class_list.append(top_1_predicted_label)
                actual_class_list.append(hra_class)

        total_coverage_per = (coverage_count * 100) / self.total_nb_of_test_images

        return y_pred, self.y_true, y_scores, total_coverage_per



if __name__ == "__main__":

    model_backend_name = 'VGG16'
    nb_of_conv_layers_to_fine_tune = 1
    violation_class = 'cl'

    # ---------------------------------------------------- #



    if violation_class == 'cl':
        main_test_dir = '/home/sandbox/Desktop/HRA-2clas-full-test/ChildLabour'
    elif violation_class =='dp':
        main_test_dir = '/home/sandbox/Desktop/HRA-2clas-full-test/DisplacedPopulations'

    if model_backend_name == 'ResNet50':
        model = HRA_ResNet50(include_top=True, weights='HRA',
                             nb_of_conv_layers_to_fine_tune=nb_of_conv_layers_to_fine_tune,
                             violation_class=violation_class)
    elif model_backend_name == 'VGG16':
        model = HRA_VGG16(include_top=True, weights='HRA',
                          nb_of_conv_layers_to_fine_tune=nb_of_conv_layers_to_fine_tune,
                          violation_class=violation_class)
    elif model_backend_name == 'VGG19':
        model = HRA_VGG19(include_top=True, weights='HRA',
                          nb_of_conv_layers_to_fine_tune=nb_of_conv_layers_to_fine_tune,
                          violation_class=violation_class)
    elif model_backend_name == 'VGG16_Places365':
        model = HRA_VGG16_Places365(include_top=True, weights='HRA',
                                    nb_of_conv_layers_to_fine_tune=nb_of_conv_layers_to_fine_tune,
                                    violation_class=violation_class)




    base_evaluator = BaseEvaluator(model=model,
                                   model_backend_name=model_backend_name,
                                   main_test_dir=main_test_dir,
                                   violation_class=violation_class
                                   )

    y_pred, y_true, y_scores, total_coverage_per = base_evaluator._obtain_y_pred()

    # print y_true
    top1_acc = accuracy_score(y_true, y_pred)

    AP = average_precision_score(y_true, y_scores, 'micro')


    string = model_backend_name+'-'+violation_class+'-'+str(nb_of_conv_layers_to_fine_tune)+'layer(s)'

    print('\n')
    print( '============================= %s =============================' %string)
    print(' Top-1 acc. =>  ' + str(top1_acc))
    print(' Coverage =>  ' + str(total_coverage_per) + '%')
    print(' Average Precision (AP)  =>  ' + str(AP) + '%')