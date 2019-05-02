from __future__ import print_function
import os
import argparse


from sklearn.metrics import accuracy_score, classification_report, precision_score, confusion_matrix, average_precision_score
from inference.displacenet_single_image_inference_unified import displaceNet_inference

class AbuseNetBaseEvaluator(object):
    """Perfofmance metrics base class.
    """


    def __init__(self,
                 hra_model_backend_name,nb_of_conv_layers_to_fine_tune,
                 emotic_model_a_backend_name,emotic_model_b_backend_name,emotic_model_c_backend_name,
                 violation_class,
                 main_test_dir ='/home/sandbox/Desktop/Human_Rights_Archive_DB/test',
                 ):

        self.hra_model_backend_name = hra_model_backend_name
        self.nb_of_conv_layers_to_fine_tune = nb_of_conv_layers_to_fine_tune
        self.emotic_model_a_backend_name = emotic_model_a_backend_name
        self.emotic_model_b_backend_name = emotic_model_b_backend_name
        self.emotic_model_c_backend_name = emotic_model_c_backend_name
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
                print(' Processing  [' + raw_img + ']')
                final_img = os.path.join(tmp, raw_img)

                preds = displaceNet_inference(img_path=final_img,
                                              emotic_model_a_backend_name=self.emotic_model_a_backend_name,
                                              emotic_model_b_backend_name=self.emotic_model_b_backend_name,
                                              emotic_model_c_backend_name=self.emotic_model_c_backend_name,
                                              hra_model_backend_name=self.hra_model_backend_name,
                                              nb_of_fine_tuned_conv_layers=self.nb_of_conv_layers_to_fine_tune,
                                              violation_class=self.violation_class)


                preds = preds[0]

                y_pred.append(int(preds[0][0]))
                y_scores.append(preds[0][2])

                top_1_predicted_probability = preds[0][2]

                # top_1_predicted = np.argmax(preds)
                top_1_predicted_label = preds[0][1]

                if top_1_predicted_probability >= prob_threshold:
                    coverage_count += 1

                # print ('`' + hra_class + '/' + raw_img + '`  ===>  `' +
                #        top_1_predicted_label + '`' + ' with ' + str(top_1_predicted_probability) + ' P')

                print('     GT `' + hra_class + '`' + '` <--> ` Pred. `' +
                      top_1_predicted_label + '`' + ' with ' + str(top_1_predicted_probability))

                print ('\n')

                predicted_class_list.append(top_1_predicted_label)
                actual_class_list.append(hra_class)

        total_coverage_per = (coverage_count * 100) / self.total_nb_of_test_images

        return y_pred, self.y_true, y_scores, total_coverage_per



if __name__ == "__main__":

    def get_args():
        parser = argparse.ArgumentParser()
        parser.add_argument("--violation_class", type=str,
                            help='One of `cl` or `dp`')

        parser.add_argument("--hra_model_backend_name", type=str,
                            help='One of `VGG16`, `VGG19`, `ResNet50`, `VGG16_Places365`')

        parser.add_argument("--nb_of_conv_layers", type=int, default=None,
                            help="Number of fine-tuned conv. layers")

        parser.add_argument("--emotic_model_a_backend_name", type=str,
                            help='One of `VGG16`, `VGG19`, `ResNet50`')

        parser.add_argument("--emotic_model_b_backend_name", type=str,
                            help='One of `VGG16`, `VGG19`, `ResNet50`', default=None)

        parser.add_argument("--emotic_model_c_backend_name", type=str,
                            help='One of `VGG16`, `VGG19`, `ResNet50`', default=None)

        args = parser.parse_args()
        return args


    args = get_args()




    # server
    if args.violation_class == 'cl':
        main_test_dir = '/home/gkallia/git/AbuseNet/datasets/HRA-2clas-full-test/ChildLabour'
    elif args.violation_class =='dp':
        main_test_dir = '/home/gkallia/git/AbuseNet/datasets/HRA-2clas-full-test/DisplacedPopulations'

    # ---------------------------------------------------- #





    base_evaluator = AbuseNetBaseEvaluator(hra_model_backend_name=args.hra_model_backend_name,
                                           nb_of_conv_layers_to_fine_tune=args.nb_of_conv_layers,
                                           emotic_model_a_backend_name=args.emotic_model_a_backend_name,
                                           emotic_model_b_backend_name=args.emotic_model_b_backend_name,
                                           emotic_model_c_backend_name=args.emotic_model_c_backend_name,
                                           violation_class=args.violation_class,
                                           main_test_dir =main_test_dir,
                                           )

    y_pred, y_true, y_scores, total_coverage_per = base_evaluator._obtain_y_pred()

    # print y_true
    top1_acc = accuracy_score(y_true, y_pred)

    AP = average_precision_score(y_true, y_scores, 'micro')


    string = args.hra_model_backend_name+'-'+args.violation_class+'-'+str(args.nb_of_conv_layers)+'layer(s)-'

    print('\n')
    print( '============================= %s =============================' %string)
    print(' Top-1 acc. =>  ' + str(top1_acc))
    print(' Coverage =>  ' + str(total_coverage_per) + '%')
    print(' Average Precision (AP)  =>  ' + str(AP) + '%')