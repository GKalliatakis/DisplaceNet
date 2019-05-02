from sklearn.metrics import accuracy_score, classification_report, precision_score, confusion_matrix,average_precision_score

import numpy as np
import matplotlib.pyplot as plt


from handcrafted_metrics import HRA_metrics, plot_confusion_matrix

from applications.hra_resnet50 import HRA_ResNet50
from applications.hra_vgg16 import HRA_VGG16
from applications.hra_vgg19 import HRA_VGG19
from applications.hra_vgg16_places365 import HRA_VGG16_Places365

# from applications.latest.hra_vgg16_checkpoint import HRA_VGG16
# from applications.latest.hra_vgg16_places365 import HRA_VGG16_Places365
# from applications.latest.compoundNet_vgg16_checkpoint import CompoundNet_VGG16


def _obtain_model(model_backend_name,
                  violation_class,
                  nb_of_conv_layers_to_fine_tune):

    if model_backend_name == 'VGG16':
        model = HRA_VGG16(weights='HRA',
                          violation_class=violation_class,
                          nb_of_conv_layers_to_fine_tune=nb_of_conv_layers_to_fine_tune)



    elif model_backend_name == 'VGG19':
        model = HRA_VGG19(weights='HRA',
                          violation_class=violation_class,
                          nb_of_conv_layers_to_fine_tune=nb_of_conv_layers_to_fine_tune)


    elif model_backend_name == 'ResNet50':
        model = HRA_ResNet50(weights='HRA',
                             violation_class=violation_class,
                             nb_of_conv_layers_to_fine_tune=nb_of_conv_layers_to_fine_tune)


    elif model_backend_name == 'VGG16_Places365':
        model = HRA_VGG16_Places365(weights='HRA',
                                    violation_class=violation_class,
                                    nb_of_conv_layers_to_fine_tune=nb_of_conv_layers_to_fine_tune)

    return model


violation_class = 'dp'
model_backend_name = 'VGG16'
nb_of_conv_layers_to_fine_tune = 2

model = _obtain_model(model_backend_name=model_backend_name,
                      violation_class=violation_class,
                      nb_of_conv_layers_to_fine_tune=nb_of_conv_layers_to_fine_tune)



metrics = HRA_metrics(main_test_dir ='/home/sandbox/Desktop/Two-class-HRV/ChildLabour/test')

[y_true, y_pred, y_score] = metrics.predict_labels(model)


print y_true
print y_pred
print y_score


# print y_true
top1_acc = accuracy_score(y_true, y_pred)

# top5_acc = top_k_accuracy_score(y_true=y_true, y_pred=y_pred,k=3,normalize=True)
coverage = metrics.coverage(model,prob_threshold=0.85)
# coverage = metrics.coverage_duo_ensemble(model_a,model_b,prob_threshold=0.85)


# AP = average_precision_score (y_true = y_true, y_score=y_score)
#
# print AP



print ('\n')
print ('=======================================================================================================')
print (model_backend_name+' Top-1 acc. =>  '+str(top1_acc))
print (model_backend_name+' Coverage =>  '+str(coverage)+'%')

#
#
# target_names = ['arms', 'child_labour', 'child_marriage', 'detention_centres', 'disability_rights', 'displaced_populations',
#                         'environment', 'no_violation', 'out_of_school']
#
# result= model_backend_name+'  =>  '+ str(accuracy_score(y_true, y_pred))+ '\n'
# result= model_backend_name+'  =>  '+str(coverage)+'%'+ '\n'
#
#
# f=open("results/coverage_late_fusion.txt", "a+")
# f.write(result+'\n\n')
# # f.write(str(y_pred)+'\n\n')
# f.close()
#
# print(classification_report(y_true, y_pred, target_names=target_names))
#
# print (precision_score(y_true, y_pred, average=None))
#
# cnf_matrix=confusion_matrix(y_true, y_pred)
# np.set_printoptions(precision=2)
#
# # Plot non-normalized confusion matrix
# plt.figure()
# plot_confusion_matrix(cnf_matrix, classes=target_names,
#                       title='Confusion matrix, without normalization')
#
# # Plot normalized confusion matrix
# plt.figure()
# plot_confusion_matrix(cnf_matrix, classes=target_names, normalize=True,
#                       title='Normalized confusion matrix')
#
# plt.show()
#
#
# print (cnf_matrix.diagonal()/cnf_matrix.sum(axis=1))