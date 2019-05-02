import matplotlib.pyplot as plt
import numpy as np

import matplotlib.cbook as cbook
from matplotlib import style

style.use('ggplot')

epoch, acc, loss, val_acc, val_loss = np.loadtxt('/home/sandbox/Desktop/ResNet50__VGG16_Places365_emotic.csv',
                                                 unpack=True,
                                                 delimiter=',')






plt.style.use("ggplot")
plt.figure()
plt.plot(epoch, acc, label='Training acc')
plt.plot(epoch, val_acc,label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()

plt.plot(epoch, loss, label='Training loss')
plt.plot(epoch, val_loss, label='Validation loss')
# plt.title('Training and validation fbeta_score')
plt.legend()
plt.show()

#
# # summarize history for accuracy
# plt.plot(acc)
# plt.plot(val_acc)
# plt.title('model accuracy')
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
# plt.legend(['train', 'val'], loc='upper left')
# plt.show()
#
# # summarize history for loss
# plt.plot(loss)
# plt.plot(val_loss)
# plt.title('model loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['train', 'val'], loc='upper left')
# plt.show()


# print (epoch)
# print (acc)
# print (loss)
# print (top_3_categorical_accuracy)


# plt.plot(acc)
# plt.plot(loss)
# plt.plot(top_3_categorical_accuracy)
# plt.title('model accuracy')
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
# plt.legend(['acc', 'loss', 'top3'], loc='upper left')
# plt.show()
#
#
#
# plt.plot(epoch,acc)
# # plt.plot(epoch,loss)
#
#
#
# plt.title('Model Training History')
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
# # plt.legend(['Top-1 ','Top-3' ,'fbeta'], loc='upper left')
# plt.legend(['Top-1 '], loc='upper left')
#
#
# plt.show()
#
#


