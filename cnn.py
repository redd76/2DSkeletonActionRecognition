import numpy as np
import cv2 as cv2
from utils import mnist_reader
from utils import plot_utils
import os
from keras.utils import np_utils
from models.CNNModel import CNNModel
from datagen.DataGenerator import DataGenerator
# uncomment for debugging
# show 9 grayscale images as examples of the data set
# ------- start show images ----------
# import sys
# for i in range(9):
#     plt.subplot(3,3,i+1)
#     plt.imshow(X_train[i].reshape(28,28), cmap='gray', interpolation='none')
#     plt.title("Class {}".format(y_train[i]))
#
# plt.show()
# sys.exit()
# ------- end show images ----------

# load data
#data_path = r"D:/dataset2/train/datax/preprocessed3/"
data_path = r"./data/train2/"
"""
classifications = {"stand": [1.0, 0.0, 0.0, 0.0, 0.0],
                   "sit": [0.0, 1.0, 0.0, 0.0, 0.0],
                   "walk": [0.0, 0.0, 1.0, 0.0, 0.0],
                   "run": [0.0, 0.0, 0.0, 1.0, 0.0],
                   "jump": [0.0, 0.0, 0.0, 0.0, 1.0]}

classifications = {"stand": 0,
                   "sit": 1,
                   "walk": 2,
                   "run": 3,
                   "jump": 4}
"""
classifications = {"stand": 0,
                   "walk": 1,
                   "squat": 2,
                   "wave": 3,
                   "punch": 4}

nr_of_people = ["np1"]
quality = ["goo"]
X, y = [], []
for cl, out_vec in classifications.items():
    cl_path = data_path + cl + r"/"
    for img in os.listdir(cl_path):
        X.append(cl_path + img)
        y.append(out_vec)
X = np.asarray(X)
y = np.asarray(y)
# shuffle the data
indices = np.arange(X.shape[0])
np.random.shuffle(indices)
X = X[indices]
y = y[indices]

# hyperparameter
nb_epoch = 50
batch_size = 32
nb_classes = len(classifications.keys())

params = {'dim': (60, 25),
 'batch_size': batch_size,
 'n_classes': nb_classes,
 'n_channels': 3,
 'shuffle': True}

partition = {"train": list(range(0, len(X) - 400)),
            "validation": list(range(len(X) - 400, len(X)))}

model = CNNModel.load_model2(nb_classes)
model.img_rows, model.img_cols, model.n_channels = params['dim'][0], params['dim'][1], params["n_channels"]
model.summary()

training_generator = DataGenerator(partition['train'], X, y, **params)
validation_generator = DataGenerator(partition['validation'], X, y, **params)

#history = model.fit(X_train, Y_train, batch_size=batch_size, epochs=nb_epoch, verbose=1, validation_data=(X_test, Y_test))
history = model.fit_generator(generator=training_generator,
 validation_data=validation_generator, epochs=nb_epoch)
model.summary()
model.save('model2')
#score = model.evaluate(X_test, Y_test, verbose=0)
#print('Test score:', score[0])
#print('Test accuracy:', score[1])

#plot_utils.plot_model_history(history)
#plot_utils.plot_result_examples(model, X_test, y_test, img_rows, img_cols)

