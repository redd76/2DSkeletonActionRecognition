import numpy as np
import cv2 as cv2
from utils import mnist_reader
from utils import plot_utils
import os
from keras.utils import np_utils
import keras.applications.densenet
from models.NNModel import NNModel
from models.CNNModel import CNNModel
from models.TwoStreamModel import TwoStreamModel
from models.AlexNetModel import AlexNetModel
from models.SqueezeNetModel import SqueezeNetModel
from datagen.DataGenerator import DataGenerator
import argparse
from datetime import datetime
import utils.model_utils as mutils
import json

parser = argparse.ArgumentParser(
    description="")
parser.add_argument('--train_images', help='Path to training image.')
parser.add_argument('--model', help='Specify model to train with (AlexNet|DenseNet|TwoStreamNet|SimpleCNNet|NNet)')
parser.add_argument('--epochs', help='Number of epochs to run', default=50)
parser.add_argument('--batch_size', help='Batch size', default=16)
parser.add_argument('--useAllJoints', help='Joint number to use (True=25, False=14)', default=1)
parser.add_argument('--dataset', help='Dataset to use (hmdb51|custom|all)')

args = parser.parse_args()


test = True

if test:
    # fake user input
    args.train_images = r"./data/train/"
    args.model = "AlexNet"
    args.epochs = 50
    args.useAllJoints = 1
    args.dataset = "all"

data_path = args.train_images
model = args.model
classifications = {"stand": 0,
                   "walk": 1,
                   "sit": 2,
                   "wave": 3,
                   "punch": 4}

nb_epoch = args.epochs
batch_size = args.batch_size
nb_classes = len(classifications.keys())
use_all_joints = args.useAllJoints
# get data
dataset = args.dataset
X, y = [], []
for cl, out_vec in classifications.items():
    cl_path = data_path + cl + r"/"
    for img in os.listdir(cl_path):
        if dataset == "custom":
            if "custom" in img.split("_"):
                X.append(cl_path + img)
                y.append(out_vec)
        elif dataset == "hmdb51":
            if not "custom" in img.split("_"):
                X.append(cl_path + img)
                y.append(out_vec)
        elif dataset == "all":
            X.append(cl_path + img)
            y.append(out_vec)
X = np.asarray(X)
y = np.asarray(y)
# shuffle the data
indices = np.arange(X.shape[0])
np.random.shuffle(indices)
X = X[indices]
y = y[indices]

dim = None
if model in ["TwoStreamNet", "SimpleCNNet"]:
    dim = (60, 25) if use_all_joints else (60, 14)
if model in ["AlexNet"]:
    dim = (256, 256)
if model in ["DenseNet"]:
    dim = (224, 224)
if model in ["SqueezeNet"]:
    dim = (227, 227)
if model in ["NNet"]:
    dim = (60 * 25 * 3,) if use_all_joints else (60 * 14 * 3,)

params = {'dim': dim,
        'batch_size': batch_size,
        'n_classes': nb_classes,
        'n_channels': 3,
        'shuffle': True,
        'model':model,
        'use_all_joints':use_all_joints}

partition = {"train": list(range(0, len(X) - int(len(X) * 0.3))),
            "validation": list(range(len(X) - int(len(X) * 0.3), len(X)))}
if model == "AlexNet":
    model = AlexNetModel.load_model(nb_classes)
if model == "DenseNet":
    model = keras.applications.DenseNet121(
    include_top=True,
    weights=None,
    input_tensor=None,
    input_shape=(224,224,3),
    pooling=None,
    classes=nb_classes)
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
if model == "SqueezeNet":
    model = SqueezeNetModel(include_top=True, weights=None,
                            input_tensor=None, input_shape=(227,227,3),
                            pooling=None,
                            classes=nb_classes)
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
if model == "TwoStreamNet":
    net = TwoStreamModel(*dim)
    model = net.load_model(nb_classes)
if model == "SimpleCNNet":
    net = CNNModel(*dim)
    model = net.load_model(nb_classes)
if model == "NNet":
    net = NNModel(dim)
    model = net.load_model(nb_classes)

training_generator = DataGenerator(partition['train'], X, y, **params)
validation_generator = DataGenerator(partition['validation'], X, y, **params)

#history = model.fit(X_train, Y_train, batch_size=batch_size, epochs=nb_epoch, verbose=1, validation_data=(X_test, Y_test))
history = model.fit_generator(generator=training_generator,
 validation_data=validation_generator, epochs=nb_epoch)

current_time = datetime.now()
output_path = r"./data/model_output/{0}".format(current_time.strftime('%Y%m%d%H%M%S'))
if not os.path.exists(output_path):
    os.makedirs(output_path)
mutils.save_model_to_json(os.path.join(output_path, "model.json"), os.path.join(output_path, "weights.hd5"), model)
score = model.evaluate_generator(validation_generator)
print('Test score:', score[0])
print('Test accuracy:', score[1])

plot_utils.plot_model_history_and_save(history, output_path)

plot_utils.save_history_for_plot(history, score, model, os.path.join(output_path, 'history.json'))