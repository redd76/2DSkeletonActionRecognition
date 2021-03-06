import argparse
import os
from datetime import datetime
import time
import random

import keras.applications.densenet
import numpy as np

import utils.model_utils as mutils
from datagen.DataGenerator import DataGenerator
from models.AlexNetModel import AlexNetModel
from models.CNNModel import CNNModel
from models.NNModel import NNModel
from models.ResNetModel import ResNetModel
from models.TwoStreamModel import TwoStreamModel
from utils import plot_utils

# define params from parser
parser = argparse.ArgumentParser(
    description="")
parser.add_argument('--train_images', help='Path to training image.')
parser.add_argument('--model', help='Specify model to train with (AlexNet|DenseNet|ResNet|TwoStreamNet|SimpleCNNet|NNet)')
parser.add_argument('--epochs', help='Number of epochs to run', default=50)
parser.add_argument('--batch_size', help='Batch size', default=16)
parser.add_argument('--useAllJoints', help='Joint number to use (True=25, False=14)', default=1)
parser.add_argument('--dataset', help='Dataset to use (hmdb51|custom|all)')

args = parser.parse_args()

# debug to test the functionalities
debug = False
if debug:
    # fake user input
    args.train_images = r"./data/train/"
    args.model = "DenseNet"
    args.epochs = 50
    args.useAllJoints = 1
    args.batch_size = 30
    args.dataset = "custom"
seed = 50
datalimit_per_class = 400

data_path = args.train_images
model = args.model
# classes
classifications = {"stand": 0,
                   "walk": 1,
                   "sit": 2,
                   "wave": 3,
                   "punch": 4}

# limit the data to balance the data between the classes
nb_epoch = args.epochs
batch_size = args.batch_size
nb_classes = len(classifications.keys())
use_all_joints = args.useAllJoints
# get data
dataset = args.dataset
datacount = {"stand": 0,
            "walk": 0,
            "sit": 0,
            "wave": 0,
            "punch": 0}
# get the input and output for the training
X, y = [], []
data_dict = {}
for cl, out_vec in classifications.items():
    cl_path = data_path + cl + r"/"
    if not cl in data_dict.keys():
        data_dict[cl] = []
    for img in os.listdir(cl_path):
        if args.dataset == "custom":
            if not 'custom' in img.split('_'):
                continue
        if args.dataset == "hmdb51":
            if 'custom' in img.split('_'):
                continue
        if args.dataset == "good":
            if 'goo' not in img.split("_"):
                continue
        data_dict[cl].append(cl_path + img)
        datacount[cl] += 1
print(datacount)

for cl, l in data_dict.items():
    np.random.shuffle(l)
    for i in range(len(l)):
        if i > datalimit_per_class:
            break
        X.append(l[i])
        y.append(classifications[cl])
# shuffle data
X = np.asarray(X)
y = np.asarray(y)
# shuffle the data
indices = np.arange(X.shape[0])
#np.random.seed(seed)
np.random.shuffle(indices)
X = X[indices]
y = y[indices]

# define the dims based on given model
dim = None
if model in ["TwoStreamNet", "SimpleCNNet"]:
    dim = (60, 25) if use_all_joints else (60, 14)
if model in ["AlexNet"]:
    dim = (256, 256)
if model in ["DenseNet", "ResNet"]:
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

training_set = (0, int(len(X) * .35))
validation_set = (training_set[1], int(len(X) * .7))
test_set = (validation_set[1], len(X))
# split training data in train and validation
partition = {"train": list(range(training_set[0], training_set[1])),
            "validation": list(range(validation_set[0], validation_set[1])),
            "test": list(range(test_set[0], test_set[1]))}

# define the model
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
if model == "ResNet":
    net = ResNetModel()
    model = net.load_model(nb_classes)
if model == "TwoStreamNet":
    net = TwoStreamModel(*dim)
    model = net.load_model(nb_classes)
if model == "SimpleCNNet":
    net = CNNModel(*dim)
    model = net.load_model(nb_classes)
if model == "NNet":
    net = NNModel(dim)
    model = net.load_model(nb_classes)

# initialize generators
training_generator = DataGenerator(partition['train'], X, y, **params)
validation_generator = DataGenerator(partition['validation'], X, y, **params)
test_generator = DataGenerator(partition['test'], X, y, **params)

# train
#history = model.fit(X_train, Y_train, batch_size=batch_size, epochs=nb_epoch, verbose=1, validation_data=(X_test, Y_test))
history = model.fit_generator(generator=training_generator,
 validation_data=validation_generator, epochs=nb_epoch)

current_time = datetime.now()
output_path = r"./data/model_output/{0}".format(current_time.strftime('%Y%m%d%H%M%S'))
if not os.path.exists(output_path):
    os.makedirs(output_path)
mutils.save_model_to_json(os.path.join(output_path, "model.json"), os.path.join(output_path, "weights.hd5"), model)
score = model.evaluate_generator(test_generator)
print('Test score:', score[0])
print('Test accuracy:', score[1])
#print('Time:', str(current_time - datetime.now()))
# save plots
plot_utils.plot_model_history_and_save(history, output_path)

# save model history
plot_utils.save_history_for_plot(history, score, model, os.path.join(output_path, 'history.json'))