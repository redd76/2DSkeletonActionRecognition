from keras.models import Sequential, Model
from keras.layers import Dropout, Convolution2D, MaxPooling2D, Flatten, Dense, Activation
from keras.layers import Conv2D, BatchNormalization, LeakyReLU, MaxPool2D, Permute, concatenate
from keras.layers import InputLayer

import numpy as np


class NNModel:
    input_layer = 60 * 25 * 3

    def __init__(self, dim):
        self.input_layer = np.prod(dim)
        print(self.input_layer)

    def load_inputshape(self):
        return (self.input_layer,)

    def reshape_input_data(self, x_train, x_test):
        x_train = x_train.reshape(x_train.shape[0], self.input_layer)
        x_test = x_test.reshape(x_test.shape[0], self.input_layer)
        return x_train, x_test

    def load_model(self, classes=10):
        # TODO build your own model here
        model = Sequential()
        model.add(InputLayer(input_shape=self.load_inputshape()))
        model.add(Dense(128, activation="relu"))
        model.add(Dense(64, activation="relu"))
        model.add(Dense(64, activation="relu"))
        model.add(Dense(classes, activation="softmax"))
        model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
        return model




