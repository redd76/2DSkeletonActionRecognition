import numpy as np
from keras.layers import Dense, InputLayer, Dropout
import keras.optimizers

from keras.models import Sequential


class NNModel:
    """
    Simple fully connected Neural Network
    Input images will be flattened to fit input size
    """
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
        model.add(Dropout(.2, input_shape=self.load_inputshape()))
        model.add(Dense(512, activation="relu"))
        model.add(Dense(256, activation="relu"))
        model.add(Dense(128, activation="relu"))
        model.add(Dense(classes, activation="softmax"))
        opt = keras.optimizers.Adam(lr=0.001)
        #opt = keras.optimizers.SGD(lr=0.01, momentum=0.99)
        model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
        return model




