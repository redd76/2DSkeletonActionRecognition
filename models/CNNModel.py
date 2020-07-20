from keras.models import Sequential, Model
from keras.layers import Dropout, Convolution2D, MaxPooling2D, Flatten, Dense, Activation
from keras.layers import Input, Conv2D, BatchNormalization, LeakyReLU, MaxPool2D, Permute, concatenate


class CNNModel:
    img_rows, img_cols, n_channels = 60, 25, 3

    def __init__(self, img_rows, img_cols):
        self.img_rows = img_rows
        self.img_cols = img_cols

    def load_inputshape(self):
        return self.img_rows, self.img_cols, self.n_channels

    def reshape_input_data(self, x_train, x_test):
        x_train = x_train.reshape(x_train.shape[0], self.img_rows, self.img_cols, self.n_channels)
        x_test = x_test.reshape(x_test.shape[0], self.img_rows, self.img_cols, self.n_channels)
        return x_train, x_test

    def load_model(self, classes=10):
        # TODO build your own model here
        model = Sequential()
        model.add(Convolution2D(64, kernel_size=(3,3), activation="relu", input_shape=self.load_inputshape()))
        model.add(Convolution2D(64, kernel_size=(3, 3), activation="relu"))
        model.add(Flatten())
        model.add(Dense(64, activation="relu"))
        model.add(Dense(64, activation="relu"))
        model.add(Dense(classes, activation="softmax"))
        model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
        return model




