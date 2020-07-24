"""
Reference taken from https://engmrk.com/alexnet-implementation-using-keras/
"""

from keras.layers import Dropout, Convolution2D, MaxPooling2D, Flatten, Dense
from keras.models import Sequential


class AlexNetModel:
    """
    Creates a model based on the AlexNet
    Input image will be scaled to fit the input size of AlexNet
    """
    img_rows, img_cols, n_channels = 256, 256, 3

    @staticmethod
    def load_inputshape():
        return AlexNetModel.img_rows, AlexNetModel.img_cols, AlexNetModel.n_channels

    @staticmethod
    def reshape_input_data(x_train, x_test):
        x_train = x_train.reshape(x_train.shape[0], AlexNetModel.img_rows, AlexNetModel.img_cols, AlexNetModel.n_channels)
        x_test = x_test.reshape(x_test.shape[0], AlexNetModel.img_rows, AlexNetModel.img_cols, AlexNetModel.n_channels)
        return x_train, x_test

    @staticmethod
    def load_model(classes=10):
        # TODO build your own model here
        model = Sequential()
        model.add(Convolution2D(96, kernel_size=(11,11), strides=(4,4), activation="relu", input_shape=AlexNetModel.load_inputshape()))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2), padding='valid'))
        model.add(Convolution2D(256, kernel_size=(11, 11), strides=(1,1), activation="relu", padding='valid'))
        model.add(MaxPooling2D(pool_size=(3, 3), strides=(2,2), padding='valid'))
        model.add(Convolution2D(384, kernel_size=(3, 3), strides=(1,1), activation="relu", padding='valid'))
        model.add(Convolution2D(384, kernel_size=(3, 3), strides=(1,1), activation="relu", padding='valid'))
        model.add(Convolution2D(256, kernel_size=(3, 3), strides=(1,1), activation="relu", padding='valid'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2), padding='valid'))
        model.add(Flatten())
        model.add(Dense(4096, activation="relu"))
        model.add(Dropout(0.4))
        model.add(Dense(4096, activation="relu"))
        model.add(Dropout(0.4))
        model.add(Dense(1000, activation="relu"))
        model.add(Dropout(0.4))
        model.add(Dense(classes, activation="softmax"))
        model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
        return model




