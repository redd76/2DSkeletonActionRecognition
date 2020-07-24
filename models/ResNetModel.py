from keras.layers import Dropout, Convolution2D, MaxPooling2D, Flatten, Dense, BatchNormalization, UpSampling2D
from keras.models import Sequential
from keras.applications.vgg16 import VGG16
from keras.applications.resnet_v2 import ResNet50V2
from keras.models import Model


class ResNetModel:
    """
    ResNet Model with a top to classify
    Will be fully fine tuned based on the new Data
    """
    img_rows, img_cols, n_channels = 224, 224, 3

    @staticmethod
    def load_inputshape():
        return ResNetModel.img_rows, ResNetModel.img_cols, ResNetModel.n_channels

    @staticmethod
    def reshape_input_data(x_train, x_test):
        x_train = x_train.reshape(x_train.shape[0], ResNetModel.img_rows, ResNetModel.img_cols, ResNetModel.n_channels)
        x_test = x_test.reshape(x_test.shape[0], ResNetModel.img_rows, ResNetModel.img_cols, ResNetModel.n_channels)
        return x_train, x_test

    @staticmethod
    def load_model(classes=10):
        # TODO build your own model here
        model = Sequential()
        resnet = ResNet50V2(include_top=False, input_shape=ResNetModel.load_inputshape())
        #vgg.get_layer('block1_conv1').trainable = False
        #vgg.get_layer('block1_conv2').trainable = False
        #vgg.get_layer('block2_conv1').trainable = False
        for layer in enumerate(resnet.layers):
            if isinstance(layer, BatchNormalization):
                layer.trainable = True
            else:
                layer.trainable = False
        #model.add(UpSampling2D)
        model.add(resnet)
        model.add(Flatten())
        model.add(Dense(64, activation="relu"))
        model.add(Dense(64, activation="relu"))
        model.add(Dense(classes, activation="softmax"))
        model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
        return model




