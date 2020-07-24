import json
from keras.models import model_from_json


def save_model_to_json(file_path, weight_path, model):
    # serialize model to JSON
    model_json = model.to_json()
    with open(file_path, "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(weight_path)

def load_model_from_json(file_path, weight_path):
    with open(file_path, "r") as jfile:
        json_dat = jfile.read()
    print(json_dat)
    model = model_from_json(json_dat)
    model.load_weights(weight_path)
    return model