from matplotlib import pyplot as plt
import numpy as np
import os
import json
import io

def get_model_summary(model):
    stream = io.StringIO()
    model.summary(print_fn=lambda x: stream.write(x + '\n'))
    summary_string = stream.getvalue()
    stream.close()
    return summary_string

def save_history_for_plot(history, score, model, output_path):
    json_data = {}
    json_data['accuracy'] = [float(val) for val in history.history['accuracy']]
    json_data['val_accuracy'] = [float(val) for val in history.history['val_accuracy']]

    json_data['loss'] = [float(val) for val in history.history['loss']]
    json_data['val_loss'] = [float(val) for val in history.history['val_loss']]

    json_data['score'] = [float(val) for val in score]
    json_data['model_summary'] = get_model_summary(model)

    with open(output_path, "w") as jfile:
        json.dump(json_data, jfile)

def plot_model_history(history):
    # list all data in history
    print(history.history.keys())
    # summarize history for accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

def plot_model_history_and_save(history, h_path):
    # list all data in history
    print(history.history.keys())
    # summarize history for accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(os.path.join(h_path, "modelAccuracy.png"))
    plt.clf()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(os.path.join(h_path, "modelLoss.png"))
    plt.clf()



def plot_result_examples(model, X_test, y_test, img_rows, img_cols):
    """
    The predict_classes function outputs the highest probability class
    according to the trained classifier for each input example.
    :return:
    """
    predicted_classes = model.predict_classes(X_test)

    # Check which items we got right / wrong
    correct_indices = np.nonzero(predicted_classes == y_test)[0]
    incorrect_indices = np.nonzero(predicted_classes != y_test)[0]

    plt.figure()
    for i, correct in enumerate(correct_indices[:9]):
        plt.subplot(3, 3, i + 1)
        plt.imshow(X_test[correct].reshape(img_rows, img_cols), cmap='gray', interpolation='none')
        plt.title("Predicted {}, Class {}".format(predicted_classes[correct], y_test[correct]))

    plt.figure()
    for i, incorrect in enumerate(incorrect_indices[:9]):
        plt.subplot(3, 3, i + 1)
        plt.imshow(X_test[incorrect].reshape(img_rows, img_cols), cmap='gray', interpolation='none')
        plt.title("Predicted {}, Class {}".format(predicted_classes[incorrect], y_test[incorrect]))

    plt.show()