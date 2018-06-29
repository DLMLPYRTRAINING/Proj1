from keras.models import model_from_yaml
import data_gen as dg
import numpy as np
from keras.utils import to_categorical
import os
import matplotlib.pyplot as plt


def model_loader(model_name="", file_path = "Models"):
    if model_name != "":
        # load YAML and create model
        yaml_file = open(file_path + os.sep + model_name + ".yaml", 'r')
        loaded_model_yaml = yaml_file.read()
        yaml_file.close()
        loaded_model = model_from_yaml(loaded_model_yaml)
        # load weights into new model
        loaded_model.load_weights(file_path + os.sep + model_name + ".h5")
        print("Loaded model from disk")

        return loaded_model
        # # evaluate loaded model on test data
        # loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
        # score = loaded_model.evaluate(X, Y, verbose=0)
        # print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))

def run_random_imges():
    train_images, train_labels, test_images, test_labels = dg.load_data(split_ratio=0,total_row=10)

    print('Testing data shape : ', test_images.shape, test_labels.shape)
    print(test_labels)

    # Find the unique numbers from the train labels
    classes = np.unique(test_labels)
    nClasses = len(classes)
    print('Total number of outputs : ', nClasses)
    print('Output classes : ', classes)

    # ==============================================================================
    # Preprocess the Data
    # Find the shape of input images and create the variable input_shape
    nRows, nCols, nDims = test_images.shape[1:]
    test_data = test_images.reshape(test_images.shape[0], nRows, nCols, nDims)

    # Change to float datatype
    test_data = test_data.astype('float32')

    # Scale the data to lie between 0 to 1
    test_data /= 255

    # Change the labels from integer to categorical data
    test_labels_one_hot = to_categorical(test_labels)

    # Display the change for category label using one-hot encoding
    print('Original label 0 : ', test_labels)
    print('After conversion to categorical ( one-hot ) : ', test_labels_one_hot)

    model_name = "Model1_20180629_142228"
    model = model_loader(model_name=model_name)
    # # evaluate loaded model on test data
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    # score = model.evaluate(test_images, test_labels_one_hot, verbose=1)
    # print("%s: %.2f%%" % (model.metrics_names[1], score[1] * 100))

    prediction = model.predict_classes(test_images)
    print("prediction:", prediction)
    print("Actual:    ", test_labels.reshape(1, -1)[0])


    plt.imshow(test_images[0, :, :], cmap='gray')
    plt.title("Ground Truth : {} Prediction: {}".format(test_labels[0], prediction[0]))
    plt.show()
# ------------------------------------------------------------------------------
# main starts from here
run_random_imges()
