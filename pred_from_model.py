from keras.models import model_from_yaml
import object_generator as og
import numpy as np
from keras.utils import to_categorical
import os


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

def run_random_imges(test_images = None, test_labels = None, count = 10):
    if not isinstance(test_images,np.ndarray) or not isinstance(test_labels,np.ndarray):
        test_images, test_labels = og.obj_provider(count=count, data_type="Test")

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

    model_name = "Model1_20180619_143729"
    model = model_loader(model_name=model_name)
    # # evaluate loaded model on test data
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    # score = model.evaluate(test_images, test_labels_one_hot, verbose=1)
    # print("%s: %.2f%%" % (model.metrics_names[1], score[1] * 100))

    prediction = model.predict_classes(test_images)
    print("prediction:", prediction)
    print("Actual:    ", test_labels.reshape(1, -1)[0])

# run a image
def run_perticuler_img():
    file_path = r'C:\Users\slaik\Documents\Sandeep\Scripts\Python\PycharmProjects\imageClassifier\random_test_data'
    # og.obj_pop(count=1, file_path=file_path)
    og.split_objs(ratio=0,file_path=file_path)
    x, y = og.obj_provider(count=1, file_path=file_path, data_type="Test")
    run_random_imges(test_images=x, test_labels=y)
    return
# ------------------------------------------------------------------------------
# main starts from here
# run_random_imges(count=1)
run_perticuler_img()
