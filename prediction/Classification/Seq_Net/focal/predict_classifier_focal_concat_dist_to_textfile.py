from __future__ import print_function

import os, cv2, sys
from keras.applications.inception_v3 import InceptionV3
from keras.applications.imagenet_utils import preprocess_input
from keras.models import Model
from keras.layers import Dense, Flatten, Input, Concatenate
from keras.utils.np_utils import to_categorical
from keras import optimizers
import numpy as np
import glob, os
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.allow_soft_placement = True
set_session(tf.Session(config=config))


IMAGE_FILE_PATH_DISTORTED = ""
path_to_weights = 'weights_06_2.49.h5'

filename_results = os.path.split(path_to_weights)[0]+'/results.txt'

if os.path.exists(filename_results):
    sys.exit("file exists")

classes_focal = list(np.arange(40, 501, 10))
classes_distortion = list(np.arange(0, 61, 1) / 50.)

def get_paths(IMAGE_FILE_PATH_DISTORTED):

    paths_test = glob.glob(IMAGE_FILE_PATH_DISTORTED + 'test/' + "*.jpg")
    paths_test.sort()
    parameters = []
    labels_focal_test = []
    for path in paths_test:
        curr_parameter = float((path.split('_f_'))[1].split('_d_')[0])
        parameters.append(curr_parameter)
        curr_class = classes_focal.index(curr_parameter)
        labels_focal_test.append(curr_class)
    labels_distortion_test = []

    for path in paths_test:
        curr_parameter = float((path.split('_d_'))[1].split('.jpg')[0])
        parameters.append(curr_parameter)
        curr_class = classes_distortion.index(curr_parameter)
        labels_distortion_test.append(curr_class)

    c = list(zip(paths_test, labels_focal_test, labels_distortion_test))

    paths_test, labels_focal_test, labels_distortion_test = zip(*c)
    paths_test, labels_focal_test, labels_distortion_test = list(paths_test), list(labels_focal_test), list(
        labels_distortion_test)
    labels_test = labels_distortion_test
    input_test = [list(a) for a in zip(paths_test, labels_focal_test)]

    return input_test, labels_test

input_test, labels_test = get_paths(IMAGE_FILE_PATH_DISTORTED)

print(len(input_test), 'test samples')

with tf.device('/gpu:1'):
    image_shape = (299, 299, 3)
    image_input = Input(shape=image_shape, dtype='float32', name='main_input')
    input_shape_concat = (len(classes_distortion),)
    concat_input = Input(shape=input_shape_concat, dtype='float32', name='concat_input')
    phi_model = InceptionV3(weights='imagenet', include_top=False, input_tensor=image_input, input_shape=image_shape)
    phi_features = phi_model.output
    phi_flattened = Flatten(name='phi-flattened')(phi_features)
    phi_concat = Concatenate(axis=-1)([phi_flattened,concat_input])
    final_output_focal = Dense(len(classes_focal), activation='softmax', name='output_focal')(phi_concat)

    layer_index = 0
    for layer in phi_model.layers:
        layer.name = layer.name + "_phi"

    model = Model(input=[image_input, concat_input], output=final_output_focal)
    model.load_weights(path_to_weights)

    n_acc_focal = 0
    n_acc_dist = 0
    print(len(input_test))
    file = open(filename_results, 'a')
    for i, curr_input in enumerate(input_test):
        if i % 1000 == 0:
            print(i,' ',len(input_test))
        image = cv2.imread(curr_input[0])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image / 255.
        image = image - 0.5
        image = image * 2.
        image = np.expand_dims(image,0)

        image = preprocess_input(image) 

        # loop
        input_distortion = to_categorical(int(curr_input[1]), len(classes_distortion))
        prediction_focal = model.predict({'main_input_phi':image, 'concat_input':input_distortion.reshape((1, -1))})

        n_acc_focal += classes_focal[np.argmax(prediction_focal[0])]

        file.write(curr_input[0] + '\tlabel_distortion\t' + str(classes_distortion[curr_input[1]]) + '\tlabel_focal\t' + str(classes_focal[labels_test[i]]) + '\tprediction_focal\t' + str(classes_focal[np.argmax(prediction_focal[0])])+'\n')

    print('focal:')
    print(n_acc_focal/len(paths_test))
    file.close()
