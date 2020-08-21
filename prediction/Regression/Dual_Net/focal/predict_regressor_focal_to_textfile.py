from __future__ import print_function

import os, cv2, sys
from keras.applications.inception_v3 import InceptionV3
from keras.applications.imagenet_utils import preprocess_input
from keras.models import Model
from keras.layers import Dense, Flatten, Input
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
path_to_weights = 'weights_06_0.02.h5'

filename_results = os.path.split(path_to_weights)[0]+'/results.txt'

if os.path.exists(filename_results):
    sys.exit("file exists")

focal_start = 80
focal_end = 300
classes_focal = list(np.arange(focal_start, focal_end+1, 10))
classes_distortion = list(np.arange(0, 41, 1) / 40.)

def get_paths(IMAGE_FILE_PATH_DISTORTED):

    paths_test = glob.glob(IMAGE_FILE_PATH_DISTORTED + 'test/' + "*.jpg")
    paths_test.sort()
    parameters = []
    labels_focal_test = []
    for path in paths_test:
        curr_parameter = float((path.split('_f_'))[1].split('_d_')[0])
        labels_focal_test.append((curr_parameter - focal_start*1.) / (focal_end+1. - focal_start*1.)) #normalize bewteen 0 and 1
    labels_distortion_test = []
    for path in paths_test:
        curr_parameter = float((path.split('_d_'))[1].split('.jpg')[0])
        labels_distortion_test.append(curr_parameter*1.2)

    c = list(zip(paths_test, labels_focal_test, labels_distortion_test))
    paths_test, labels_focal_test, labels_distortion_test = zip(*c)
    paths_test, labels_focal_test, labels_distortion_test = list(paths_test), list(labels_focal_test), list(
        labels_distortion_test)
    labels_test = [list(a) for a in zip(labels_focal_test, labels_distortion_test)]

    return paths_test, labels_test

paths_test, labels_test = get_paths(IMAGE_FILE_PATH_DISTORTED)

print(len(paths_test), 'test samples')

with tf.device('/gpu:0'):
    input_shape = (299, 299, 3)
    main_input = Input(shape=input_shape, dtype='float32', name='main_input')
    phi_model = InceptionV3(weights='imagenet', include_top=False, input_tensor=main_input, input_shape=input_shape)
    phi_features = phi_model.output
    phi_flattened = Flatten(name='phi-flattened')(phi_features)
    final_output_focal = Dense(1, activation='sigmoid', name='output_focal')(phi_flattened)

    layer_index = 0
    for layer in phi_model.layers:
        layer.name = layer.name + "_phi"

    model = Model(input=main_input, output=final_output_focal)
    model.load_weights(path_to_weights)

    n_acc_focal = 0
    n_acc_dist = 0
    print(len(paths_test))
    file = open(filename_results, 'a')
    for i, path in enumerate(paths_test):
        if i % 1000 == 0:
            print(i,' ',len(paths_test))
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image / 255.
        image = image - 0.5
        image = image * 2.
        image = np.expand_dims(image,0)

        image = preprocess_input(image)

        # loop
        prediction_focal = model.predict(image)[0]
        prediction_dist = model.predict(image)[1]

        if np.argmax(prediction_focal[0]) == labels_test[i][0]:
            n_acc_focal = n_acc_focal + 1
        if np.argmax(prediction_dist[0]) == labels_test[i][1]:
            n_acc_dist = n_acc_dist + 1

        curr_focal_label = labels_test[i][0] * (focal_end+1. - focal_start*1.) + focal_start*1.
        curr_focal_pred = prediction_focal[0][0] * (focal_end+1. - focal_start*1.) + focal_start*1.
        curr_dist_label = labels_test[i][1]*1.2
        curr_dist_pred = prediction_dist[0][0]*1.2
        file.write(path + '\tlabel_focal\t' + str(curr_focal_label) + '\tprediction_focal\t' + str(curr_focal_pred) + '\tlabel_dist\t' + str(curr_dist_label) + '\tprediction_dist\t' + str(curr_dist_pred)+'\n')

    print('focal:')
    print(n_acc_focal)
    print(len(paths_test))
    print(n_acc_focal*1.0/(len(paths_test)*1.0))

    print('dist:')
    print(n_acc_dist)
    print(len(paths_test))
    print(n_acc_dist * 1.0 / (len(paths_test) * 1.0))
    file.close()
