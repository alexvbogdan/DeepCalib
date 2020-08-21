from __future__ import division
import sys

import math
import cv2
import numpy as np
import random
import tensorflow as tf
from PIL import Image, ImageOps
import keras
from keras.preprocessing.image import Iterator
from keras.utils.np_utils import to_categorical
import keras.backend as K
import multiprocessing
IMAGE_WIDTH = 299
IMAGE_HEIGHT = 299
PI = math.pi
#Create Dictionaries
ROTATE = {}
ANGLE = {}
SLICE = {}

class CustomModelCheckpoint(keras.callbacks.Callback):

    def __init__(self, model_for_saving, filepath, monitor='val_loss', verbose=0,
                 save_best_only=False, save_weights_only=False,
                 mode='auto', period=1):
        super(CustomModelCheckpoint, self).__init__()
        self.monitor = monitor
        self.verbose = verbose
        self.filepath = filepath
        self.save_best_only = save_best_only
        self.save_weights_only = save_weights_only
        self.period = period
        self.epochs_since_last_save = 0
        self.model_for_saving = model_for_saving

        if mode not in ['auto', 'min', 'max']:
            warnings.warn('ModelCheckpoint mode %s is unknown, '
                          'fallback to auto mode.' % (mode),
                          RuntimeWarning)
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.Inf
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.Inf
        else:
            if 'acc' in self.monitor or self.monitor.startswith('fmeasure'):
                self.monitor_op = np.greater
                self.best = -np.Inf
            else:
                self.monitor_op = np.less
                self.best = np.Inf

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epochs_since_last_save += 1
        if self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0
            filepath = self.filepath.format(epoch=epoch + 1, **logs)
            if self.save_best_only:
                current = logs.get(self.monitor)
                if current is None:
                    warnings.warn('Can save best model only with %s available, '
                                  'skipping.' % (self.monitor), RuntimeWarning)
                else:
                    if self.monitor_op(current, self.best):
                        if self.verbose > 0:
                            print('\nEpoch %05d: %s improved from %0.5f to %0.5f,'
                                  ' saving model to %s'
                                  % (epoch + 1, self.monitor, self.best,
                                     current, filepath))
                        self.best = current
                        if self.save_weights_only:
                            self.model_for_saving.save_weights(filepath, overwrite=True)
                        else:
                            self.model_for_saving.save(filepath, overwrite=True)
                    else:
                        if self.verbose > 0:
                            print('\nEpoch %05d: %s did not improve' %
                                  (epoch + 1, self.monitor))
            else:
                if self.verbose > 0:
                    print('\nEpoch %05d: saving model to %s' % (epoch + 1, filepath))
                if self.save_weights_only:
                    self.model_for_saving.save_weights(filepath, overwrite=True)
                else:
                    self.model_for_saving.save(filepath, overwrite=True)



def angle_difference(x, y):
    """
    Calculate minimum difference between two angles.

    """

    return 180 - abs(abs(x-y) - 180)

def angle_error(y_true, y_pred):
    """
    Calculate the mean diference between the
    true angles
    and the predicted angles. Each angle is represented
    as a float number between 0 and 1.
    """
    diff = angle_difference(K.argmax(y_true), K.argmax(y_pred))
    return K.mean(K.cast(K.abs(diff), K.floatx()))

def slice_image(image, slice_dict):

    if slice_dict == 0:
        sliced_image = image

    else:
        map_coordinate = SLICE[slice_dict]
        map_x = map_coordinate[0]
        map_y = map_coordinate[1]
        # np.save(PROJECT_FILE_PATH+"x_generated_sliced.npy", map_x)
        # np.save(PROJECT_FILE_PATH+"y_generated_sliced.npy", map_y)
        sliced_image=cv2.remap(image, map_x, map_y, cv2.INTER_NEAREST)

    return sliced_image


def add_noise(image):
    image = np.clip(image,0,255)
    if bool(random.getrandbits(1)):
        total = IMAGE_WIDTH * IMAGE_HEIGHT * 3
        a = np.random.randint(-3,4, size=total)
        a = a.reshape(IMAGE_HEIGHT,IMAGE_WIDTH,3)
        noise = image + a
        to_int_noise = noise.astype('uint8')
    else:
        row,col,ch= image.shape
        mean = 0
        var = 0.1
        sigma = var**0.5
        gauss = np.random.normal(mean,sigma,(row,col,ch))
        gauss = gauss.reshape(row,col,ch)
        noisy = image + gauss
        to_int_noise = noisy.astype('uint8')
    return to_int_noise

def add_gaussian_noise(image):
    image = np.clip(image, 0, 255)
    row,col,ch= image.shape
    mean = 0
    var = 0.1
    sigma = var**0.5
    gauss = np.random.normal(mean,sigma,(row,col,ch))
    gauss = gauss.reshape(row,col,ch)
    noisy = image + gauss
    to_int_noisy = noisy.astype('uint8')
    return to_int_noisy

def add_contrast_brightness(image):
    image = np.clip(image,0,255)
    contrast = np.random.uniform(0.9,1.1)
    brightness = np.random.randint(-5,5)
    adjusted_image = image * contrast + brightness
    to_int_adjusted = adjusted_image.astype('uint8')
    return to_int_adjusted

def blur_randomly(image):
    image = np.clip(image, 0, 255)
    std = np.random.uniform(0, 1.5)
    blurred = cv2.GaussianBlur(image, (5,5), std)
    return blurred

def erase_randomly(image):
    area = random.uniform(0.1, 0.6)
    prob = random.getrandbits(1)
    if prob:
        image = Image.fromarray(image)
        w, h = IMAGE_WIDTH, IMAGE_HEIGHT

        w_occlusion_max = int(w * area)
        h_occlusion_max = int(h * area)

        w_occlusion_min = int(w * 0.1)
        h_occlusion_min = int(h * 0.1)

        w_occlusion = random.randint(w_occlusion_min, w_occlusion_max)
        h_occlusion = random.randint(h_occlusion_min, h_occlusion_max)

        if len(image.getbands()) == 1:
            rectangle = Image.fromarray(np.uint8(np.random.rand(w_occlusion, h_occlusion) * 255))
        else:
            rectangle = Image.fromarray(np.uint8(np.random.rand(w_occlusion, h_occlusion, len(image.getbands())) * 255))

        random_position_x = random.randint(0, w - w_occlusion)
        random_position_y = random.randint(0, h - h_occlusion)

        image.paste(rectangle, (random_position_x, random_position_y))
        image = np.array(image)

    return image


def crop_image(image):
    min_y = np.random.randint(3)
    max_y = np.random.randint(3)
    min_x = np.random.randint(3)
    max_x = np.random.randint(3)
    image = image[min_y:299-max_y, min_x:299-max_x]
    if min_x!=0 or max_x!=0 or min_y!=0 or max_y!=0:
        image = cv2.resize(image, (299,299))
    return image

def rotate_image(image, rotation_number):
    """
    Rotate an OpenCV 2 / NumPy image around it's centre by the given angle
    (in degrees). The returned image will have the same size as the new image.

    Adapted from: http://stackoverflow.com/questions/16702966/rotate-image-and-crop-out-black-borders
    """

    # Get the image size
    # No that's not an error - NumPy stores image matricies backwards
    map_coordinate = ROTATE[rotation_number]
    map_x = map_coordinate[0]
    map_y = map_coordinate[1]
    rotate_coin = np.random.randint(3)
    if rotate_coin == 0:
        rotated_image=cv2.remap(image, map_x, map_y, cv2.INTER_NEAREST)
    elif rotate_coin == 1:
        rotated_image=cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR)
    else:
        rotated_image=cv2.remap(image, map_x, map_y, cv2.INTER_CUBIC)
    return rotated_image

def give_random_angle_phi():
    A = np.arange(1,179)
    return random.choice(A)


def generate_rotated_sliced_image(image, rotation_number, sliced_dict, size=None, crop_center=False,
                                    crop_largest_rect=False):
    """
    Generate a valid rotated image for the RotNetDataGenerator. If the
    image is rectangular, the crop_center option should be used to make
    it square. To crop out the black borders after rotation, use the
    crop_largest_rect option. To resize the final image, use the size
    option.
    """

    sliced_image = slice_image(image, sliced_dict)
    rotated_image = rotate_image(sliced_image, rotation_number)

    return rotated_image


class RotNetDataGenerator(object):
    def __init__(self, input_shape=None, color_mode='rgb', batch_size=64, one_hot=True,
                preprocess_func=None, rotate=True, sliced=True, flip=True, crop_center=False, crop_largest_rect=False, contrast_and_brightness=True,
                shuffle=False, seed=None):

        self.input_shape = input_shape
        self.color_mode = color_mode
        self.batch_size = batch_size
        self.one_hot = one_hot
        self.preprocess_func = preprocess_func
        self.rotate = rotate
        self.crop_center = crop_center
        self.crop_largest_rect = crop_largest_rect
        self.shuffle = shuffle
        self.sliced = sliced
        self.flip = flip
        self.contrast_and_brightness = contrast_and_brightness

    #@threadsafe_generator
    def generate(self, image_path,labels,num_classes_focal,num_classes_distortion):
        # Infinite loop
        while 1:
            # Generate order of exploration of dataset
            if self.shuffle:
                c = list(zip(image_path, labels))
                random.shuffle(c)
                image_path, labels = zip(*c)
            # Generate batches
            imax = int(len(image_path)/self.batch_size)
            for i in range(imax):
                start_index = i * self.batch_size
                end_index = (i+1) * self.batch_size
                image_path_temp = image_path[start_index:end_index]
                labels_temp = labels[start_index:end_index]
                # Generate data
                # try:
                X, y = self.__data_generation(image_path_temp,labels_temp,num_classes_focal,num_classes_distortion)

                yield X, y
                # except:
                #     print 'failed ',image_path_temp
                #     continue



    def __data_generation(self, image_path_temp,labels,num_classes_focal,num_classes_distortion):
        # create array to hold the images
        batch_x = np.zeros((self.batch_size,) + self.input_shape, dtype='float32')
        # create array to hold the labels
        # batch_label_focal = []
        # batch_label_distortion = []
        batch_label_focal = []
        batch_label_distortion = []
        # iterate through the current batch
        for index, current_path in enumerate(image_path_temp):
            is_color = int(self.color_mode == 'rgb')
            image = cv2.imread(current_path, is_color)
            if is_color:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            if bool(random.getrandbits(1)):
                image = add_noise(image)
            if bool(random.getrandbits(1)):
                image = add_contrast_brightness(image)
            if self.flip:
                if bool(random.getrandbits(1)):
                    image = cv2.flip(image, 1)

            image = image / 255.
            image = image - 0.5
            image = image * 2.
            batch_x[index] = image
            # if self.one_hot:
            label_focal = to_categorical(int(labels[index][0]), num_classes_focal)
            label_distortion = to_categorical(int(labels[index][1]),num_classes_distortion)
            batch_label_focal.append(label_focal)
            batch_label_distortion.append(label_distortion)
        batch_label_focal = np.array(batch_label_focal)
        batch_label_distortion = np.array(batch_label_distortion)
        # preprocess input images
        if self.preprocess_func:
            batch_x = self.preprocess_func(batch_x)

        return batch_x, {'output_focal': batch_label_focal, 'output_distortion': batch_label_distortion}



