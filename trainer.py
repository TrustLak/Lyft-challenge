# Lyft Challenge main code.

## Include stuff:
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tensorflow as tf
from keras.optimizers import Adam, SGD, RMSprop
import json
from utils import *
import glob
from random import shuffle
from PIL import Image

## Define hyper parameters for training
MINIBATCH_SIZE  = 8 
NUM_EPOCHS      = 100
LEARNING_RATE   = 0.0001 # for adam optimizer
VALID_SPLIT     = 0.2

## Image properties to be used by our model
IMAGE_WIDTH     = 512# 800/2
IMAGE_HIGHT     = 400# 600/2
DEPTH           = 6
in_shape        = [IMAGE_HIGHT, IMAGE_WIDTH, DEPTH]
img_w = IMAGE_WIDTH
img_h = IMAGE_HIGHT
img_d = DEPTH

## Training set properties:
N_LABELS        = 4
X_train_batch   = []#np.zeros((100, IMAGE_WIDTH, IMAGE_HIGHT, DEPTH))
y_train_batch   = []#np.zeros((100, IMAGE_WIDTH, IMAGE_HIGHT, N_LABELS))

PIXEL_LABEL_SHAPE = np.zeros((1, N_LABELS))

## Read dataset
rgb_train = []
label_train = []
rgb_valid = []
label_valid = []
rgb_test = []
label_test = []

# read data from original posting
rgb_train += glob.glob('Train/**/CameraRGB/*.png')
label_train += glob.glob('Train/**/CameraSeg/*.png')

# read data from Train100 folder
rgb_train += glob.glob('Train100/**/CameraRGB/*.png')
label_train += glob.glob('Train100/**/CameraSeg/*.png')

# read data from Train80 folder
rgb_train += glob.glob('Train80/**/CameraRGB/*.png')
label_train += glob.glob('Train80/**/CameraSeg/*.png')


# read data from world_2_little_cars folder
rgb_train += glob.glob('world_2_little_cars/**/CameraRGB/*.png')
label_train += glob.glob('world_2_little_cars/**/CameraSeg/*.png')

# read data from world2_cars folder
rgb_train += glob.glob('world2_cars/CameraRGB/*.png')
label_train += glob.glob('world2_cars/SegCamera/*.png')

# read data from Train120
rgb_train += glob.glob('Train120/Train/CameraRGB/*.png')
label_train += glob.glob('Train120/Train/CameraSeg/*.png')

rgb_valid += glob.glob('Train120/Valid/CameraRGB/*.png')
label_valid += glob.glob('Train120/Valid/CameraSeg/*.png')

rgb_test += glob.glob('Train120/Test/CameraRGB/*.png')
label_test += glob.glob('Train120/Test/CameraSeg/*.png')

# combine all data
rgb_total = rgb_train + rgb_test + rgb_valid
label_total = label_train + label_test + label_valid

# shuffle entire dataset
names_zip = list(zip(rgb_total, label_total))
shuffle(names_zip)
rgb_total, label_total = zip(*names_zip)
#split: 
rgb_train = rgb_total[0:int(0.9*len(rgb_total))]
label_train = label_total[0:int(0.9*len(label_total))]
rgb_valid= rgb_total[int(0.9*len(rgb_total)):]
label_valid = label_total[int(0.9*len(label_total)):]

print("Number of training examples: ", len(rgb_train))
print("Number of validation examples: ", len(rgb_valid))
print("Number of test examples: ", len(rgb_test))

VALID_STEPS     = int(len(rgb_valid)/MINIBATCH_SIZE)
STEPS_PER_EPOCH = int(len(rgb_train)/MINIBATCH_SIZE)


def trainBatchGenerator(train_list, label_list, valid):
    while 1:
        for k in range(len(train_list)):
            # Reset batch to empty lists:
            X_train_batch   = []
            y_train_batch   = []
            for j in range(MINIBATCH_SIZE):
                # Add training exmaple
                X_image = np.asarray(Image.open(train_list[k]))
                y_image = np.asarray(Image.open(label_list[k]))
                y_label = label_map(y_image[:,:,0])
                RGB_stack, label_stack = process_img(X_image, y_label)
                
                #print(preprocessed_img.shape)
                X_train_batch.append(RGB_stack)
                y_train_batch.append(label_stack)
            # finished a batch.
            X_train = np.array(X_train_batch)
            y_train = np.array(y_train_batch)

            yield X_train, y_train


## initialize network
print("Building model: ")
model = build_model()

opt = Adam(lr=LEARNING_RATE)
model.compile(loss='binary_crossentropy', optimizer=opt,  metrics=['accuracy'])

print("Training: ")
for ep in range(NUM_EPOCHS):
    model.fit_generator(trainBatchGenerator(rgb_train, label_train, 0), steps_per_epoch = STEPS_PER_EPOCH, epochs = 1,\
     verbose = 1, validation_data=trainBatchGenerator(rgb_valid, label_valid, 1), validation_steps=VALID_STEPS)

    # keep models after each episode to compare performance and to check for overfitting
    print("Episode number: ", ep)
    model_json = model.to_json()
    model_file_name = 'model__'+str(ep)+'.json'
    with open(model_file_name, "w") as outfile:
        outfile.write(model_json)
    model_weight_name = 'model__'+str(ep)+'.h5'
    model.save_weights(model_weight_name, overwrite=True)

## save final model
model_json = model.to_json()
with open("model.json", "w") as outfile:
    outfile.write(model_json)
model.save_weights("model.h5", overwrite=True)