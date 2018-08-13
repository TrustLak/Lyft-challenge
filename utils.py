import tensorflow as tf
import numpy as np
import keras
from keras.models import Sequential, Model
from keras import initializers
from keras.layers.core import Activation, Reshape, Permute, Lambda
from keras.layers.convolutional import Convolution2D, MaxPooling2D, UpSampling2D
from keras.layers.normalization import BatchNormalization
from keras.layers import Input, concatenate, Concatenate, merge, Merge
import keras.backend as K
import json
import cv2

from PIL import Image
from utils import *
import cv2
import matplotlib.pyplot as plt

IMAGE_WIDTH     = 512# 800/2
IMAGE_HIGHT     = 400# 600/2
img_w = IMAGE_WIDTH
img_h = IMAGE_HIGHT
img_d = 6
n_labels = 4
in_shape = (img_h, img_w, img_d)

f_w = 3
weight_initializer = initializers.TruncatedNormal(mean=0.0, stddev=0.01)
bias_initializer   = initializers.RandomUniform(minval=-0.0003, maxval=0.0003, seed=None)

def build_model():
    encoder_1_1  = Input(shape=in_shape)
    norm         = BatchNormalization()(encoder_1_1)
    encoder_1_2  = Convolution2D(24, 3, 3, kernel_initializer=weight_initializer, bias_initializer=bias_initializer, border_mode='same')(norm)
    encoder_1_3  = BatchNormalization()(encoder_1_2)
    encoder_1_7  = Activation("relu")(encoder_1_3)
    encoder_1_21 = Convolution2D(24, 3, 3, kernel_initializer=weight_initializer, bias_initializer=bias_initializer, border_mode='same')(encoder_1_7)
    encoder_1_31 = BatchNormalization()(encoder_1_21)
    encoder_1_71 = Activation("relu")(encoder_1_31)

    encoder_2_1  = MaxPooling2D(padding='same')(encoder_1_71)
    encoder_2_2  = Convolution2D(32, 3, 3, kernel_initializer=weight_initializer, bias_initializer=bias_initializer, border_mode='same')(encoder_2_1)
    encoder_2_3  = BatchNormalization()(encoder_2_2)
    encoder_2_7  = Activation("relu")(encoder_2_3)
    encoder_2_21 = Convolution2D(32, f_w, f_w, kernel_initializer=weight_initializer, bias_initializer=bias_initializer, border_mode='same')(encoder_2_7)
    encoder_2_31 = BatchNormalization()(encoder_2_21)
    encoder_2_71 = Activation("relu")(encoder_2_31)

    encoder_3_1  = MaxPooling2D(padding='same')(encoder_2_71)
    encoder_3_2  = Convolution2D(64, f_w, f_w, kernel_initializer=weight_initializer, bias_initializer=bias_initializer, border_mode='same')(encoder_3_1)
    encoder_3_3  = BatchNormalization()(encoder_3_2)
    encoder_3_7  = Activation("relu")(encoder_3_3)
    encoder_3_21 = Convolution2D(64, f_w, f_w, kernel_initializer=weight_initializer, bias_initializer=bias_initializer, border_mode='same')(encoder_3_7)
    encoder_3_31 = BatchNormalization()(encoder_3_21)
    encoder_3_71 = Activation("relu")(encoder_3_31)

    mid_1_1 = MaxPooling2D(padding='same')(encoder_3_71)
    mid_1_2 = Convolution2D(256, f_w, f_w, kernel_initializer=weight_initializer, bias_initializer=bias_initializer, border_mode='same')(mid_1_1)
    mid_1_3 = BatchNormalization()(mid_1_2)
    mid_1_4 = Activation("relu")(mid_1_3)

    mid_2_2 = Convolution2D(256, f_w, f_w, kernel_initializer=weight_initializer, bias_initializer=bias_initializer, border_mode='same')(mid_1_4)
    mid_2_3 = BatchNormalization()(mid_2_2)
    mid_2_4 = Activation("relu")(mid_2_3)

    # mid_3_2 = Convolution2D(512, f_w, f_w, kernel_initializer=weight_initializer, bias_initializer=bias_initializer, border_mode='same')(mid_2_4)
    # mid_3_3 = BatchNormalization()(mid_3_2)
    # mid_3_4 = Activation("relu")(mid_3_3)

    # mid_4_2 = Convolution2D(256, f_w, f_w, kernel_initializer=weight_initializer, bias_initializer=bias_initializer, border_mode='same')(mid_2_4)
    # mid_4_3 = BatchNormalization()(mid_4_2)
    # mid_4_4 = Activation("relu")(mid_4_3)

    mid_5_2 = Convolution2D(256, f_w, f_w, kernel_initializer=weight_initializer, bias_initializer=bias_initializer, border_mode='same')(mid_2_4)
    mid_5_3 = BatchNormalization()(mid_5_2)
    mid_5_4 = Activation("relu")(mid_5_3)
    mid_5_5 =UpSampling2D(size=(2, 2))(mid_5_4)

    merged_1     = keras.layers.concatenate([mid_5_5, encoder_3_71], axis=3)
    decoder_1_2  = Convolution2D(64, f_w, f_w, kernel_initializer=weight_initializer, bias_initializer=bias_initializer, border_mode='same')(merged_1)
    decoder_1_3  = BatchNormalization()(decoder_1_2)
    decoder_1_4  = Activation("relu")(decoder_1_3)
    decoder_1_21 = Convolution2D(64, f_w, f_w, kernel_initializer=weight_initializer, bias_initializer=bias_initializer, border_mode='same')(decoder_1_4)
    decoder_1_31 = BatchNormalization()(decoder_1_21)
    decoder_1_41 = Activation("relu")(decoder_1_31)
    decoder_1_8  = UpSampling2D(size=(2, 2))(decoder_1_41)

    merged_2     = keras.layers.concatenate([decoder_1_8, encoder_2_71], axis=3)
    decoder_2_2  = Convolution2D(32, f_w, f_w, kernel_initializer=weight_initializer, bias_initializer=bias_initializer, border_mode='same')(merged_2)
    decoder_2_3  = BatchNormalization()(decoder_2_2)
    decoder_2_4  = Activation("relu")(decoder_2_3)
    decoder_2_21 = Convolution2D(32, f_w, f_w, kernel_initializer=weight_initializer, bias_initializer=bias_initializer, border_mode='same')(decoder_2_4)
    decoder_2_31 = BatchNormalization()(decoder_2_21)
    decoder_2_41 = Activation("relu")(decoder_2_31)
    decoder_2_8  = UpSampling2D(size=(2, 2))(decoder_2_41)

    merged_3     = keras.layers.concatenate([decoder_2_8, encoder_1_71], axis=3)
    decoder_3_2  = Convolution2D(24, 3, 3, kernel_initializer=weight_initializer, bias_initializer=bias_initializer, border_mode='same')(merged_3)
    decoder_3_3  = BatchNormalization()(decoder_3_2)
    decoder_3_4  = Activation("relu")(decoder_3_3)
    decoder_3_21 = Convolution2D(24, 3, 3, kernel_initializer=weight_initializer, bias_initializer=bias_initializer, border_mode='same')(decoder_3_4)
    decoder_3_31 = BatchNormalization()(decoder_3_21)
    decoder_3_41 = Activation("relu")(decoder_3_31)

    decoder_3_8 = Convolution2D(n_labels, 3, 3, border_mode='same')(decoder_3_41)
    decoder_3_9  = BatchNormalization()(decoder_3_8)
    decoder_3_10 = Activation("sigmoid")(decoder_3_9)

    # model = Model(input = encoder_1.input, output= decoder_3.output)
    model = Model(inputs=encoder_1_1, outputs=decoder_3_10)
    # print model summary to check dimensions
    model.summary()
    # model.summary()
    return model


def label_map(labels):
    # convert original label to 2 layers for vehicles and roads
    label = np.zeros([600, 800, 2]) 

    vehicles = np.where((labels==10),1,0).astype('uint8')
    label[:,:,0] = vehicles*1.0# 10

    lane = np.where((labels==6),1,0).astype('uint8')
    lane_lines = np.where((labels==7),1,0).astype('uint8')
    label[:,:,1] = (lane | lane_lines)*1.0 

    return label


def process_img(img, label):
    # image size is resized to 512, 512
    rgb_stack = np.zeros((img_h,img_w,6)).astype(np.uint8)
    label_stack = np.zeros((img_h,img_w,4)).astype(np.uint8)
    rgb_img = np.copy(img)
    y_label = np.copy(label)

    # brightness:
    if np.random.rand() > 0.7:
        rgb_img = randomBrightness(rgb_img)
 
    if np.random.rand() > 0.7:
        rgb_img = randomHueSaturationValue(rgb_img)

    cropped_img = rgb_img[218:530,:,:]
    cropped_label = y_label[218:530,:,:]

    center_rgb = rgb_img[262:408,150:550]
    y_label_center = y_label[262:408,150:550] 

    resized_rgb = cv2.resize(cropped_img, (512, 400))
    resized_center_rgb = cv2.resize(center_rgb, (512, 400))

    resized_y_label= cv2.resize(cropped_label, (512, 400))
    resized_y_center_label= cv2.resize(y_label_center, (512, 400))

    if np.random.rand() > 0.5:
        resized_rgb = np.flip( resized_rgb, 1)
        resized_y_label = np.flip( resized_y_label, 1)
        resized_center_rgb = np.flip( resized_center_rgb, 1)
        resized_y_center_label = np.flip( resized_y_center_label, 1)
# 
    if np.random.rand() > 0.8:
        angle = np.random.uniform(30)-30/2
        Rot_M = cv2.getRotationMatrix2D((256,200),angle,1)
        resized_rgb = cv2.warpAffine(resized_rgb,Rot_M,(512,400))
        resized_y_label = cv2.warpAffine(resized_y_label,Rot_M,(512,400))
        resized_center_rgb = cv2.warpAffine(resized_center_rgb,Rot_M,(512,400))
        resized_y_center_label = cv2.warpAffine(resized_y_center_label,Rot_M,(512,400))


    rgb_stack[:,:,0:3] = resized_rgb# [:, x_mid:x_end,0:3]
    rgb_stack[:,:,3:6] = resized_center_rgb# [y_start:y_mid, x_start:x_mid,0:3]

    # print(label.shape)
    label_stack[:,:,0:2]= resized_y_label#[y_start:y_mid, x_mid:x_end,:]
    label_stack[:,:,2:4]= resized_y_center_label#[y_start:y_mid, x_start:x_mid,:]

    
    return rgb_stack, label_stack


# https://github.com/malhotraa/carvana-image-masking-challenge/blob/master/notebooks/model_cnn.ipynb
def randomHueSaturationValue(image,  hue_shift_limit=(-50, 50),
                             sat_shift_limit=(-5, 5),
                             val_shift_limit=(-15, 15)):
    
    image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(image)

    hue_shift = np.random.uniform(hue_shift_limit[0], hue_shift_limit[1])
    h = cv2.add(h, hue_shift)
    sat_shift = np.random.uniform(sat_shift_limit[0], sat_shift_limit[1])
    s = cv2.add(s, sat_shift)
    val_shift = np.random.uniform(val_shift_limit[0], val_shift_limit[1])
    v = cv2.add(v, val_shift)
    image = cv2.merge((h, s, v))
    image = cv2.cvtColor(image, cv2.COLOR_HSV2RGB)

    return image

def randomBrightness(rgb_img):
    # useful link: https://github.com/vxy10/ImageAugmentation/blob/master/README.md
    image1 = cv2.cvtColor(rgb_img,cv2.COLOR_RGB2HSV)
    r = np.random.uniform()
    random_bright = .25+r
    #print(random_bright)
    image1[:,:,2] = image1[:,:,2]*random_bright
    rgb_img = cv2.cvtColor(image1,cv2.COLOR_HSV2RGB)
    return rgb_img