# test model

## Include stuff:
import tensorflow as tf
import numpy as np
import json
from random import shuffle
from PIL import Image
from utils import *
import cv2
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential, Model
import glob


IMAGE_WIDTH     = 800
IMAGE_HIGHT     = 512
img_w = IMAGE_WIDTH
img_h = IMAGE_HIGHT
img_d = 3


X_test_batch = []
y_test_batch = []
X_images = []

rgb_test = glob.glob('Test/RGB/*.png')
label_test = glob.glob('Test/Seg/*.png')
for i in range(len(rgb_test)):
    # Add training exmaple
    X_image = np.asarray(Image.open(rgb_test[i]))
    label_y = np.asarray(Image.open(label_test[i]))
    y_label = label_map(label_y[:,:,0])
    X_images.append(X_image)
    img, label= process_img(X_image, y_label)
    # plt.imshow(img)
    # plt.show()
    X_images.append(X_image)
    X_test_batch.append(img)  
    #y_test_batch.append(label)
    
X_test = np.array(X_test_batch)#.reshape((8,400,400,3))
y_test = np.array(y_test_batch)

# create a new model then load its weights:
model = build_model()
model.load_weights("model11.h5")
# test model:
result = model.predict(X_test)


def get_bin_images(labels):
    q_v0 = labels[:,:,0]
    q_v1 = labels[:,:,2]
    q_v2 = labels[:,:,4]
    q_v3 = labels[:,:,6]
    q_r0 = labels[:,:,1]
    q_r1 = labels[:,:,3]
    q_r2 = labels[:,:,5]
    q_r3 = labels[:,:,7]
    print(q_r3.shape)
    # vehicles:
    l_v_0 = np.where(q_v0 > 0.5,1,0).astype('uint8')
    l_v_1 = np.where(q_v1 > 0.5,1,0).astype('uint8')
    l_v_2 = np.where(q_v2 > 0.5,1,0).astype('uint8')
    l_v_3 = np.where(q_v3 > 0.5,1,0).astype('uint8')
    #roads:
    
    # label_roads = np.where(q_0y > 0.7,1,0).astype('uint8')
    #road_binary_result = roads #* 255
    return  l_v_0, l_v_1, l_v_2, l_v_3

temp = np.zeros((600,800))
for i in range(len(rgb_test)):
    v0, v1, v2, v3 = get_bin_images(result[i])
    #plt.imshow(cv2.resize(vehicle_result, (800, 600)))
    cv2.imshow("Original", X_images[i])
    cv2.imshow("road1", v0*255)
    cv2.imshow("road2", v1*255)
    cv2.imshow("road3", v2*255)
    cv2.imshow("road4", v3*255)

    cv2.waitKey(3000)
    cv2.destroyAllWindows()

# print(vehicle_result.shape)