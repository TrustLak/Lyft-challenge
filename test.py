# test model

## Include stuff:
import tensorflow as tf
import numpy as np
from keras.optimizers import Adam
import json
from random import shuffle
from PIL import Image
from utils import *
import cv2
import matplotlib.pyplot as plt

n_test_imgs = 1

IMAGE_WIDTH     = 400
IMAGE_HIGHT     = 400
img_w = IMAGE_WIDTH
img_h = IMAGE_HIGHT
img_d = 3
n_labels = 2
in_shape = (img_w, img_h, img_d)

i = 156

X_test_batch = []
y_test_batch = []
for j in range(10):
    # Add training exmaple
    file_name_X = 'Test/RGB/' + str(896+j)+ '.png'
    X_image = np.asarray(Image.open(file_name_X))
    X_image = cv2.resize(X_image, (400, 400)) 
    X_test_batch.append(X_image)
    

X_test = np.array(X_test_batch)
y_test = np.array(y_test_batch)
print(y_test.shape)
# create a new model then load its weights:
model = build_model()
model.load_weights("model.h5")
# test model:
result = model.predict(X_test)
print("Testing syntax 1:")
print(result.shape)
print(result[0,0, :])


# visualize result:
# we care about indeces: {6: RoadLines, 7: Roads, 10: Vehicles}
# Grab cars (from semantic segmentation medium post by Kyle)
def get_bin_images(labels):
    # labes is 4d array with dim: (1, w, h, 13)
    vehicle_result = np.zeros([img_h, img_w])    
    road_result    = np.zeros([img_h, img_w]) 
    flat_labels = np.argmax(labels, axis=2)
    print("flat_labels.shape")
    print(flat_labels.shape)

    # converting to binary images: https://medium.com/@kylesf/udacitylyftchallenge-ebffe71a6b22
    # vehicles:
    vehicle_binary_result = np.where(flat_labels==10,1,0).astype('uint8')
    vehicle_binary_result[320:,:] = 0
    vehicle_binary_result = vehicle_binary_result * 255

    # road
    road_lines = np.where((flat_labels==6),np.ones([1,13]),np.zeros([1,13])).astype('uint8')
    roads = np.where((flat_labels==7),1,0).astype('uint8')
    road_binary_result = (road_lines | roads) * 255

    return vehicle_binary_result, road_binary_result

print(result[0].shape)


for i in range(14):
    vehicle_result, road_result = get_bin_images(result[i])
    #plt.imshow(cv2.resize(vehicle_result, (800, 600)))
    plt.imshow(road_result)
    plt.show()
print(vehicle_result.shape)

# model_labels = [None, vehicles, road]
def label_map(labels):
    flat_labels = np.argmax(labels, axis=2)
    label_map = np.zeros([img_h, img_w, n_labels]) 

    vehicles = np.where((labels==10),1,0).astype('uint8')
    label_map[:,:,0] = vehicles

    lanes = np.where((labels==6),1,0).astype('uint8')
    lane_lines = np.where((labels==7),1,0).astype('uint8')
    road = (lanes | lane_lines) 
    label_map[:,:,1] = road

    # l1 = not label_map[:,:,1]
    non_vehicle = np.where((labels!=10),1,0).astype('uint8')
    non_road = np.where((road==0),1,0).astype('uint8')
    label_map[:,:,2] = (non_vehicle & non_road)

    label_map[420:,:,:] = np.array([0,0,1])
    return label_map
