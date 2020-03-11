# This code used for testing the trained model
# Saved weights are imported before testing

import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.core import Activation, Dropout, Flatten, Dense, Lambda
from keras.layers import ELU
from keras.optimizers import Adam
import keras.backend.tensorflow_backend as KTF
from keras.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np 
import cv2
import os, os.path
from random import shuffle
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.gridspec as gridspec

WEIGHTS = 'model-weights.h5'

test_filenameX = "test_dataX.npy"
test_filenameY = "test_dataY.npy"
real_test_filename = "ProcessedTestData.npy"

test_dataX = np.load(test_filenameX)
test_dataY = np.load(test_filenameY)
real_test_data = np.load(real_test_filename)

N_img_height = 66
N_img_width = 220
N_img_channels = 3
inputShape = (N_img_height, N_img_width, N_img_channels)

model = Sequential()
# normalization
model.add(Lambda(lambda x: x/ 127.5 - 1, input_shape = inputShape))

model.add(Convolution2D(24, 5, 5, 
                        subsample=(2,2), 
                        border_mode = 'valid',
                        init = 'he_normal',
                        name = 'conv1'))

model.add(ELU())    
model.add(Convolution2D(36, 5, 5, 
                        subsample=(2,2), 
                        border_mode = 'valid',
                        init = 'he_normal',
                        name = 'conv2'))

model.add(ELU())    
model.add(Convolution2D(48, 5, 5, 
                        subsample=(2,2), 
                        border_mode = 'valid',
                        init = 'he_normal',
                        name = 'conv3'))
model.add(ELU())
model.add(Dropout(0.5))
model.add(Convolution2D(64, 3, 3, 
                        subsample = (1,1), 
                        border_mode = 'valid',
                        init = 'he_normal', #gaussian init
                        name = 'conv4'))

model.add(ELU())              
model.add(Convolution2D(64, 3, 3, 
                        subsample= (1,1), 
                        border_mode = 'valid',
                        init = 'he_normal',
                        name = 'conv5'))
          
          
model.add(Flatten(name = 'flatten'))
model.add(ELU())
model.add(Dense(100, init = 'he_normal', name = 'fc1'))
model.add(ELU())
model.add(Dense(50, init = 'he_normal', name = 'fc2'))
model.add(ELU())
model.add(Dense(10, init = 'he_normal', name = 'fc3'))
model.add(ELU())

# do not put activation at the end because we want to exact output, not a class identifier
model.add(Dense(1, name = 'output', init = 'he_normal'))

adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
model.compile(optimizer = adam, loss = 'mse')

model.load_weights(WEIGHTS)
score = model.evaluate(test_dataX,test_dataY,verbose=1)
print(score)


pred = model.predict(test_dataX,verbose=1)
np.save("predeicted_for_video.npy",pred)

error = 0
length = len(test_dataY)
for i in range(length):
	temp = test_dataY[i] - pred[i][0]
	error += temp**2
error /= length
print("The MSE error is:",error)

for i in range(10):
	print("Ground Truth:",test_dataY[i])
	print("Prediction:",pred[i][0])
	print("")

print("\n")
print("Prediction on Real Test Data:")
my_prediction = model.predict(real_test_data,verbose=1)
np.save("test_predicted.npy",my_prediction)

file = open("test.txt","w+")

print("Size of the Real Test Data:",len(real_test_data))
for i in range(len(real_test_data)):
	file.write(str(my_prediction[i][0])+"\n")
#print(my_prediction)
#print('Accuracy on test data: {}% \nError on test data: {}\n'.format(prediction[1], 1 - prediction[1])) 