# This training file where my model is trained
# After training is finished weights are saved

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


def print_speed(arr):
	for i in range(20):
		print(arr[i][1])

train_filenameX = "train_dataX.npy"
train_filenameY = "train_dataY.npy"

test_filenameX = "test_dataX.npy"
test_filenameY = "test_dataY.npy"

train_dataX = np.load(train_filenameX)
train_dataY = np.load(train_filenameY)



filepath = 'model-weights.h5'
earlyStopping = EarlyStopping(monitor='val_loss', 
                              patience=1, 
                              verbose=1, 
                              min_delta = 0.23,
                              mode='min',)
modelCheckpoint = ModelCheckpoint(filepath, 
                                  monitor = 'val_loss', 
                                  save_best_only = True, 
                                  mode = 'min', 
                                  verbose = 1,
                                 save_weights_only = True)
callbacks_list = [modelCheckpoint]


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

val_length = int(len(train_dataX)*0.1)
history = model.fit(x = train_dataX,
		y = train_dataY, 
        batch_size = 1, 
        epochs = 20,
        callbacks = callbacks_list,
        verbose = 1,
        validation_split = 0.1)

print(history)

### plot the training and validation loss for each epoch
fig, ax = plt.subplots(figsize=(20,10))
plt.plot(history.history['loss'], 'ro--')
plt.plot(history.history['val_loss'], 'go--')
plt.title('Model-v2test mean squared error loss 15 epochs')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.savefig('./assets/MSE_per_epoch.png')
plt.close()

print('done')