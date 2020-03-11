# This code is used to split data into training and testing sets

import numpy as np 
import cv2
import os, os.path
from random import shuffle


filenameX = "DataX.npy"
filenameY = "DataY.npy"

DatasetX = np.load(filenameX)
DatasetY = np.load(filenameY)

myData = []

for i in range(len(DatasetX)):
	myData.append([DatasetX[i],DatasetY[i]])

shuffle(myData)
split_length = int(len(DatasetX)*0.1)


test_data = myData[:split_length]
train_data = myData[split_length:]
myData = None


shuffle(train_data)
shuffle(test_data)

train_dataX = []
train_dataY = []
test_dataX = []
test_dataY = []

for i in range(len(train_data)):
	train_dataX.append(train_data[i][0])
	train_dataY.append(train_data[i][1])

for i in range(len(test_data)):
	test_dataX.append(test_data[i][0])
	test_dataY.append(test_data[i][1])

np.save("train_dataX.npy",train_dataX)
np.save("train_dataY.npy",train_dataY)
np.save("test_dataX.npy",test_dataX)
np.save("test_dataY.npy",test_dataY)
print("finished")