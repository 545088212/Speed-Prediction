# This code used to read train and test data (images and labels) from folder
# Then data is cropped and resized, after that put into numpy file


import numpy as np 
import os, os.path
import cv2
import glob

"""
#imageDir = "train/"
imageDir = "test/"

image_list = glob.glob(imageDir+"*.jpg")
image_list.sort()

image_array = []

for imagePath in image_list:
	image = cv2.imread(imagePath)
	image_cropped = image[100:440,50:670]
	image_resized = cv2.resize(image_cropped, (220, 66), interpolation = cv2.INTER_AREA)
	image_array.append(image_resized)
	print(imagePath,"is opened and added to array")

#np.save("MyDatasetX.npy",image_array)
np.save("MyRealTestData.npy",image_array)
print("Successfull")

"""

trainY = []
filename = ("train.txt")
input_file = open(filename,"r")
print(filename,"is opened")

if input_file.mode == 'r':
	lines = input_file.readlines()

	for x in lines:
		speed = x.split(" ")
		speed = speed[0]
		speed = speed[:-2]
		speed = float(speed)
		print("This is the speed:",speed)
		trainY.append(speed)
	
print(trainY[0],trainY[1])
np.save("MyDatasetY.npy",trainY) 