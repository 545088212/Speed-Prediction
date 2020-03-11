# This code is used to make a video from frames
# Text with predicted speed is written to each frame

import numpy as np 
import os, os.path
import cv2
import glob
from moviepy.editor import VideoFileClip
from moviepy.editor import ImageSequenceClip

imageDir = "test/"
filePredicted = "test_predicted.npy"

dataPredicted = np.load(filePredicted)

offset = 30
font = cv2.FONT_HERSHEY_SIMPLEX


image_list = glob.glob(imageDir+"*.jpg")
image_list.sort()

i = 0
my_images = []
for imagePath in image_list:
	if i > 600:
		break
	i += 1
	image = cv2.imread(imagePath)
	predicted = dataPredicted[i][0]

	cv2.putText(image,'Predicted: ' + str(predicted)[:5],(5,offset), font, 1,(0,255,0),1,cv2.LINE_AA)
	image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
	my_images.append(image)


cv2.imshow("Edited Image", my_images[0])
cv2.waitKey(0)
cv2.destroyAllWindows()

clip = ImageSequenceClip(my_images, fps=11.7552)
clip.write_videofile("myVideo.mp4", fps = 11.7552)
print('done creating video')