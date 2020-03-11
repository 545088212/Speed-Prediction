# This code is used to read data from numpy file and process it
# At first brightness is changed then optical flow is applied to two consecituve frames


import os, os.path
import numpy as np
import cv2
from random import shuffle


filename_DataX = "MyDatasetX.npy"
filename_DataY = "MyDatasetY.npy"
filename_RealTestX = "MyRealTestData.npy"


def change_brightness(image, bright_factor):
    """
    Augments the brightness of the image by multiplying the saturation by a uniform random variable
    Input: image (RGB)
    returns: image with brightness augmentation
    """
    
    hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    # perform brightness augmentation only on the second channel
    hsv_image[:,:,2] = hsv_image[:,:,2] * bright_factor
    
    # change back to RGB
    image_rgb = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2RGB)
    return image_rgb



def opticalFlowDense(image_current, image_next):
    """
    input: image_current, image_next (RGB images)
    calculates optical flow magnitude and angle and places it into HSV image
    * Set the saturation to the saturation value of image_next
    * Set the hue to the angles returned from computing the flow params
    * set the value to the magnitude returned from computing the flow params
    * Convert from HSV to RGB and return RGB image with same size as original image
    """
    
   
    
    gray_current = cv2.cvtColor(image_current, cv2.COLOR_RGB2GRAY)
    gray_next = cv2.cvtColor(image_next, cv2.COLOR_RGB2GRAY)
    
    
    hsv = np.zeros((66, 220, 3))
    #hsv = np.zeros((540, 720, 3))
    # set saturation
    hsv[:,:,1] = cv2.cvtColor(image_next, cv2.COLOR_RGB2HSV)[:,:,1]
 
    # Flow Parameters
	# low_mat = cv2.CV_32FC2
    flow_mat = None
    image_scale = 0.5
    nb_images = 1
    win_size = 15
    nb_iterations = 2
    deg_expansion = 5
    STD = 1.3
    extra = 0
	
	# obtain dense optical flow paramters
    flow = cv2.calcOpticalFlowFarneback(gray_current, gray_next,  
                                        flow_mat, 
                                        image_scale, 
                                        nb_images, 
                                        win_size, 
                                        nb_iterations, 
                                        deg_expansion, 
                                        STD, 
                                        0)
                                        
        
    # convert from cartesian to polar
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])  
        
    # hue corresponds to direction
    hsv[:,:,0] = ang * (180/ np.pi / 2)
    
    # value corresponds to magnitude
    hsv[:,:,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
    
    # convert HSV to int32's
    hsv = np.asarray(hsv, dtype= np.float32)
    rgb_flow = cv2.cvtColor(hsv,cv2.COLOR_HSV2RGB)
    return rgb_flow


DataX = np.load(filename_DataX)
DataY = np.load(filename_DataY)
RealTestX = np.load(filename_RealTestX)

new_Data = []
new_RealTestX = []

length = len(DataX)
for i in range(length):
	if i+1 < length:
		images = [DataX[i],DataX[i+1]]
		speeds = [DataY[i],DataY[i+1]]
		new_Data.append([images,speeds])

		#new_RealTestX.append([RealTestX[i],RealTestX[i+1]])
		#new_Data.append(images)

lll = len(RealTestX)
for i in range(lll):
	if i+1 < lll:
		new_RealTestX.append([RealTestX[i],RealTestX[i+1]])


print("Deleting some memory")
#Freeing memory
DataX = None
DataY = None
RealTestX = None

#np.save("IntermediateData.npy",new_Data)

brightness = np.random.uniform()
print("This is the BrightFactor:",brightness)


final_DataX = []
final_DataY = []
test_Data = []
length = len(new_Data)
for i in range(length):
	image_curr = new_Data[i][0][0]
	image_next = new_Data[i][0][1]
	image_curr_illuminated = change_brightness(image_curr,brightness)
	image_next_illuminated = change_brightness(image_next,brightness)


	flow_image = opticalFlowDense(image_curr_illuminated,image_next_illuminated)
	mean_speed = np.mean([new_Data[i][1][0],new_Data[i][1][1]])
	final_DataX.append(flow_image)
	final_DataY.append(mean_speed)


lll = len(new_RealTestX)
for i in range(lll):
	test_curr = new_RealTestX[i][0]
	test_next = new_RealTestX[i][1]
	test_curr_illuminated = change_brightness(test_curr,brightness)
	test_next_illuminated = change_brightness(test_next,brightness)

	test_flow = opticalFlowDense(test_curr_illuminated,test_next_illuminated)
	test_Data.append(test_flow)


print("Deleting some memory")
#Freeing memory
new_Data = None
new_RealTestX = None

#shuffle(final_Data)
cv2.imshow("Flow Image", final_DataX[0])
cv2.waitKey(0)
cv2.destroyAllWindows()

np.save("DataX.npy",final_DataX)
np.save("DataY.npy",final_DataY)
np.save("ProcessedTestData.npy",test_Data)
print("Successful")