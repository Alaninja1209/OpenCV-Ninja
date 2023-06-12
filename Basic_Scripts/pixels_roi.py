import cv2 as cv
import numpy as np
import os

# Reading image
img = cv.imread(r'Basic_Scripts\images\pic1.png')

# Modifying a pixel in the image
img[0, 0] = [0, 0, 255]

# Setting all green values to 0
img[:, :, 1] = 0

# Defining region of interest
my_roi = img[0:100, 0:100]

# Paste data of region of interest
img[300:400, 300:400] = my_roi

cv.imshow('Modified pixel', img)

# Printing data of an image
print(img.shape) # Height, Width and Channels
print(img.size) # Number of elements in an array
print(img.dtype) # Datatype of the array elements

cv.waitKey()
cv.destroyAllWindows()