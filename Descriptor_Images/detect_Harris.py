import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

'''Harris Detector uses Sobel operator which detects edges by measuring horizontal and vertical
differences between pixel values in neighborhood'''

'''The Harris Detector return as a floating-point format, it represents a score for the
corresponding pixel in the source image'''

img = cv.imread(r'Descriptor_Images\images\chess_board.png')

# Apply grascale
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# Apply the Harris Detector
# The third parameter must be an odd number between 3 and 31
dst = cv.cornerHarris(gray, 2, 23, 0.04)

img[dst > 0.01 * dst.max()] = [0, 0, 255]
cv.imshow('Harris detector', img)
cv.waitKey()
cv.destroyAllWindows()