import cv2 as cv
import numpy as np
from scipy import ndimage

kernel_3x3 = np.array([[-1, -1, -1],
                       [-1, 8, -1],
                       [-1, -1, -1]])

# Reading the image
img = cv.imread(r'Extras\images\people.jpg', 0)

# Provides convolve function for multidimensional arrays
k3 = ndimage.convolve(img, kernel_3x3)

# Applying Gaussian noise
blurred = cv.GaussianBlur(img, (17, 17), 0)
g_hpf = img - blurred

cv.imshow('3x3', k3)
cv.imshow('Blurred', blurred)
cv.imshow('HPF', g_hpf)

cv.waitKey()
cv.destroyAllWindows()