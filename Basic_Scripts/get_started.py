import cv2 as cv
import numpy as np

# An image is a multidimensional array with columns and arrays with pixels
# Values for each pixel are in range (0,255). 0 is black and 255 is white
img = np.zeros((3, 3), dtype=np.uint8)

# Turning image into BGR, three channels represent Red, Green and Blue
img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)

# Converting images png to jpeg
image = cv.imread(r'Basic_Scripts\images\pic1.png')
cv.imwrite('Pic1.jpg', image)

# Reading an image as grayscale
gray_img = cv.imread(r'Basic_Scripts\images\pic1.png', cv.IMREAD_GRAYSCALE)
cv.imshow('Gray image', gray_img)

cv.waitKey()
cv.destroyAllWindows()