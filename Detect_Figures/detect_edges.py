import cv2 as cv
import numpy as np

# Reading image
img = cv.imread(r'Detect_Objects\images\night_building.jpg')

# Applying Canny filter
canny = cv.Canny(img, 200, 300)

cv.imshow('Canny', canny)
cv.waitKey()
cv.destroyAllWindows()