import cv2 as cv
import numpy as np

'''Probabilistic Hough Transform, analyze a set of the image points and estimates the probability 
that these points belong to the same line'''

'''The ideal source for Hough Transform is an image denoised and only represent edges'''

img = cv.imread(r'Detect_Figures\images\night_building.jpg')

# Turning to gray scale
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# Applying Canny edges
edges = cv.Canny(gray, 50, 120)

minLineLength = 5
maxLineGap = 20

# rho, search for separate lines as one pixel and degree
lines = cv.HoughLinesP(edges, 1, np.pi / 180.0, 20, minLineLength, maxLineGap)

for x1, y1, x2, y2 in lines[0]:
    cv.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

cv.imshow('Edges', edges)
cv.imshow('Lines', img)

cv.waitKey()
cv.destroyAllWindows()