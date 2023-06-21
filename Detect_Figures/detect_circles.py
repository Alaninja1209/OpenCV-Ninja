import cv2 as cv
import numpy as np

'''Apply Hough Circles Transform, define minimum distance between circle center and 
the maximum and minimum for radius'''

figures = cv.imread(r'Detect_Objects\images\figuras.png')

gray = cv.cvtColor(figures, cv.COLOR_BGR2GRAY)
gray = cv.medianBlur(gray, 5)

circles = cv.HoughCircles(gray, cv.HOUGH_GRADIENT, 1, 120, param1=100, param2=30, minRadius=0, maxRadius=0)

circles = np.uint16(np.around(circles))

for i in circles[0, :]:
    cv.circle(figures, (i[0], i[1]), i[2], (0, 255, 0), 2)

    cv.circle(figures, (i[0], i[1]), 2, (0, 0, 255), 3)

cv.imshow('Hough Circles', figures)
cv.waitKey()
cv.destroyAllWindows()