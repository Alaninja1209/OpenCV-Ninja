import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

'''Morphology consist of dilating (expanding) or eroding (contracting) the white regions of the image, here we apply
an open operation, first erosion and then dilating'''

img = cv.imread(r'Img_Segmentation\images\matthew.jpg')

# Turning to gray scale
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# Applying threshold to divide the image in two regions, white and black
ret, thresh = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)

# Applying morphological transformation to remove noise
kernel = np.ones((3, 3), np.uint8)
opening = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel, iterations=2)

# Finding the sure background region
sure_bg = cv.dilate(opening, kernel, iterations=3)

# Finding the sure foreground region
dist_transform = cv.distanceTransform(opening, cv.DIST_L2, 5)
ret, sure_fg = cv.threshold(dist_transform, 0.7 * dist_transform.max(), 225, 0)
sure_fg = sure_fg.astype(np.uint8)

# Finding the regions in between
unknown = cv.subtract(sure_bg, sure_fg)

'''Here we are bulding barries based on the theory of GrabCut algorithm, nodes are connected by edges,
but some of them would be not which are the water valleys'''
ret, markers = cv.connectedComponents(sure_fg)
markers += 1

# Labeling unknown regions as 0
markers[unknown==255] = 0

markers = cv.watershed(img, markers)
img[markers==-1] = [255, 0, 0]

plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
plt.show()
