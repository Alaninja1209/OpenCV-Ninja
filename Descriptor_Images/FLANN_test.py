import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

'''FLANN is known as a library for performing fast aproximate nearest neighbor searches in
high dimensional spaces'''

img0 = cv.imread(r'Descriptor_Images\images\gauguin_entre_les_lys.jpg')
img1 = cv.imread(r'Descriptor_Images\images\gauguin_paintings.png')

# Perform SIFT feature description and detection
sift = cv.SIFT_create()
kp0, des0 = sift.detectAndCompute(img0, None)
kp1, des1 = sift.detectAndCompute(img1, None)

# Define FLANN-based matching
FLANN_INDEX_KDTREE = 1

# This parameters declare the behavior of indexes and search objects that are used internally by FLANN to compute matches
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)

# Perform FLANN matching 
flann = cv.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(des0, des1, k=2)

# Prepare mask to draw good matches
mask_matches = [[0, 0] for i in range(len(matches))]

# Compute radio test
for i, (m, n) in enumerate(matches):
    if m.distance < 0.7 * n.distance: # Applying radio test
        mask_matches[i] = [1, 0]

# Plotting the results
img_matches = cv.drawMatchesKnn(img0, kp0, img1, kp1, matches, None, matchColor=(0, 255, 0), singlePointColor=(255, 0, 0), matchesMask=mask_matches, flags=0)
plt.imshow(img_matches)
plt.show()