import cv2 as cv
from matplotlib import pyplot as plt

'''To match objects in each image we use the ORB method which includes the BRIEF descriptors and 
FAST algorithm'''

img0 = cv.imread(r'Descriptor_Images\images\nasa_logo.png', cv.IMREAD_GRAYSCALE)
img1 = cv.imread(r'Descriptor_Images\images\kennedy_space_center.jpg', cv.IMREAD_GRAYSCALE)

# Perform ORB feature detection and description
orb = cv.ORB_create()
kp0, des0 = orb.detectAndCompute(img0, None)
kp1, des1 = orb.detectAndCompute(img1, None)

# Perform brute force matching
bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
matches = bf.match(des0, des1)

# Sort images by distance
img_matches = cv.drawMatches(img0, kp0, img1, kp1, matches[:25], img1, flags=cv.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)

# Show matches
plt.imshow(img_matches)
plt.show()