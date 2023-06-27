import cv2 as cv
from matplotlib import pyplot as plt

img0 = cv.imread(r'Descriptor_Images\images\nasa_logo.png', cv.IMREAD_GRAYSCALE)
img1 = cv.imread(r'Descriptor_Images\images\kennedy_space_center.jpg', cv.IMREAD_GRAYSCALE)

# Perform ORB feature detection and description
orb = cv.ORB_create()
kp0, des0 = orb.detectAndCompute(img0, None)
kp1, des1 = orb.detectAndCompute(img1, None)

# Perform brute force matching
bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=False)

# Performing KNN matching
pair_matches = bf.knnMatch(des0, des1, k=2)

# Sorting the pairs of matches based on distance
pair_matches = sorted(pair_matches, key=lambda x:x[0].distance)

# Applying the theory of ratio test
matches = [x[0] for x in pair_matches if len(x) > 1 and x[0].distance < 0.8 * x[1].distance]

# Sort images by distance
img_matches = cv.drawMatches(img0, kp0, img1, kp1, matches[:25], img1, flags=cv.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
#img_matches = cv.drawMatchesKnn(img0, kp0, img1, kp1, pair_matches[:25], img1, flags=cv.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)

# Show matches
plt.imshow(img_matches)
plt.show()
