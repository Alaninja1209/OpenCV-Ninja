import cv2 as cv

'''Sift detector is a function that detects features and is not affected by the scale
of the image'''

img = cv.imread(r'Descriptor_Images\images\Empire_state.jpg')

# Apply grayscale
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# Create SIFT detector and compute features and descriptors of the image
sift = cv.SIFT_create()
keypoints, descriptors = sift.detectAndCompute(gray, None)

cv.drawKeypoints(img, keypoints, img, (51, 163, 236), cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

cv.imshow('Sift_keypoints', img)
cv.waitKey()
cv.destroyAllWindows()