import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

'''Grabcut, gets de foreground and background of an image, great for segmentation'''

original = cv.imread(r'Img_Segmentation\images\statue_small.jpg')

# Getting copy of image
img = original.copy()

# Making mask of zeros with same size of the image
mask = np.zeros(img.shape[:2], np.uint8)

bgd_model = np.zeros((1, 65), np.float64)
fgd_model = np.zeros((1, 65), np.float64)

rect = (100, 1, 421, 378)

# Implementing the grabcut algorithm
cv.grabCut(img, mask, rect, bgd_model, fgd_model, 5, cv.GC_INIT_WITH_RECT)

# Making a mask to paint the backgound black
mask2 = np.where((mask==2) | (mask==0), 0 , 1).astype('uint8')
img = img * mask2[:, :, np.newaxis]

# Plotting both images
plt.subplot(121)
plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
plt.title("grabcut")
plt.xticks([])
plt.yticks([])

plt.subplot(122)
plt.imshow(cv.cvtColor(original, cv.COLOR_BGR2RGB))
plt.title("original")
plt.xticks([])
plt.yticks([])

plt.show()