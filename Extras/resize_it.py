import cv2 as cv
import numpy as np

# Reading the image
imagen = cv.imread(r'Detect_Objects\images\oshki.jpg')

# Resizing the image
width = 800
heigth = 600
new_image = cv.resize(imagen, (width, heigth))

# Saving new image
cv.imwrite('imagen_redimensionada.jpg', new_image)
