import cv2 as cv

'''Haar-like features describe the pattern of contrast  among adjacent image regions'''

'''Some features are distinctive, they only appear in only one type of class of object such as faces, 
this ones are called cascades'''

# Declare cascade clasifier object
face_cascade = cv.CascadeClassifier(r'Advance_Scripts\cascade\haarcascade_frontalface_default.xml')

# Reading the image
img = cv.imread(r'Advance_Scripts\images\Lingotes.jpg')

# Turn to grayscale, cascade classifier expects grayscale images
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# Begin face detection
faces = face_cascade.detectMultiScale(gray, 1.08, 8)
for (x, y, w, h) in faces:
    img = cv.rectangle(img, (x,y), (x + w, y + h), (255,160, 0), 2)

# Showing results
cv.namedWindow('Lingotes detected')
cv.imshow('Lingotes detected', img)
cv.waitKey()
cv.destroyAllWindows()