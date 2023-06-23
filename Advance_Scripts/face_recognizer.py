import cv2
import numpy as np
import os

'''The read images function return three variables: list of names, arrays of images and labels'''

'''The EigenFace algorithm works identifies principal components of a certain set of observations,
calculates divergence of the current observation and produces a value'''

'''If the value approaches to 0, it would be an exact match'''

def read_images(path, img_size):
    names = []
    training_images, training_labels = [], []
    label = 0
    for dirname, subdirnames, filenames in os.walk(path):
        for subdirname in subdirnames:
            names.append(subdirname)
            subject_path = os.path.join(dirname, subdirname)

            for filename in os.listdir(subject_path):
                img = cv2.imread(os.path.join(subject_path, filename), cv2.IMREAD_GRAYSCALE)
            
                if img is None:
                    continue

                img = cv2.resize(img, img_size)
                training_images.append(img)
                training_labels.append(label)
            
            label += 1
    training_images = np.array(training_images, np.uint8)
    training_labels = np.array(training_labels, np.int32)

    return names, training_images, training_labels

# Declaring image size and path to folder
path_to_folder = 'Extras'
training_image_size = (200, 200)

# Calling fuction to read dataset
names, training_images, training_labels = read_images(path_to_folder, training_image_size)

# Create and train face recognizer
model = cv2.face.EigenFaceRecognizer_create()
#model = cv2.face.FisherFaceRecognizer_create() 
model.train(training_images, training_labels)

# Initialize face detector
face_cascade = cv2.CascadeClassifier(r'Advance_Scripts\cascade\haarcascade_frontalface_default.xml')

# Initialize video capture
camera = cv2.VideoCapture(0)
while (cv2.waitKey(1) == -1):
    success, frame = camera.read()
    if success:
        faces = face_cascade.detectMultiScale(frame, 1.3, 5)
        for (x, y, w, h) in faces:
            pt1 = (x, y)
            pt2 = (x + w, y + h)
            cv2.rectangle(frame, pt1, pt2, (255, 0, 0), 2)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            roi_gray = gray[x:x + w, y:y + h]
            if roi_gray.size == 0:
                continue
            roi_gray = cv2.resize(roi_gray, training_image_size)
            label, confidence = model.predict(roi_gray)
            text = '%s, confidence=%.2f' % (names[label], confidence)
            cv2.putText(frame, text, (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        
        cv2.imshow('Face Recognition', frame)