import cv2 as cv
import os

'''This code tends to get the dataset for face recognition, it scans the region of the face and
save it in an pgm file'''

output_folder = 'Extras\ivan'

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

face_cascade = cv.CascadeClassifier(r'Advance_Scripts\cascade\haarcascade_frontalface_default.xml')
eye_cascade = cv.CascadeClassifier(r'Advance_Scripts\cascade\haarcascade_eye.xml')

camera = cv.VideoCapture(0)
count = 0

while (cv.waitKey(1) == -1):
    success, frame = camera.read()

    if success:
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5, minSize=(120, 120))
        for (x, y, w, h) in faces:
            cv.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            face_img = cv.resize(gray[y:y+h, x:x+w], (200, 200))
            face_filename = '%s/%d.pgm' % (output_folder, count)
            cv.imwrite(face_filename, face_img)
            count += 1

        cv.imshow('Capturing faces...', frame)