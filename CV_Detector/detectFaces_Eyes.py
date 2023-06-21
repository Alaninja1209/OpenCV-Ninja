import cv2 as cv

# Initialize two classifier objects
face_cascade = cv.CascadeClassifier(r'Advance_Scripts\cascade\haarcascade_frontalface_default.xml')
eye_cascade = cv.CascadeClassifier(r'Advance_Scripts\cascade\haarcascade_eye.xml')

# Start iterating per frame unless the user presses a key
camera = cv.VideoCapture(0)
while (cv.waitKey(1) == -1):
    success, frame = camera.read()

    if success:
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5, minSize=(120, 120))

        for (x, y, w, h) in faces:
            cv.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            # Creating ROI based on the rectangles created by face detector
            roi_gray = gray[y:y + h, x:x + w]

            eyes = eye_cascade.detectMultiScale(roi_gray, 1.03, 5, minSize=(40, 40))

            for (ex, ey, ew, eh) in eyes:
                cv.rectangle(frame, (x + ex, y + ey), (x + ex + ew, y + ey + eh), (0, 255, 0), 2)

        cv.imshow('Face Detection', frame)