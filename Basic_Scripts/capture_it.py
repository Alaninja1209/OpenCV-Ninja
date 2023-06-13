import cv2 as cv

cameraCapture = cv.VideoCapture(0)
fps = 30  # An assumption

# Adjust size of frames
size = (int(cameraCapture.get(cv.CAP_PROP_FRAME_WIDTH)), int(cameraCapture.get(cv.CAP_PROP_FRAME_HEIGHT)))

videoWriter = cv.VideoWriter('MyOutputVid.avi', cv.VideoWriter_fourcc('I','4','2','0'),fps, size)

success, frame = cameraCapture.read()
numFramesRemaining = 10 * fps - 1 # 10 seconds of frames

while success and numFramesRemaining > 0:
    videoWriter.write(frame)
    success, frame = cameraCapture.read()
    numFramesRemaining -= 1