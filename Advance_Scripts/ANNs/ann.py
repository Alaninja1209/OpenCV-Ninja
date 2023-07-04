import cv2 as cv
import numpy as np

'''ANN, better known as artificial neural networks, are based on an input layer, hidden layer and
an output layer'''

'''The error is considered the difference between the aproximate error and the espected error'''

'''To train the method we use back propagation in order to decreased the error for each time the 
neural netowrk is trained'''

# Creating untrained ann
ann = cv.ml.ANN_MLP_create()

# Configure nodes and layers
ann.setLayerSizes(np.array([9, 15, 9], np.uint8))

#Setting activation function, training method, and termination criteria
ann.setActivationFunction(cv.ml.ANN_MLP_SIGMOID_SYM, 0.6, 1.0)
ann.setTrainMethod(cv.ml.ANN_MLP_BACKPROP, 0.1, 0.1)
ann.setTermCriteria((cv.TERM_CRITERIA_MAX_ITER | cv.TERM_CRITERIA_EPS, 100, 1.0))

# Training the ANN
training_samples = np.array([[1.2, 1.3, 1.9, 2.2, 2.3, 2.9, 3.0, 3.2, 3.3]], np.float32)
layout = cv.ml.ROW_SAMPLE
training_responses = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]], np.float32)
data = cv.ml.TrainData_create(training_samples, layout, training_responses)
ann.train(data)

# Classifying and printing the result
test_samples = np.array([[1.4, 1.5, 1.2, 2.0, 2.5, 2.8, 3.0, 3.1, 3.8]], np.float32)
prediction = ann.predict(test_samples)
print(prediction)