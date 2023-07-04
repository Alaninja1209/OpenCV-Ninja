import cv2 as cv
import numpy as np
from random import randint, uniform

'''Implement function to create dataset, generate random samples of different classes and 
to declare the correct classification'''

def dog_sample():
    return[uniform(10.0, 20.0), uniform(1.0, 1.5), randint(39, 42)]

def dog_class():
    return[1, 0, 0, 0]

def condor_sample():
    return[uniform(3.0, 10.0), randint(3.0, 5.0), 0]

def condor_class():
    return[0, 1, 0, 0]

def dolphin_sample():
    return[uniform(30.0, 190.0), uniform(5.0, 15.0), randint(80, 100)]

def dolphin_class():
    return[0, 0, 1, 0]

def dragon_sample():
    return[uniform(1200.0, 1800.0), uniform(30.0, 40.0), randint(160, 180)]

def dragon_class():
    return[0, 0, 0, 1]

# Helper function to convert sample and classification into np arrays
def record(sample, classification):
    return((np.array([sample]), np.float32), np.array([classification], np.float32))

# Creating an untrained ann
ann = cv.ml.ANN_MLP_create()

# Configure nodes and layers
ann.setLayerSizes(np.array([3, 50, 4], np.uint8))

#Setting activation function, training method, and termination criteria
ann.setActivationFunction(cv.ml.ANN_MLP_SIGMOID_SYM, 0.6, 1.0)
ann.setTrainMethod(cv.ml.ANN_MLP_BACKPROP, 0.1, 0.1)
ann.setTermCriteria((cv.TERM_CRITERIA_MAX_ITER | cv.TERM_CRITERIA_EPS, 100, 1.0))

# Create fake animal data
RECORDS = 20000
records = []
for x in range(0, RECORDS):
    records.append(record(dog_sample(), dog_class()))
    records.append(record(dragon_sample(), dragon_class()))
    records.append(record(condor_sample(), condor_class()))
    records.append(record(dolphin_sample(), dolphin_class()))

# Training the ANN
epochs = 10
for e in range(0, epochs):
    print("epoch: %d" % e)
    samples = []
    responses = []
    for t, c in records:
        samples.append(t[0].astype(np.float32))
        responses.append(c[0].astype(np.float32))
    samples = np.array(samples, dtype=np.float32)
    responses = np.array(responses, dtype=np.float32)
    if ann.isTrained():
        ann.train(samples, cv.ml.ROW_SAMPLE, responses)
    else:
        ann.train(samples, cv.ml.ROW_SAMPLE, responses)




# Testing the ANN
tests = 100

dog_results = 0
for x in range(0, tests):
    clas = int(ann.predict(
        np.array([dog_sample()], np.float32))[0])
    print("class: %d" % clas)
    if clas == 0:
        dog_results += 1

condor_results = 0
for x in range(0, tests):
    clas = int(ann.predict(
        np.array([condor_sample()], np.float32))[0])
    print("class: %d" % clas)
    if clas == 1:
        condor_results += 1

dolphin_results = 0
for x in range(0, tests):
    clas = int(ann.predict(
        np.array([dolphin_sample()], np.float32))[0])
    print("class: %d" % clas)
    if clas == 2:
        dolphin_results += 1

dragon_results = 0
for x in range(0, tests):
    clas = int(ann.predict(
        np.array([dragon_sample()], np.float32))[0])
    print("class: %d" % clas)
    if clas == 3:
        dragon_results += 1

print("dog accuracy: %.2f%%" % (100.0 * dog_results / tests))
print("condor accuracy: %.2f%%" % (100.0 * condor_results / tests))
print("dolphin accuracy: %.2f%%" % \
    (100.0 * dolphin_results / tests))
print("dragon accuracy: %.2f%%" % (100.0 * dragon_results / tests))