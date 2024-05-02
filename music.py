import numpy as np
import scipy.io.wavfile as wav
from python_speech_features import mfcc
from tempfile import TemporaryFile
import os
import math
import pickle
import random
import operator
from collections import defaultdict
#Defines a function distance to calculate the distance between two instances using the distance formula.
def distance(instance1, instance2, k):
    distance = 0
    mm1 = instance1[0]
    cm1 = instance1[1]
    mm2 = instance2[0]
    cm2 = instance2[1]
    distance = np.trace(np.dot(np.linalg.inv(cm2), cm1))
    distance += (np.dot(np.dot((mm2-mm1).transpose(), np.linalg.inv(cm2)), mm2-mm1))
    distance += np.log(np.linalg.det(cm2)) - np.log(np.linalg.det(cm1))
    distance -= k
    return distance
#Defines a function getNeighbors to find the k nearest neighbors of an instance in the training set based on the distance metric.
def getNeighbors(trainingset, instance, k):
    distances = []
    for x in range(len(trainingset)):
        dist = distance(trainingset[x], instance, k) + distance(instance, trainingset[x], k)
        distances.append((trainingset[x][2], dist))
    distances.sort(key=operator.itemgetter(1))
    neighbors = []
    for x in range(k):
        neighbors.append(distances[x][0])
    return neighbors
#Defines a function nearestclass to determine the class label by selecting the most common class among the nearest neighbors
def nearestclass(neighbors):
    classVote = {}
    for x in range(len(neighbors)):
        response = neighbors[x]
        if response in classVote:
            classVote[response] += 1
        else:
            classVote[response] = 1
    sorter = sorted(classVote.items(), key=operator.itemgetter(1), reverse=True)
    return sorter[0][0]
#Defines a function getAccuracy to calculate the accuracy of the classification model based on the predicted labels and the true labels of the test set.
def getAccuracy(testSet, prediction):
    correct = 0
    for x in range(len(testSet)):
        if testSet[x][-1] == prediction[x]:
            correct += 1
    if len(testSet) == 0:
        return 0.0
    return 1.0 * correct / len(testSet)

import librosa

directory = "C:/Users/yadhv/OneDrive/Desktop/Python_WS/archive/Data/genres_original"
trainingSet = []
testSet = []

def loadDataset(filename, split, trset, teset):
    with open(filename, 'rb') as f:
        while True:
            try:
                instance = pickle.load(f)
                if random.random() < split:
                    trset.append(instance)
                else:
                    teset.append(instance)
            except EOFError:
                break

loadDataset('my.dat', 0.66, trainingSet, testSet)

if len(testSet) == 0:
    print("Error: Test set is empty. Check your data loading process.")
else:
    predictions = []
    for instance in testSet:
        predictions.append(nearestclass(getNeighbors(trainingSet, instance, 5)))

    accuracy1 = getAccuracy(testSet, predictions)
    print("Accuracy:", accuracy1)

results = defaultdict(str)

i = 1
for folder in os.listdir(directory):
    results[i] = folder
    i += 1

print(results[nearestclass(getNeighbors(trainingSet, testSet[1], 5))])
