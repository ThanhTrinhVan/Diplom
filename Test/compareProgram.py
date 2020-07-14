import pandas as pd
import numpy as np
from math import pow, sqrt
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dropout, Dense, Flatten
from keras.utils import to_categorical

pathToTrain = 'I:/GradEs/CompareNN_FL/MatlabProgram/DataTrainGenerated2.xlsx'
pathToTest = 'I:/GradEs/CompareNN_FL/MatlabProgram/TestGroupGenerated2.xlsx'

def NeuralNetwork(Ntrain, Ntest, pathToTrain, pathToTest):
    #print("Ntrain: ", Ntrain)
    #print("Load Train data: ")
    dataTrain = pd.read_excel(pathToTrain, sheet_name=0, sep='delimiter',
                              header=None)
    dataTrain = dataTrain.values[:Ntrain, :]
    labelsTrain = np.repeat(0, Ntrain)

    for sheet in range(1, 8):
        ms = pd.read_excel(pathToTrain, sheet_name=sheet,
                           sep='delimiter', header=None)
        ms = ms.values[:Ntrain, :]
        dataTrain = np.append(dataTrain, ms)
        labelsTrain = np.append(labelsTrain, np.repeat(sheet, Ntrain))

    dataTrain = dataTrain.reshape(Ntrain * 8, 6, 4)
    #print("Example of matrix input: ", dataTrain[-1])
    labelsTrain = to_categorical(labelsTrain)
    #print(labelsTrain)
    #print("Shape of input: ", dataTrain.shape)
    #print("Shape of labels: ", labelsTrain.shape)

    #print("Load test data: ")
    dataTest = pd.read_excel(pathToTest, sheet_name=0, sep='delimiter',
                             header=None)
    dataTest = dataTest.values[:Ntest, :]
    labelsTest = np.repeat(0, Ntest)

    for sheet in range(1, 8):
        ms = pd.read_excel(pathToTest, sheet_name=sheet,
                           sep='delimiter', header=None)
        ms = ms.values[:Ntest, :]
        dataTest = np.append(dataTest, ms)
        labelsTest = np.append(labelsTest, np.repeat(sheet, Ntest))

    dataTest = dataTest.reshape(Ntest * 8, 6, 4)
    #print("Example of matrix input: ", dataTest[0])
    labelsTest = to_categorical(labelsTest)
    #print(labelsTest)
    #print("Shape of input: ", dataTest.shape)
    #print("Shape of labels: ", labelsTest.shape)

    model = Sequential()
    model.add(Flatten(input_shape=(6, 4)))
    model.add(Dense(24, activation='relu'))
    model.add(Dense(64, activation='relu'))
    # model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(8, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(dataTrain, labelsTrain, epochs=20, batch_size=20, verbose=1, validation_split=0.2)
    test_loss, test_acc = model.evaluate(dataTest, labelsTest)
    return test_acc

def FuzzyLogic(Ntrain, Ntest, pathToTrain, pathToTest):
    #print("Ntrain: ", Ntrain)
    #print("Load Train data: ")
    dataTrain = pd.read_excel(pathToTrain, sheet_name=0, sep='delimiter',
                              header=None)
    dataTrain = dataTrain.values[:Ntrain, :]
    dataTrain = np.sum(dataTrain, axis=0)
    dataTrain = dataTrain / Ntrain
    # print(dataTrain)

    for sheet in range(1, 8):
        ms = pd.read_excel(pathToTrain, sheet_name=sheet,
                           sep='delimiter', header=None)
        ms = ms.values[:Ntrain, :]
        ms = np.sum(ms, axis=0)
        ms = ms / Ntrain
        dataTrain = np.append(dataTrain, ms)

    dataTrain = dataTrain.reshape(8, 24)
    #print(dataTrain.shape)

    #print("Load test data: ")
    dataTest = pd.read_excel(pathToTest, sheet_name=0, sep='delimiter',
                             header=None)
    dataTest = dataTest.values[:Ntest, :]
    labelsTest = np.repeat(0, Ntest)

    for sheet in range(1, 8):
        ms = pd.read_excel(pathToTest, sheet_name=sheet,
                           sep='delimiter', header=None)
        ms = ms.values[:Ntest, :]
        dataTest = np.append(dataTest, ms)
        labelsTest = np.append(labelsTest, np.repeat(sheet, Ntest))

    dataTest = dataTest.reshape(Ntest * 8, 24)
    # print(dataTest.shape)
    # print("Example of matrix input: ", dataTest[0])
    # print("Shape of labels: ", labelsTest)

    sA = np.sum(dataTrain, axis=1) / 24
    sB = np.sum(dataTest, axis=1) / 24
    # print("sA: ", sA)
    # print("sB: ", sB)
    skoA = np.repeat(0.0, 8)
    for i in range(0, 8):
        sko = 0
        for j in range(0, 24):
            sko += pow(dataTrain[i][j] - sA[i], 2)
        skoA[i] = sko

    skoB = np.repeat(0.0, Ntest * 8)
    for i in range(0, Ntest * 8):
        sko = 0
        for j in range(0, 24):
            sko += pow(dataTest[i][j] - sB[i], 2)
        skoB[i] = sko

    # print("skoA: ", skoA)
    # print("skoB: ", skoB)

    nCorrect = 0
    for idTest in range(0, Ntest * 8):
        mF = np.repeat(0.0, 8)
        for i in range(0, 8):
            for j in range(0, 24):
                mF[i] += ((dataTrain[i][j] - sA[i]) * (dataTest[idTest][j] - sB[idTest]))
            mF[i] = mF[i] / (sqrt(skoA[i] * skoB[idTest]))

        # print("Membership function: ", mF)
        nDisease = np.argmax(mF)
        # print(nDisease)
        # print(labelsTest[idTest])
        if (nDisease == labelsTest[idTest]):
            nCorrect = nCorrect + 1

    return (nCorrect / (Ntest * 8))

import matplotlib.pyplot as plt
Ntest  = 40

NtrainNN = [] # Объем данных для обучения
NtrainFL = []
accuracyNN = []
accuracyFL = []
for n in range(5, 100, 5):
    NtrainFL.append(n)
    tmp = FuzzyLogic(n, Ntest, pathToTrain, pathToTest)
    accuracyFL.append(tmp)
for n in range(5, 250, 50):
    NtrainNN.append(n)
    tmp = NeuralNetwork(n, Ntest, pathToTrain, pathToTest)
    accuracyNN.append(tmp)

print(NtrainNN)
print(accuracyNN)
print(NtrainFL)
print(accuracyFL)
plt.figure(1)
plt.plot(NtrainNN, accuracyNN, label='Neural Network')
plt.xlabel('Объем обучения')
plt.ylabel('Точность')
plt.title('Зависимость точности от объема обучения при СКО = 0.06')
plt.legend()
plt.figure(2)
plt.plot(NtrainFL, accuracyFL, label='Fuzzy Logic')
plt.xlabel('Объем обучения')
plt.ylabel('Точность')
plt.title('Зависимость точности от объема обучения при СКО = 0.06')
plt.legend()
plt.show()