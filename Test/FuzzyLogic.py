import pandas as pd
import numpy as np
from math import pow, sqrt

pathToTrain = 'I:/GradEs/CompareNN_FL/MatlabProgram/DataTrainGenerated2.xlsx'
pathToTest = 'I:/GradEs/CompareNN_FL/MatlabProgram/TestGroupGenerated2.xlsx'

Ntrain = 20 # Объем данных для обучения
print("Ntrain: ", Ntrain)
print("Load Train data: ")
dataTrain = pd.read_excel(pathToTrain, sheet_name=0, sep='delimiter',
                          header=None)
dataTrain = dataTrain.values[:Ntrain,:]
dataTrain = np.sum(dataTrain, axis=0)
dataTrain = dataTrain/Ntrain
#print(dataTrain)

for sheet in range(1,8):
    ms = pd.read_excel(pathToTrain, sheet_name=sheet,
                       sep='delimiter', header=None)
    ms = ms.values[:Ntrain,:]
    ms = np.sum(ms, axis=0)
    ms = ms/Ntrain
    dataTrain = np.append(dataTrain, ms)

dataTrain = dataTrain.reshape(8, 24)
print(dataTrain.shape)

print("Load test data: ")
Ntest = 40 # Number matrix for 1 disease
dataTest = pd.read_excel(pathToTest, sheet_name=0, sep='delimiter',
                          header=None)
dataTest = dataTest.values[:Ntest,:]
labelsTest = np.repeat(0, Ntest)

for sheet in range(1,8):
    ms = pd.read_excel(pathToTest, sheet_name=sheet,
                       sep='delimiter', header=None)
    ms = ms.values[:Ntest,:]
    dataTest = np.append(dataTest, ms)
    labelsTest = np.append(labelsTest, np.repeat(sheet, Ntest))

dataTest = dataTest.reshape(Ntest*8, 24)
print(dataTest.shape)
#print("Example of matrix input: ", dataTest[0])
#print("Shape of labels: ", labelsTest)

sA = np.sum(dataTrain, axis=1) / 24
sB = np.sum(dataTest, axis=1) / 24
#print("sA: ", sA)
#print("sB: ", sB)
skoA = np.repeat(0.0, 8)
for i in range(0,8):
    sko = 0
    for j in range(0,24):
        sko += pow(dataTrain[i][j]-sA[i], 2)
    skoA[i] = sko

skoB = np.repeat(0.0, Ntest*8)
for i in range(0, Ntest*8):
    sko = 0
    for j in range(0,24):
        sko += pow(dataTest[i][j]-sB[i], 2)
    skoB[i] = sko

#print("skoA: ", skoA)
#print("skoB: ", skoB)

nCorrect = 0
for idTest in range(0, Ntest*8):
    mF = np.repeat(0.0, 8)
    for i in range(0,8):
        for j in range(0,24):
            mF[i] += ((dataTrain[i][j]-sA[i])*(dataTest[idTest][j]-sB[idTest]))
        mF[i] = mF[i]/(sqrt(skoA[i]*skoB[idTest]))

    #print("Membership function: ", mF)
    nDisease = np.argmax(mF)
    #print(nDisease)
    #print(labelsTest[idTest])
    if (nDisease == labelsTest[idTest]):
        nCorrect = nCorrect + 1

print("Accuracy: ", nCorrect/(Ntest*8))