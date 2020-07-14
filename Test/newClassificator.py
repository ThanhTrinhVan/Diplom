import pandas as pd
import numpy as np
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dropout, Dense, Flatten
from keras.utils import to_categorical

pathToTrain = 'I:/GradEs/CompareNN_FL/MatlabProgram/DataTrainGenerated2.xlsx'
pathToTest = 'I:/GradEs/CompareNN_FL/MatlabProgram/TestGroupGenerated2.xlsx'

Ntrain = 1000 # Объем данных для обучения
print("Ntrain: ", Ntrain)
print("Load Train data: ")
dataTrain = pd.read_excel(pathToTrain, sheet_name=0, sep='delimiter',
                          header=None)
dataTrain = dataTrain.values[:Ntrain,:]
labelsTrain = np.repeat(0, Ntrain)

for sheet in range(1,8):
    ms = pd.read_excel(pathToTrain, sheet_name=sheet,
                       sep='delimiter', header=None)
    ms = ms.values[:Ntrain,:]
    dataTrain = np.append(dataTrain, ms)
    labelsTrain = np.append(labelsTrain, np.repeat(sheet, Ntrain))

dataTrain = dataTrain.reshape(Ntrain*8, 6, 4)
print("Example of matrix input: ", dataTrain[-1])
labelsTrain = to_categorical(labelsTrain)
print(labelsTrain)
print("Shape of input: ", dataTrain.shape)
print("Shape of labels: ", labelsTrain.shape)

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

dataTest = dataTest.reshape(Ntest*8, 6, 4)
print("Example of matrix input: ", dataTest[0])
labelsTest = to_categorical(labelsTest)
print(labelsTest)
print("Shape of input: ", dataTest.shape)
print("Shape of labels: ", labelsTest.shape)

model = Sequential()
model.add(Flatten(input_shape=(6,4)))
model.add(Dense(24, activation='relu'))
model.add(Dense(64, activation='relu'))
#model.add(Dense(128, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(8, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(dataTrain, labelsTrain, epochs=20, batch_size=20, verbose=1, validation_split=0.2)
test_loss, test_acc = model.evaluate(dataTest, labelsTest)
print('Acc: ', test_acc)