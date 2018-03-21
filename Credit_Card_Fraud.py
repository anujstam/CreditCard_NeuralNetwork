'''
Basic program to play around with Keras and TF using a csv format dataset found on kaggle.

Made by Anuj Tambwekar
'''

# Importing modules
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from sklearn.model_selection import train_test_split

# Data found here: https://www.kaggle.com/mlg-ulb/creditcardfraud. Saved as credit_card_data.csv
filename = 'credit_card_data.csv'
dataset = pd.read_csv(filename).values

# Split training set and testing set (70%,30%)
y = dataset[:, 30]
x = dataset[:, 1:30]

xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.3)

# Creating the model
model = Sequential()
model.add(Dense(1000, input_shape=(29,)))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(100))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(10))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(1))
model.add(Activation('sigmoid'))

# need sparse otherwise shape is wrong. check why
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

print('Fitting the data to the model')
batch_size = 20
epochs = 7
history = model.fit(xtrain, ytrain, batch_size=batch_size, epochs=epochs, verbose=1, validation_split=0.1)
print('Evaluating the test data on the model')
score = model.evaluate(xtest, ytest, batch_size=batch_size, verbose=1)

# Test result
print('Test accuracy:', score[1]*100, "%")
