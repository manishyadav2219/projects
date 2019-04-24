import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset_train = pd.read_csv('Google_Stock_Price_Train.csv')
trainset = dataset_train.iloc[:,1:2].values

from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler()
trainset_scaled = sc.fit_transform(trainset)

x_train =[]
y_train =[]
for i in range(60,1258):
    x_train.append(trainset_scaled[i-60:i,0])
    y_train.append(trainset_scaled[i,0])
x_train,y_train =np.array(x_train),np.array(y_train)

x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))

#building RNN
from keras.models import Sequential
from keras.layers import Dropout
from keras.layers import Dense
from keras.layers import LSTM

regr = Sequential()
regr.add(LSTM(units =50, return_sequences =True , input_shape =(x_train.shape[1],1)))
regr.add(Dropout(0.2))

regr.add(LSTM(units =50, return_sequences =True) )
regr.add(Dropout(0.2))

regr.add(LSTM(units =50, return_sequences =True ))
regr.add(Dropout(0.2))

regr.add(LSTM(units =50, return_sequences =False))
regr.add(Dropout(0.2))

#output ayer
regr.add(Dense(units =1))

regr.compile(optimizer = 'adam' ,loss = 'mean_squared_error')
regr.fit(x_train,y_train,batch_size =32,epochs =100)


#predicting 
dataset_test = pd.read_csv('Google_Stock_Price_Test.csv')
testset = dataset_test.iloc[:,1:2].values
dataset = pd.concat((dataset_train['Open'],dataset_test['Open']) , axis =0)
inputs = dataset[len(dataset)-len(testset)-60:].values

inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)

x_test =[]

for i in range(60,80):
    x_test.append(inputs[i-60:i,0])
    
x_test =np.array(x_test)
x_test = np.reshape(x_test,(x_test.shape[0],x_test.shape[1],1))
prediction1 = sc.inverse_transform(regr.predict(x_test))

import math
from sklearn.metrics import mean_squared_error
rmse_rmsprop = math.sqrt(mean_squared_error(testset, prediction))

rmse_adam = math.sqrt(mean_squared_error(testset, prediction1))

#visualisation

plt.plot(testset,color = 'red')
plt.plot(prediction,color = 'blue',label = 'predicted')
plt.plot(prediction1,color = 'green',label = 'predicted-adam')
plt.legend()
plt.xlabel('Time')
plt.ylabel('Stock price')
plt.show()

