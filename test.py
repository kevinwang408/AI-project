import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pandas import read_csv
import math
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

dataset=pd.read_csv('SolarPrediction_time_aligned.csv',engine='python')
dataset=dataset.drop("Data",axis=1)
dataset=dataset.drop("Time",axis=1)
dataset=dataset.drop("TimeSunRise",axis=1)
dataset=dataset.drop("TimeSunSet",axis=1)
dataset.head(5)

dataset=dataset.values



np.random.seed(7)


dataset=dataset.astype('float32')

scalar1=MinMaxScaler(feature_range=(0,1))
scalar2=MinMaxScaler(feature_range=(0,1))
scalar_dim=dataset[:,1]
dataset=scalar1.fit_transform(dataset)

scalar_dim=scalar_dim.reshape(len(dataset),1)
print(scalar_dim)
scalar_dim=scalar2.fit_transform(scalar_dim)
print(scalar_dim)

train_size=int(len(dataset)*0.67)
test_size=len(dataset)- train_size
train,test=dataset[0:train_size,:],dataset[train_size:len(dataset),:]

print(train.shape)
print(train)
print(test.shape)
#print(test)

def create_dataset(dataset,look_back):
    dataX,dataY=[],[]
    for i in range(len(dataset)-look_back-1):
        #a=dataset第0行i~i+look_back個
        a=dataset[i:(i+look_back),0:7]
        dataX.append(a)
        dataY.append(dataset[i+look_back,1])
    return np.array(dataX),np.array(dataY)

#根據前面幾個資料來看下一步 彙整成look_back行的資料
#X用以輸入 Y用用以predict
look_back=5
trainX,trainY=create_dataset(train,look_back)
testX,testY=create_dataset(test,look_back)
print(testX.shape)
#print(testX)
print(testY.shape)
print(trainX.shape)
print(trainY.shape)

model=Sequential()
#添加的一個 LSTM 層。
#這個層有一個名為 4 的參數，這表示該 LSTM 層有 4 個隱藏單元（hidden units）。
# input_shape=(look_back, 1) 則指定了輸入數據的形狀，
# 其中 look_back 是時間窗口的大小，1 表示每個時間步的特徵數。
model.add(LSTM(4,input_shape=(look_back,7)))
#Dense全連接後輸出一層
model.add(Dense(1))
model.compile(loss='mean_squared_error',optimizer='adam')
model.summary()
loss_arr=[]
epochs=20
for i in range(epochs):
    history=model.fit(trainX,trainY,epochs=1,batch_size=1,verbose=2)
    loss_arr.append(history.history['loss'][0])
    
    trainPre=model.predict(trainX)
    testPre=model.predict(testX)
    
    trainPre=scalar2.inverse_transform(trainPre)
    trainY=scalar2.inverse_transform([trainY])
    testPre=scalar2.inverse_transform(testPre)
    testY=scalar2.inverse_transform([testY])
    
    #MSE是均方根誤差 即平均誤差 會被平均除下去 所以數據多不代表誤差會增加
    trainScore=math.sqrt(mean_squared_error(trainY[0,:],trainPre[:,0]))
    print('Train Score:%.2f RMSE'%(trainScore))
    testScore=math.sqrt(mean_squared_error(testY[0,:],testPre[:,0]))
    print('Test Score:%.2f RMSE'%(testScore))
    
    trainPredictplot=np.empty_like(scalar_dim)
    #print(trainPredictplot)
    trainPredictplot[:,:]=np.nan
    #print(trainPredictplot)
    trainPredictplot[look_back:len(trainPre)+look_back,:]=trainPre

    
    testPredictPlot=np.empty_like(scalar_dim)
    testPredictPlot[:,:]=np.NaN
    testPredictPlot[len(trainPre)+(look_back*2)+1:len(dataset)-1,:]=testPre
    ground_truth=scalar2.inverse_transform(scalar_dim)
    
    plt.subplot(311)
    plt.cla()
    plt.plot(ground_truth,label='Ground Truth')
    plt.plot(trainPredictplot,label='Prediction(training)')
    plt.plot(testPredictPlot,label='Prediction(testing)')
    plt.title('Epoch:'+str(i))
    plt.legend()
    plt.subplot(312)
    plt.cla()
    plt.plot(scalar1.inverse_transform(dataset))
    plt.plot(trainPredictplot)
    plt.plot(testPredictPlot)
    plt.legend()
    plt.subplot(313)
    plt.cla()
    plt.plot(loss_arr,label='loss value')
    plt.title('learning curve')
    plt.legend()
    plt.show()
    plt.pause(0.01)
    plt.close()
    
    
    