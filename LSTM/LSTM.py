import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pandas import read_csv
import math
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM,Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import predict


dataset=pd.read_csv('SolarPrediction_aligned_Sun.csv',engine='python',nrows=576*15)
dataset=dataset.drop("Data",axis=1)
dataset=dataset.drop("Time",axis=1)
# dataset=dataset.drop("TimeSunRise",axis=1)
# dataset=dataset.drop("TimeSunSet",axis=1)
#target=dataset["Radiation"]
dataset.head(5)



np.random.seed(7)

dataset=dataset.values

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

def create_dataset(dataset,look_back):
    dataX,dataY=[],[]
    for i in range(len(dataset)-look_back-1):
        #a=dataset第0行i~i+look_back個
        a=dataset[i:(i+look_back),0:9]
        dataX.append(a)
        dataY.append(dataset[i+look_back,1])
    return np.array(dataX),np.array(dataY)

    

    
#根據前面幾個資料來看下一步 彙整成look_back行的資料
look_back=5
trainX,trainY=create_dataset(train,look_back)
testX,testY=create_dataset(test,look_back)



model=Sequential()
#添加的一個 LSTM 層。
#這個層有一個名為 4 的參數，這表示該 LSTM 層有 4 個隱藏單元（hidden units）。
# input_shape=(look_back, 1) 則指定了輸入數據的形狀，
# 其中 look_back 是時間窗口的大小，1 表示每個時間步的特徵數。
model.add(LSTM(105,activation='relu',input_shape=(look_back,9)))
model.add(Dropout(0.3618106763168733))
#Dense全連接後輸出一層
model.add(Dense(1))
model.compile(loss='mean_squared_error',optimizer='adam')
model.summary()

loss_arr=[]
epochs=100
for i in range(epochs):
    history=model.fit(trainX,trainY,epochs=1,batch_size=3,verbose=2)
    loss_arr.append(history.history['loss'][0])
    trainPre=model.predict(trainX)
    testPre=model.predict(testX)

    trainPre=scalar2.inverse_transform(trainPre)
    trainY_ord=scalar2.inverse_transform([trainY])
    testPre=scalar2.inverse_transform(testPre)
    testY_ord=scalar2.inverse_transform([testY])
    
    trainScore=math.sqrt(mean_squared_error(trainY_ord[0,:],trainPre[:,0]))
    print('Train Score:%.2f RMSE'%(trainScore))
    testScore=math.sqrt(mean_squared_error(testY_ord[0,:],testPre[:,0]))
    print('Train Test:%.2f RMSE'%(testScore))
    MAE=mean_absolute_error(testY_ord[0,:],testPre[:,0])
    print('Test Score:%.2f MAE'%(MAE))

    trainPredictplot=np.empty_like(scalar_dim)
    trainPredictplot[:,:]=np.nan
    trainPredictplot[look_back:len(trainPre)+look_back,:]=trainPre

    testPredictPlot=np.empty_like(scalar_dim)
    testPredictPlot[:,:]=np.NaN
    testPredictPlot[len(trainPre)+(look_back*2)+1:len(dataset)-1,:]=testPre
    
    plt.subplot(311)
    plt.cla()
    #plt.plot(scalar2.inverse_transform(scalar_dim),label='Ground Truth')
    plt.plot(testY_ord[0,:],label='Ground Truth')
    #plt.plot(trainPredictplot,label='Prediction(training)')
    plt.plot(testPre[:,0],label='Prediction(testing)')
    #plt.plot(testPredictPlot,label='Prediction(testing)')
    plt.title('Epoch:'+str(i+1))
    plt.legend()
    plt.subplot(312)
    plt.cla()
    #plt.plot(scalar2.inverse_transform(scalar_dim))
    plt.plot(testY_ord[0,:])
    plt.plot(testPre[:,0],label='Prediction(testing)')
    #plt.plot(trainPredictplot)
    #plt.plot(testPredictPlot)
    plt.legend()
    plt.subplot(313)
    plt.cla()
    plt.plot(loss_arr,label='loss value')
    plt.title('learning curve')
    plt.legend()
    plt.show()
    plt.pause(0.01)
    plt.close()

predict.my_self(testY_ord[0,:],testPre[:,0],'LSTM')
predict.score_calculation(testY_ord[0,:],testPre[:,0])
predict.plot_pred(testY_ord[0,:],testPre[:,0],'LSTM')
predict.plot_residuals(testY_ord[0,:],testPre[:,0],'LSTM')
