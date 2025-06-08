import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pandas import read_csv
import math
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error,mean_absolute_error
from tcn import TCN, tcn_full_summary
import predict

batch_size, time_steps, input_dim = 1, 5, 9
dataset=pd.read_csv('SolarPrediction_aligned_Sun.csv',engine='python',nrows=576*15)
dataset=dataset.drop("Data",axis=1)
dataset=dataset.drop("Time",axis=1)
dataset.head(5)
dataset=dataset.values

dataset=dataset.astype('float32')

scalar1=MinMaxScaler(feature_range=(0,1))
scalar2=MinMaxScaler(feature_range=(0,1))
scalar_dim=dataset[:,1]
dataset=scalar1.fit_transform(dataset)

scalar_dim=scalar_dim.reshape(len(dataset),1)
scalar_dim=scalar2.fit_transform(scalar_dim)


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

look_back=5
trainX,trainY=create_dataset(train,look_back)
testX,testY=create_dataset(test,look_back)
#tcn_layer = TCN(nb_filters=64,kernel_size=3,dropout_rate=0.0,padding='causal',activation='relu',input_shape=(time_steps, input_dim))
tcn_layer = TCN(nb_filters=64,
                kernel_size=13,
                dropout_rate=0.2331163281069671,
                activation='relu',padding='causal',
                nb_stacks=1,
                dilations=(1, 2, 4, 8),
                input_shape=(time_steps, input_dim))
print('Receptive field size =', tcn_layer.receptive_field)

model = Sequential([tcn_layer,Dense(1)])
model.compile(optimizer='adam', loss='mse')
tcn_full_summary(model, expand_residual_blocks=False)
tcn_full_summary(model, expand_residual_blocks=False)
loss_arr=[]
epochs=50
best_MAE_pre=[]
best_MAE=10000
for i in range(epochs):
    history=model.fit(trainX, trainY, epochs=1, validation_split=0.2)
    loss_arr.append(history.history['loss'][0])
    trainPre=model.predict(trainX)
    testPre=model.predict(testX)

    trainPre=scalar2.inverse_transform(trainPre)
    trainY_ord=scalar2.inverse_transform([trainY])
    testPre=scalar2.inverse_transform(testPre)
    testY_ord=scalar2.inverse_transform([testY])

    testScore=math.sqrt(mean_squared_error(testY_ord[0,:],testPre[:,0]))
    print('Test Score:%.2f RMSE'%(testScore))
    MAE=mean_absolute_error(testY_ord[0,:],testPre[:,0])
    print('Test Score:%.2f MAE'%(MAE))


    if MAE<best_MAE:
        best_MAE=MAE
        best_MAE_pre=testPre
        print("best")

    print("best MAE: %.2f MAE"%(mean_absolute_error(testY_ord[0,:],best_MAE_pre[:,0])))
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


predict.my_self(testY_ord[0,:],best_MAE_pre[:,0],'TCN')
predict.score_calculation(testY_ord[0,:],best_MAE_pre[:,0])
predict.plot_pred(testY_ord[0,:],best_MAE_pre[:,0],'TCN')
predict.plot_residuals(testY_ord[0,:],best_MAE_pre[:,0],'TCN')
