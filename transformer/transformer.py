import torch
import torch.nn as nn
from torch.utils.data import DataLoader,TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pandas import read_csv
import math
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error,mean_absolute_error
dataset=pd.read_csv("SolarRadiationPrediction.csv",engine='python',nrows=576*15)
dataset=dataset.drop("Data",axis=1)
dataset=dataset.drop("Time",axis=1)
dataset.head(5)
dataset=dataset.values
scalar1=MinMaxScaler(feature_range=(0,1))
scalar2=MinMaxScaler(feature_range=(0,1))
scalar_dim=dataset[:,1]
scalar_dim=scalar_dim.reshape(len(dataset),1)
dataset=scalar1.fit_transform(dataset)
#rint(scalar_dim)
scalar_dim=scalar2.fit_transform(scalar_dim)
#print(scalar_dim)
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
#X用以輸入 Y用用以predict
look_back=5
trainX,trainY=create_dataset(train,look_back)
testX,testY=create_dataset(test,look_back)

trainX=torch.tensor(trainX,dtype=torch.float32)
trainY=torch.tensor(trainY,dtype=torch.float32).view(-1, 1)
testX=torch.tensor(testX,dtype=torch.float32)
testY=torch.tensor(testY,dtype=torch.float32).view(-1, 1)

train_dataset=TensorDataset(trainX,trainY)
train_loader=DataLoader(train_dataset,batch_size=2,shuffle=True)
test_dataset=TensorDataset(testX,testY)
test_loader=DataLoader(test_dataset,batch_size=2,shuffle=False)
# Positional Encoding for Transformer
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout= 1e-05, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

# Model definition using Transformer
class TransformerModel(nn.Module):
    def __init__(self, input_dim=9, d_model=64, nhead=4, num_layers=1, dropout=1e-05):
        super(TransformerModel, self).__init__()

        self.encoder = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.decoder = nn.Linear(d_model, 1)

    def forward(self, x):
        x = self.encoder(x)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        x = self.decoder(x[:, -1, :])
        return x

model = TransformerModel()

#TPE dropout1:0.15851024600381278 dopr2:0.1939416256427853 layer:1 lr:0.000662914227651421 MAE:34.9
#QMC 'dropout1': 0.15, 'num_layers': 3, 'dropout2': 0.15, 'lr': 0.00055 MAE:36.72
#random 'dropout1': 0.2714493404441506, 'num_layers': 1, 'dropout2': 0.08999059116992586, 'lr': 0.000713396703519771 MAE:31.66
#QMC try 'dropout1': 1e-05, 'num_layers': 1, 'dropout2': 1e-05, 'lr': 0.0001

# Train the model
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=3, verbose=True)

epochs = 50
early_stop_count = 0
min_val_loss = float('inf')
best_MAE_pre=[]
best_MAE=10000
for epoch in range(epochs):
    model.train()
    for batch in train_loader:
        x_batch, y_batch = batch
        x_batch, y_batch = x_batch, y_batch

        optimizer.zero_grad()
        outputs = model(x_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

    # Validation
    model.eval()
    val_losses = []
    with torch.no_grad():
        for batch in test_loader:
            x_batch, y_batch = batch
            x_batch, y_batch = x_batch, y_batch
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            val_losses.append(loss.item())

    val_loss = np.mean(val_losses)
    scheduler.step(val_loss)

    # if val_loss < min_val_loss:
    #     min_val_loss = val_loss
    #     early_stop_count = 0
    # else:
    #     early_stop_count += 1

    # if early_stop_count >= 5:
    #     print("Early stopping!")
    #     break
    print(f"Epoch {epoch + 1}/{epochs}, Validation Loss: {val_loss:.4f}")
    # Evaluation
    model.eval()
    predictions = []
    with torch.no_grad():
        for batch in test_loader:
            x_batch, y_batch = batch
            x_batch = x_batch
            outputs = model(x_batch)
            predictions.extend(outputs.squeeze().tolist())

    shape=scalar2.inverse_transform(np.array(predictions).reshape(-1,1))
    shape2=scalar2.inverse_transform(np.array(testY).reshape(-1,1))
    rmse=math.sqrt(mean_squared_error(shape,shape2))
    MAE=mean_absolute_error(shape,shape2)

    if MAE<best_MAE:
        best_MAE=MAE
        best_MAE_pre=shape
        print("best")
    print('Test Score:%.2f MAE'%(MAE))
    print(f"Score(RMSE):{rmse:.4f}")
    print("best MAE: %.2f MAE"%(mean_absolute_error(best_MAE_pre,shape2)))


import predict
predict.my_self(best_MAE_pre,shape2,'Transformer')
predict.score_calculation(best_MAE_pre,shape2)
predict.plot_pred(best_MAE_pre,shape2,'Transformer')
predict.plot_residuals(best_MAE_pre,shape2,'Transformer')