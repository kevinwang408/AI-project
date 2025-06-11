#divide dataset into time seqences

import numpy as np

def split_dataset(dataset, ratio=0.67):
    train_size = int(len(dataset) * ratio)
    train, test = dataset[:train_size], dataset[train_size:]
    return train, test

def create_dataset(dataset, look_back=5, input_dim=9, target_index=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back - 1):
        x = dataset[i:(i + look_back), :input_dim]
        y = dataset[i + look_back, target_index]
        dataX.append(x)
        dataY.append(y)
    return np.array(dataX), np.array(dataY)
