from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout

def build_GRU(input_shape):
    # ==============
    # Build GRU model
    # ==============
    model = Sequential()
    model.add(GRU(units=193, activation='tanh', input_shape=input_shape))
    model.add(Dense(units=64, activation="relu"))
    model.add(Dropout(rate=0.082379383823238))
    model.add(Dense(units=1, activation="linear"))  # Output layer for regression
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.summary()
    return model
