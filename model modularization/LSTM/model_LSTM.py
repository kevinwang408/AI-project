from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout

def build_LSTM(input_shape):
    # =====================
    # Build LSTM Network
    # =====================
    model = Sequential()
    model.add(LSTM(105, activation='relu', input_shape=input_shape))
    model.add(Dropout(0.3618106763168733))
    model.add(Dense(1))  # Output layer
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.summary()
    return model