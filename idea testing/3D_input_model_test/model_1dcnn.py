#build and train 1D CNN model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense

def build_1dcnn(input_shape):
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(1, activation='linear'))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def build_predict(LOOK_BACK,INPUT_DIM,trainX,trainY,testX,testY):
    model = build_1dcnn(input_shape=(LOOK_BACK, INPUT_DIM))
    model.fit(trainX, trainY, epochs=5, batch_size=21, validation_data=(testX, testY))

    # prediction and evaluation
    trainPred = model.predict(trainX)
    testPred = model.predict(testX)
    return trainPred, testPred