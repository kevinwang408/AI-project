from sklearn import svm

def build_SVR():
    model = svm.SVR(kernel='linear', C=0.10127678320148709, epsilon=0.029240378785064282, gamma='auto')
    return model

def build_and_predict(trainX,testX,trainY):
    trainX = trainX.reshape(trainX.shape[0], -1)
    testX = testX.reshape(testX.shape[0], -1)
    # build and train the model
    model = build_SVR()
    model.fit(trainX, trainY)
    # prediction and evaluation
    trainPred = model.predict(trainX).reshape(-1, 1)
    testPred = model.predict(testX).reshape(-1, 1)
    return trainPred, testPred
    

    
