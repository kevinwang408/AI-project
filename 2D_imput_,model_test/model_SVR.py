from sklearn import svm

def build_SVR():
    model = svm.SVR(kernel='linear', C=0.10127678320148709, epsilon=0.029240378785064282, gamma='auto')
    return model