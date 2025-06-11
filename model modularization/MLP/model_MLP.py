from sklearn.neural_network import MLPRegressor

def build_MLP():
    model = MLPRegressor(hidden_layer_sizes=(4, 210), max_iter=328, alpha=0.0201293522716579, random_state=42)
    return model