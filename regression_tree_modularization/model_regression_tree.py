from sklearn.tree import DecisionTreeRegressor

def build_regression_tree():
    model = DecisionTreeRegressor(
        max_depth=5,
        min_samples_split=18,
        min_samples_leaf=20,
        max_features='sqrt'  # For regression, this means sqrt(n_features)
    )
    return model