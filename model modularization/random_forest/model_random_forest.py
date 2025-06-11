from sklearn.ensemble import RandomForestRegressor

def build_random_forest():
    model = model = RandomForestRegressor(
        n_estimators=152,
        max_depth=4,
        max_features='sqrt',
        random_state=42
    )
    return model