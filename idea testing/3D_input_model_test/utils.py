#evaluate the model's performance
import numpy as np
import math
from sklearn.metrics import mean_squared_error, mean_absolute_error

def inverse_transform_and_evaluate(scaler, y_true, y_pred):
    y_pred_inv = scaler.inverse_transform(y_pred)
    y_true_inv = scaler.inverse_transform([y_true])
    
    rmse = math.sqrt(mean_squared_error(y_true_inv[0], y_pred_inv[:, 0]))
    mae = mean_absolute_error(y_true_inv[0], y_pred_inv[:, 0])
    
    print(f'RMSE: {rmse:.2f}, MAE: {mae:.2f}')
    return y_true_inv[0], y_pred_inv[:, 0], rmse, mae
