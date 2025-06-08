import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# 讀取CSV檔案
file_path = "SolarPrediction_time_aligned.csv"
data = pd.read_csv(file_path)

# 假設目標變數是 'target_variable'，請替換成實際的目標變數名稱
target_variable = 'target_variable'

# 選擇特徵變數（X）和目標變數（y）
X = data.drop(target_variable, axis=1)
y = data[target_variable]

# 切分訓練集和測試集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 初始化Random Forest回歸模型
model = RandomForestRegressor(n_estimators=100, random_state=42)

# 訓練模型
model.fit(X_train, y_train)

# 預測測試集
y_pred = model.predict(X_test)

# 評估模型性能
mse = mean_squared_error(y_test, y_pred)
print(f"均方誤差 (MSE): {mse}")
