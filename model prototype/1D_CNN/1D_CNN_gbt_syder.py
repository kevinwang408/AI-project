import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Dense, Flatten

# 讀取CSV檔案
file_path = "SolarRadiationPrediction.csv"
data = pd.read_csv(file_path)

# 假設目標變數是 'target_variable'，請替換成實際的目標變數名稱
target_variable = 'Radiation'

# 選擇特徵變數（X）和目標變數（y）
X = data.drop('Radiation', axis=1)
X=X.drop("Data",axis=1)
X=X.drop("Time",axis=1)
X=X.drop("TimeSunRise",axis=1)
X=X.drop("TimeSunSet",axis=1)
y = data[target_variable]

# 切分訓練集和測試集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 標準化特徵變數
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 轉換維度以符合1D CNN的輸入要求
X_train = np.expand_dims(X_train, axis=-1)
X_test = np.expand_dims(X_test, axis=-1)

# 初始化1D CNN模型
model = Sequential()
model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], 1)))
model.add(Flatten())
model.add(Dense(1, activation='linear'))

# 編譯模型
model.compile(optimizer='adam', loss='mean_squared_error')

# 訓練模型
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# 預測測試集
y_pred = model.predict(X_test)

# 評估模型性能
mse = mean_squared_error(y_test, y_pred)
print(f"均方誤差 (MSE): {mse}")
