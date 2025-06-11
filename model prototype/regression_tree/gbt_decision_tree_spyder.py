# 引入必要的庫
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

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

# 初始化決策樹回歸模型
model = DecisionTreeRegressor()

# 訓練模型
model.fit(X_train, y_train)

# 預測測試集
y_pred = model.predict(X_test)

# 評估模型性能
mse = mean_squared_error(y_test, y_pred)
print(f"均方誤差 (MSE): {mse}")

# 可以進一步分析模型的特徵重要性
feature_importances = model.feature_importances_
features = X.columns
feature_df = pd.DataFrame({'Feature': features, 'Importance': feature_importances})
feature_df = feature_df.sort_values(by='Importance', ascending=False)

# 顯示前10個重要特徵
print("前10個重要特徵：")
print(feature_df.head(10))

# 如果需要視覺化決策樹，可以使用以下程式碼
# from sklearn.tree import plot_tree
# plt.figure(figsize=(20, 10))
# plot_tree(model, filled=True, feature_names=X.columns, rounded=True, fontsize=10)
# plt.show()
