# 🌴 Hawaii Solar Irradiance Forecasting (AI‑Project)

对夏威夷環境資料（例如氣溫、濕度、風速、大氣壓力和歷史輻射）進行 **1 時步太陽能輻射預測**，並使用多種模型（1D-CNN、GRU、LSTM、TCN、Transformer、MLP、SVR、Regression Tree、Random Forest Regression）比較性能，找出最有效的模型。

---

## 🚀 功能亮點

- 模組化架構，方便擴展
- 支援多種深度學習與傳統回歸模型
- 專為時間序列設計，預測下一時間點的輻射值
- 提供詳細的性能評估與比較

---

## 📁 專案結構（示例）

AI-project/
├── data/ # 原始與清洗後資料
├── models/ # 各模型程式碼
│ ├── cnn.py
│ ├── gru.py
│ ├── lstm.py
│ ├── tcn.py
│ ├── transformer.py
│ ├── mlp.py
│ ├── svr.py
│ ├── tree.py
│ └── forest.py
├── utils/ # 工具函式，例如資料處理、評估、畫圖
│ ├── data_loader.py
│ ├── metrics.py
│ └── plot_utils.py
├── results/ # 訓練結果、圖表與 log
├── main.py # 主程式，支援 --model、--compare 等選項
├── requirements.txt
└── README.md


---

## 📦 安裝與環境準備

1. Clone repo：
   ```bash
   git clone https://github.com/kevinwang408/AI-project.git
   cd AI-project

2. 建議使用虛擬環境並安裝依賴：
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt

⚙️ 使用方式
單模型訓練與測試：
    ```bash
    python main.py --model transformer --epochs 50 --batch_size 32
批量比較多個模型：
    ```bash
    python main.py --compare all --epochs 30

常用參數列表
參數	說明	範例
--model	選擇模型名稱（cnn / gru / lstm / tcn / transformer / mlp / svr / tree / forest）	--model lstm
--compare	選擇同時比較多個模型，可設 all 或逗號分隔列表	--compare cnn,gru,svr
--epochs	訓練 epochs 數量	--epochs 50
--batch_size	訓練批次大小	--batch_size 64

📊 評估指標
每個模型會計算並輸出以下指標：

均方根誤差 (RMSE)

平均絕對誤差 (MAE)

R² 決定係數

訓練時間

預測時間（可選）

結果與圖表會儲存在 results/ 目錄中。

📈 範例結果（視實際輸出調整）
模型	RMSE	MAE	R²	訓練時間
1D-CNN	15.2	11.3	0.88	45 s
LSTM	14.8	10.9	0.89	60 s
GRU	14.5	10.7	0.90	55 s
Transformer	13.1	9.8	0.92	80 s
Random Forest	16.5	12.1	0.85	20 s

Transformer 模型在整體預測上表現最佳，RMSE 和 MAE 均最低，R² 最高

🧩 如何加入新模型
在 models/ 資料夾中新增 .py 模組，實現 train()、predict()、evaluate()。

更新 main.py：將模型加入如下字典：

python
複製
編輯
MODELS = {
    'cnn': CNN(),
    'lstm': LSTM(),
    # 新增
    'my_model': MyModel(),
}
呼叫方式：

bash
複製
編輯
python main.py --model my_model
📚 參考文獻與資料來源
使用的夏威夷環境資料集說明來源（請補上來源連結）

時間序列預測與模型比較方法參考文獻

各模型實現參考 (e.g. Transformer 時序預測架構相關論文)

🧑‍💻 作者與聯絡方式
作者：Kevin Wang

Github：kevinwang408

Email：talktalkai.kevin@gmail.com

📄 License
本專案遵循 MIT License，詳見 LICENSE 文件。

✅ 建議後續改進
增加 cross-validation 比較模型穩健性

增補可視化儀錶板（如 Plotly、Dash 或 Streamlit）

擴大預測 horizon 為多步預測

嘗試 ensemble 模型提升性能

