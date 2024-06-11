# XOR 神經網路訓練

此專案展示了使用倒傳遞法訓練一個淺層神經網路來解決 XOR 問題。實作中包含了 Adam 優化器的使用，並針對不同的隱藏層大小進行模型的訓練和評估。

## 專案結構

.
├── Actual vs Predicted Values with Best Hidden Layer Size.png # 使用最佳隱藏層大小的實際值與預測值的散點圖
├── best_train_model.npz # 保存的最佳訓練模型
├── best_val_model.npz # 保存的最佳驗證模型
├── dataset.csv # XOR 數據集
├── dataset distribute graph.png # 數據集分佈圖
├── Error Histogram (val).png # 驗證集的預測誤差直方圖
├── Hidden Layer Size vs MSE.png # 隱藏層大小與均方誤差的圖表
├── Learning Curve (train).png # 訓練集的學習曲線
├── Learning Curve (val).png # 驗證集的學習曲線
├── log.txt # 訓練日誌文件
├── Model Performance for Different Hidden Layer Sizes.png # 不同隱藏層大小的模型性能箱型圖
├── scaler_x0_mean.npy # 標準化縮放器的均值（特徵x0）
├── scaler_x0_scale.npy # 標準化縮放器的縮放值（特徵x0）
├── scaler_x1_mean.npy # 標準化縮放器的均值（特徵x1）
├── scaler_x1_scale.npy # 標準化縮放器的縮放值（特徵x1）
├── test_model.ipynb # 測試模型的 Jupyter Notebook
└── XOR_train.py # 訓練模型的 Python 腳本


## 需求

此專案需要以下的庫：
- numpy
- cupy
- tqdm
- matplotlib
- scikit-learn
- pandas
- logging

## 執行方式

1. 確保已安裝所有需求庫。
2. 執行 `XOR_train.py` 腳本進行模型訓練。此腳本將：
   - 生成或加載 XOR 數據集。
   - 初始化並訓練不同隱藏層大小的淺層神經網路。
   - 保存最佳的訓練和驗證模型。
   - 繪製並保存各種性能指標和圖表。

## 函式說明

### `generate_data(num_samples=10000)`
生成隨機數據，並計算 XOR 運算。

### `check_and_save_data(csv_path, num_samples)`
檢查是否存在數據集文件，若不存在則生成新數據並保存。

### `ShallowNeuralNetwork` 類
實作了淺層神經網路，包含以下方法：
- `__init__(self, input_size, hidden_size, output_size, learning_rate=0.01)`：初始化網路結構及參數。
- `initialize_weights(self)`：初始化權重和偏置。
- `initialize_adam_parameters(self)`：初始化 Adam 優化器參數。
- `forward(self, x)`：前向傳播計算輸出。
- `predict(self, X)`：預測輸出。
- `backward(self, x, y)`：反向傳播計算梯度。
- `update_weights(self, dW1, db1, dW2, db2)`：使用 Adam 優化器更新權重。
- `train(self, X, y, X_val, y_val, epochs=1, batch_size=1024, patience=10)`：訓練模型。
- `save(self, file_path)`：保存模型參數。
- `load(self, file_path)`：加載模型參數。

### `Config` 類
包含訓練過程的超參數配置：
- `EPOCHS`：訓練週期數（預設為 100）。
- `BATCH_SIZE`：批次大小（預設為 16）。
- `NUM_SAMPLES`：樣本數量（預設為 16384）。
- `LEARNING_RATE`：學習率（預設為 0.1）。
- `MIN_RANGE`：隱藏層大小最小值（預設為 2）。
- `MAX_RANGE`：隱藏層大小最大值（預設為 2）。
- `HIDDEN_SIZES`：隱藏層大小範圍（預設為 [2, 3, ..., 10]）。
- `ROUNDS`：訓練輪數（預設為 30）。
- `PATIENCE`：早停耐心度（預設為 50）。
  
## 配置

`Config` 類包含所有訓練過程的配置參數，包括：
- 學習率
- 訓練週期數
- 批次大小
- 樣本數量
- 隱藏層大小範圍
- 訓練輪數
- 早停耐心度
  
## 訓練過程

訓練過程包括：
- 將數據集拆分為訓練集和驗證集
- 正規化輸入特徵
- 使用不同的隱藏層大小進行模型訓練
- 根據驗證損失進行早停
- 訓練過程記錄到 `log.txt` 文件中

## 結果

以下結果作為圖像保存到專案目錄中：
- `Model Performance for Different Hidden Layer Sizes.png`：不同隱藏層大小的模型性能箱型圖。
- `Actual vs Predicted Values with Best Hidden Layer Size.png`：使用最佳隱藏層大小的實際值與預測值的散點圖。
- `Learning Curve (train).png` 和 `Learning Curve (val).png`：訓練集和驗證集的學習曲線。
- `Error Histogram (val).png`：驗證集的預測誤差直方圖。
- `Hidden Layer Size vs MSE.png`：隱藏層大小與均方誤差的圖表。

## 日誌

所有的訓練日誌保存於 `log.txt`。

## 模型保存與加載

最佳的訓練和驗證模型分別保存為 `best_train_model.npz` 和 `best_val_model.npz`。`ShallowNeuralNetwork` 類中包括保存和加載模型的方法。

