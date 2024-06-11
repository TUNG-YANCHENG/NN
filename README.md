# README

## 專案概述

此專案訓練一個淺層神經網絡來預測兩個隨機生成整數的總和。該模型使用反向傳播進行訓練，並且無使用任何的模型套件建立模型，並借助CuPy進行GPU加速。

## 目錄結構

- `Actual vs Predicted Values with Best Hidden Layer Size.png`：比較最佳隱藏層大小下實際值和預測值的可視化圖。
- `Backprogation artificial neature network-HW2-GPU.py`：包含神經網絡實現和訓練邏輯的主腳本。
- `Best Model Predictions vs True Values (train).png`：訓練數據集中最佳模型預測值與真實值的可視化圖。
- `Best Model Predictions vs True Values (val).png`：驗證數據集中最佳模型預測值與真實值的可視化圖。
- `best_train_model.npz`：最佳訓練模型的權重。
- `best_val_model.npz`：最佳驗證模型的權重。
- `dataset.csv`：包含訓練和驗證數據集的CSV文件。
- `Error Histogram (val).png`：驗證數據集的誤差直方圖。
- `Hidden Layer Size vs MSE.png`：隱藏層大小與均方誤差的圖表。
- `Learning Curve (train).png`：訓練數據集的學習曲線。
- `Learning Curve (val).png`：驗證數據集的學習曲線。
- `log.txt`：包含詳細訓練日誌的日誌文件。
- `Model Performance for Different Hidden Layer Sizes.png`：不同隱藏層大小下模型性能的箱形圖。
- `scaler_x0_scale.npy`，`scaler_x1_scale.npy`，`scaler_y_scale.npy`：包含輸入特徵和目標變量縮放參數的Numpy文件。
- `Test model.ipynb`：測試訓練模型的Jupyter notebook。

## 安裝

運行此專案需要安裝以下依賴項：

- Python 3.x
- CuPy
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- TQDM

您可以使用pip安裝所需的軟件包：

```bash
pip install numpy cupy pandas matplotlib scikit-learn tqdm
```

## 使用方法

- 生成和加載數據：

數據生成和加載過程在腳本中處理。數據集將從dataset.csv加載（如果存在），或者新生成並保存到文件中。

- 初始化和訓練模型：

神經網絡在ShallowNeuralNetwork類中實現。可以使用所需參數初始化模型，並使用train方法對其進行訓練。

- 評估模型：

使用各種指標和可視化圖評估模型性能，並將結果保存為PNG文件。

- 結果可視化：

生成的可視化圖有助於理解模型的性能以及不同隱藏層大小對均方誤差的影響。

## 超參數配置

超參數配置存儲在Config類中。以下是主要的超參數及其說明：

- class Config:
```bash
EPOCHS = 100                  # 訓練的總週期數

BATCH_SIZE = 64               # 每批訓練樣本數

NUM_SAMPLES = 16384           # 總樣本數

LEARNING_RATE = 0.001         # 學習率

MIN_RANGE = 1                 # 隱藏層神經元數量的最小值

MAX_RANGE = 20                # 隱藏層神經元數量的最大值

HIDDEN_SIZES = range(MIN_RANGE, MAX_RANGE + 1)  # 隱藏層神經元數量範圍

ROUNDS = 30                   # 每個隱藏層神經元數量的訓練回合數

PATIENCE = int(EPOCHS * 0.1)  # 提早停止的耐心值
```
## 主要函數和類

`generate\_data(num\_samples=10000)`

生成隨機整數及其總和的數據集。

`ShallowNeuralNetwork`

此類實現一個具有一個隱藏層的淺層神經網絡。主要方法包括：

`__init__(self, input_size, hidden_size, output_size, learning_rate=0.01, init_method='xavier')`：初始化網絡。

`initialize_weights(self, method)`：使用指定方法（Xavier或He）初始化權重。

`forward(self, x)`：執行前向傳播。

`predict(self, X)`：預測給定輸入的輸出。

`backward(self, x, y)`：執行反向傳播並更新權重。

`train(self, X, y, X_val, y_val, epochs=1, batch_size=1024, patience=10)`：使用提供的數據集訓練模型。

Config

保存訓練過程超參數的配置類。

## 訓練日誌

訓練過程的日誌保存在log.txt中，提供了每個訓練時期的詳細信息，包括訓練和驗證損失，以及任何提前停止事件。

## 結果

最佳模型配置及其性能可視化並保存為PNG文件。這些可視化圖包括學習曲線、誤差直方圖以及實際值和預測值的比較。

