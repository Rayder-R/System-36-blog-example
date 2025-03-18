import os
import numpy as np
import tensorflow as tf
import keras

# 設定亂數種子，確保結果一致
np.random.seed(10)

# 讀取 MNIST 資料集
(train_feature, train_label), (test_feature, test_label) = keras.datasets.mnist.load_data()

# 轉換為 4D 張量 (批次大小, 高度, 寬度, 通道數)
train_feature_vector = train_feature.reshape(-1, 28, 28, 1).astype("float32")
test_feature_vector = test_feature.reshape(-1, 28, 28, 1).astype("float32")

# 進行標準化 (將像素值 0-255 轉換為 0-1)
train_feature_normalize = train_feature_vector / 255.0
test_feature_normalize = test_feature_vector / 255.0

# One-Hot 編碼
train_label_onehot = keras.utils.to_categorical(train_label)
test_label_onehot = keras.utils.to_categorical(test_label)

# 建立 CNN 模型
model = keras.Sequential([
    # 第一個卷積層
    keras.layers.Conv2D(filters=32, kernel_size=(3, 3), padding="same", 
                        activation="relu", input_shape=(28, 28, 1)),
    keras.layers.MaxPool2D(pool_size=(2, 2)),

    # 第二個卷積層
    keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding="same", activation="relu"),
    keras.layers.MaxPool2D(pool_size=(2, 2)),

    # Dropout 層 (防止過擬合)
    keras.layers.Dropout(0.3),

    # 展平層
    keras.layers.Flatten(),

    # 全連接層
    keras.layers.Dense(units=256, activation="relu"),
    keras.layers.Dropout(0.4),

    # 輸出層 (10 個類別)
    keras.layers.Dense(units=10, activation="softmax")
])

# 編譯模型
model.compile(loss="categorical_crossentropy",
              optimizer="adam", metrics=["accuracy"])

# 訓練模型
train_history = model.fit(x=train_feature_normalize,
                          y=train_label_onehot,
                          validation_split=0.2,
                          epochs=10, batch_size=128, verbose=2)

# 評估模型
scores = model.evaluate(test_feature_normalize, test_label_onehot)
print("\n測試集準確率 =", scores[1])

# 確保儲存目錄存在
os.makedirs("CNN", exist_ok=True)

# 儲存模型
model.save("CNN/cnn_model.h5")
print("\ncnn_model.h5 模型儲存完畢！")
