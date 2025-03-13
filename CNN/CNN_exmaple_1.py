from tensorflow import keras
from keras.datasets import mnist
from keras.models import load_model
from keras.utils import to_categorical
import numpy as np

# 讀取 MNIST 手寫數字資料集
(_, _), (test_feature, test_label) = mnist.load_data()

# 將測試資料的特徵值調整為 4 維張量 (批次大小, 高度, 寬度, 通道數)
test_feature_vector = test_feature.reshape(len(test_feature), 28, 28, 1).astype("float32")

# 進行特徵標準化，將像素值 (0-255) 縮放至 0-1 之間
test_feature_normalize = test_feature_vector / 255

# 轉換標籤為 One-Hot 編碼
test_label_onehot = to_categorical(test_label)

# 載入已訓練的 CNN 模型
model = load_model(".\CNN\cnn_model.h5")

# 進行預測，並取出最大機率的類別索引
prediction = np.argmax(model.predict(test_feature_normalize), axis=1)

# 顯示前幾筆預測結果
for i in range(10):
    print(f"預測結果: {prediction[i]}, 真實標籤: {test_label[i]}")