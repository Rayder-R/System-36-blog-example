'''
# 一個簡單的單層感知器（SLP）模型，用於二分類問題。


'''


# %%
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


# %%

# 生成 2D 資料點（兩類）
np.random.seed(40)  # 固定隨機種子
class_0 = np.random.randn(100, 2) + np.array([-2, 0])  # 第一類
class_1 = np.random.randn(100, 2) + np.array([2, 0])   # 第二類

print("class_0 shape:", class_0[:1], class_0.shape)
print("class_1 shape:", class_0[:1], class_1.shape)

# 合併資料與標籤
data = np.vstack([class_0, class_1]).astype(np.float32)
labels = np.array([0]*100 + [1]*100).astype(np.float32).reshape(-1, 1)

# 顯示合併後資料與 shape
print("Combined data sample:", data[:1], "shape:", data.shape)
print("Labels sample:", labels[:1], "shape:", labels.shape)

# 繪製資料點
plt.scatter(class_0[:, 0], class_0[:, 1],
           alpha=0.6,               # 透明度：讓點半透明但仍清晰
                linewidth=0.1,           # 外框線條寬度
                s=40,                    # 點的大小，預設是20，適度加大
                color='#003060',
                label='Class 0'
            )
plt.scatter(class_1[:, 0], class_1[:, 1],
          alpha=0.7,               # 透明度：讓點半透明但仍清晰
                linewidth=0.1,           # 外框線條寬度
                s=40,                    # 點的大小，預設是20，適度加大
                color='#ADADAD',
                label='Class 1'
            )

plt.legend()
plt.title("Generated Data")
plt.show()

# %%

# 定義神經網路層


class SimpleNeuralNetwork:
    def __init__(self, input_dim):
        self.weights = tf.Variable(tf.random.normal([input_dim, 1]))
        self.bias = tf.Variable(tf.zeros([1]))

        # 前向傳播
    def forward(self, x):
        return tf.matmul(x, self.weights) + self.bias

# 定義交叉熵損失函數


def binary_crossentropy_loss(y_true, y_pred):
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true, logits=y_pred))

# 訓練模型


def train_model(model, data, labels, learning_rate=0.1, epochs=100):

    # 優化器與學習率
    optimizer = tf.optimizers.SGD(learning_rate)

    loss_history = []
    for epoch in range(epochs):
        with tf.GradientTape() as tape:
            logits = model.forward(data)
            loss = binary_crossentropy_loss(labels, logits)

        # 計算梯度並更新權重
        gradients = tape.gradient(loss, [model.weights, model.bias])
        optimizer.apply_gradients(zip(gradients, [model.weights, model.bias]))

        # 損失函數變化紀錄
        loss_history.append(loss.numpy())
        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1}, Loss: {loss.numpy():.4f}")

    return model

# 計算分類機率


def predict(model, data):
    logits = model.forward(data)
    return tf.sigmoid(logits).numpy() > 0.5

# 計算模型準確率


def evaluate_accuracy(model, data, labels):
    predictions = predict(model, data)
    accuracy = np.mean(predictions == labels)
    print(f"Model Accuracy: {accuracy * 100:.2f}%")
    return accuracy

# 視覺化決策邊界


def plot_decision_boundary(model, data, labels):
    x_min, x_max = data[:, 0].min() - 1, data[:, 0].max() + 1
    y_min, y_max = data[:, 1].min() - 1, data[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))
    grid_data = np.c_[xx.ravel(), yy.ravel()].astype(np.float32)

    # 預測網格點
    predictions = predict(model, grid_data).reshape(xx.shape)

    plt.contourf(xx, yy, predictions, cmap="binary", alpha=0.1)
    plt.scatter(data[:100, 0], data[:100, 1], 
                alpha=0.6,               # 透明度：讓點半透明但仍清晰
                linewidth=0.1,           # 外框線條寬度
                s=40,                    # 點的大小，預設是20，適度加大
                color='#003060',
                label='Class 0'
                )
    plt.scatter(data[100:, 0], data[100:, 1], 
                alpha=0.7,               # 透明度：讓點半透明但仍清晰
                linewidth=0.1,           # 外框線條寬度
                s=40,                    # 點的大小，預設是20，適度加大
                color='#ADADAD',
                label='Class 1'
                )
    plt.legend()
    plt.title("Decision Boundary")
    plt.show()


# 訓練與測試
model = SimpleNeuralNetwork(input_dim=2)  # 建立模型
model = train_model(model, data, labels, learning_rate=0.1, epochs=100)  # 訓練模型
evaluate_accuracy(model, data, labels)  # 計算模型準確率
plot_decision_boundary(model, data, labels)  # 視覺化推理結果

# %%
