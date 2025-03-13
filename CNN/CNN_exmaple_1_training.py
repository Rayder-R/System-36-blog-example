from tensorflow import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Dropout, Flatten, Dense
from keras.models import Sequential
from keras.datasets import mnist
from keras.utils import to_categorical
import numpy as np

np.random.seed(10)

(train_feature, train_label), (test_feature, test_label) = mnist.load_data()

print(len(train_feature))

train_feature_vector = train_feature.reshape(
    len(train_feature), 28, 28, 1).astype('float32')

test_feature_vector = test_feature.reshape(
    len(test_feature), 28, 28, 1).astype('float32')

train_feature_normalize = train_feature_vector / 255
test_feature_normalize = test_feature_vector / 255

train_feature_onehot = to_categorical(train_label)
test_feature_onehot = to_categorical(test_label)

print(f"One-Hot Encoding for label 1: {test_feature_onehot[1]}")
print(f"Original label 1: {test_label[1]}")


model = Sequential()

model.add(Conv2D(filters=10,
                 kernel_size=(3, 3),
                 padding='same',
                 input_shape=(28, 28, 1),
                 activation='relu'))

model.add(MaxPool2D(pool_size=(2, 2)))  

model.add(Conv2D(filters=10,
                 kernel_size=(3, 3),
                 padding='same',
                 activation='relu'))

model.add(MaxPool2D(pool_size=(2, 2)))  

model.add(Dropout(0.2))

model.add(Flatten())

model.add(Dense(units=256, activation='relu'))

model.add(Dense(units=10, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam', metrics=['accuracy'])

train_history = model.fit(x=train_feature_normalize,
                          y=train_feature_onehot, validation_split=0.2,
                          epochs=10, batch_size=200, verbose=2)

scores = model.evaluate(test_feature_normalize, test_feature_onehot)
print('\n準確率=', scores[1])

model.save('.\CNN\cnn_model.h5')
print("\n cnn_model.h5 模型儲存完畢")
