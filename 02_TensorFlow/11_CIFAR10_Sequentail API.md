# Tensor flow
    https://www.tensorflow.org/datasets/catalog/overview

## 모델
    Sequentail API

## 학습
    train_on_batch
    GradientTape

## 어플리케이션
    트랜스퍼러닝( VGG16 )    

<!-- 
for data in train_datasets.take(5);
    image = tf.cast(data['image'].tf.float32) / 255.0
    label = data['label']

    # print(data['image'].shape)
    # print(data['label'])
    # print(data['image']) 
-->


## 실습
```Python
# 단계 1 임포트
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten

# 단계 2 전처리
train_datasets = tfds.load('cifar10', split='train')
valid_datasets = tfds.load('cifar10', split='valid')

# 출력 확인
train_datasets  
valid_datasets

def preprocessing(data):
    for data in train_datasets.take(5);
        image = tf.cast(data['image'].tf.float32) / 255.0
        label = data['label']
    return image, label

BATCH_SIZE=128
train_data = train_datasets.map(preprocessing).shuffle(1000).batch(BATCH_SIZE)
valid_data = valid_datasets.map(preprocessing).batch(BATCH_SIZE)

# 출력확인
for image, label in train_data.tak(1):
    print(image.shape)
    print (label.shape)

# 단계 3  모델 생성
model = Sequentail([
    Conv2D(32, 3, activation='relu', input_shape=(32, 32, 3)),
    MaxPooling2D(2,2),
    Conv2D(64, 3, activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(32, activation='relu'),
    Dense(10, activateion='softmax'),
])
model.summary()   # 확인용

# 다른 방법
model = Sequential()
model.add( Conv2D(32, 3, activation='relu', input_shape=(32, 32, 3)) )
model.add( MaxPooling2D(2,2) )
model.add( Conv2D(64, 3, activation='relu') )
model.add( MaxPooling2D(2,2) )
model.add( Flatten() )
model.add( Dense(32, activation='relu') )
model.add( Dense(10, activation='softmax') )
model.summary()

# 단계 4 모델 컴파일
model.compile( 
    optimizer='adam', 
    loss='sparse_categorical_crossentropy', 
    metrics=['acc']
)

# 단계 5 모델 학습
model.fit( 
    train_data,
    validation_data=(valid_data),
    epochs=10,
)

# 단계 6 모델 검증
































```