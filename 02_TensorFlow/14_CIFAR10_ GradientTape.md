# Tensor flow
    https://www.tensorflow.org/datasets/catalog/overview

## 모델
    Functional API

## 학습
    GradientTape

## 어플리케이션
    트랜스퍼러닝( VGG16 )    

## 실습
```Python
# 단계 1 임포트
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

# Sequential API 용
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten

# functional API 용
from tensorflow.keras.models import Model
from tensorflow.keras.layers import input


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
input_ = input( shape( 32, 32, 3) )
x = Conv2D(32, 3, activation='relu')(input_)
x = MaxPooling2D(2, 2)(x)
x = Conv2D(64, 3, activation='relu')(x)
x = MaxPooling2D(2,2)(x)
x = Flatten()(x)
x = Dense(32, activation='relu')(x)
x = Dense(10, activation='softmax')(x)
model = Model(input_, x )
model.summary()

# 단계 4 모델 컴파일
model.compile( 
    optimizer='adam', 
    loss='sparse_categorical_crossentropy', 
    metrics=['acc']
)

# 단계 5 모델 학습
loss_function = tf.keras.losses.SparseCategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam()

train_loss  = tf.keras.metrics.Mean( name='train_loss')
train_acc   = tf.keras.metrics.SparseCategoricalAccuracy( name='train_acc')

valid_loss  = tf.keras.metrics.Mean( name='valid_loss')
valid_acc   = tf.keras.metrics.SparseCategoricalAccuracy( name='valid_acc')

@tf.tunction
def train_step( image, label):
    with tf.GradientTape() as tape:
        prediction = model(image, training=True)
        loss = loss_function( label, prediction )

    gradients = tape.gradient(loss, model.trainable_bariables)
    optimizer.apply_graients(zip(gradients.model.trainable_variables))

    train_loss(loss)
    train_acc(lable, prediction)

@tf.tunction
def valid_step( image, label):
    with tf.GradientTape() as tape:
        prediction = model(image, training=False)
        loss = loss_function( label, prediction )

    valid_loss(loss)
    valid_acc(label, prediction)

EPOCHS = 10 
for epoch in range(EPOCHS):
    train_loss.reset_states()
    train_acc.reset_states()

    valid_loss.reset_states()
    valid_acc.reset_states()

    for image, label in train_data:
        train_step(image, label)

    for image, label in valid_data:
        valid_step(image, label)

    print(f' 
        epoch: {epoch + 1}, 
        loss: {train_loss.result()}, 
        acc: {train_acc.result()},
        val_loss: {valid_loss.result()},
        val_acc: {valid_acc.result()}
    ')
































```