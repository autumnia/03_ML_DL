# Natural Language Processing

```Python
    # Sarcasm 데이터세    
        0: 정상  1: 비꼬는 기사

    # 토큰화  ( 쪼갠다. )
        I am a boy and I am not a girl
        0 1            0  1

        1단계: 단어별 사전 만들기
        2단계: 치환
        3단계: 문장의 길이가 달라 길이를 정한 후  짧은 경우 0으로 채우고 길면 잘라낸다.

```


```Python
# RNN 을 활용한 텍스트 분류 (Text Classification)

# 1 단계
import json
import tensorflow as tf
import numpy as np
import urllib

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint

# 2단계
    # 필요한 데이터셋 다운로드
url = 'https://storage.googleapis.com/download.tensorflow.org/data/sarcasm.json'
urllib.request.urlretrieve(url, 'sarcasm.json')

    # 데이타 로드
with open('sarcasm.json') as f:
    datas = json.load(f)

    # 데이터 5개만 출력
datas[:5]

    # 전처리 데이터셋 구성 X (Feature): sentences   Y (Label): label
    # 문장 5개만 출력
sentences = []
labels = []
for data in datas:
    sentences.append(data['headline'])
    labels.append(data['is_sarcastic'])
    # 출력    
sentences[:5]
labels[:5]

    # Train / Valid set 분리
training_size = 20000
train_sentences = sentences[:training_size]
train_labels = labels[:training_size]
validation_sentences = sentences[training_size:]
validation_labels = labels[training_size:]

# 3단계  자연어 전처리 
# 토큰나이저   OOV : Out Of Vocab token
vocab_size = 1000
oov_tok = "<OOV>"
tokenizer = Tokenizer(num_words=vocab_size, oov_token='<OOV>')

#  단어사전 만들기
tokenizer.fit_on_texts(train_sentences)
for key, value in tokenizer.word_index.items():
    print('{}  \t======>\t {}'.format(key, value))
    if value == 25:
        break
len(tokenizer.word_index)  # 사이즈 확인

word_index = tokenizer.word_index
# 데이터 확인
word_index['trump']
word_index['hello']
word_index['<OOV>']


train_sequences = tokenizer.texts_to_sequences(train_sentences)
validation_sequences = tokenizer.texts_to_sequences(validation_sentences)
train_sequences[:5]   # 변환된 시퀀스 확인

train_sentences[4]
word_index['j'], word_index['k'], word_index['rowling'], word_index['wishes'], word_index['snape'], word_index['happy']
train_sequences[4]

max_length = 120    # 한 문장의 최대 단어 숫자
trunc_type='post'   # 잘라낼 문장의 위치
padding_type='post' # 채워줄 문장의 위치

train_padded = pad_sequences(train_sequences, maxlen=max_length, truncating=trunc_type, padding=padding_type)
validation_padded = pad_sequences(validation_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

train_padded.shape  # 변환후 세이프 확인
 train_padded[0]    # 확인차 줄력

# 4단계
# label 값을 numpy array로 변환 ( model이 list type은 받아들이지 못하므로, numpy array로 변환합니다. )
train_labels = np.array(train_labels)
validation_labels = np.array(validation_labels)

embedding_dim = 16
sample = np.array(train_padded[0])
sample

x = Embedding(vocab_size, embedding_dim, input_length=max_length)
x(sample)[0]

# 5단계  모델정의
model = Sequential([
    Embedding(vocab_size, embedding_dim, input_length=max_length),
    Bidirectional(LSTM(64, return_sequences=True)),
    Bidirectional(LSTM(64)),
    Dense(32, activation='relu'),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])
model.summary()

# 6단계 컴파일
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
checkpoint_path = 'my_checkpoint.ckpt'
checkpoint = ModelCheckpoint(checkpoint_path, 
                             save_weights_only=True, 
                             save_best_only=True, 
                             monitor='val_loss',
                             verbose=1)

# 7단계 학습
epochs=10
history = model.fit(train_padded, train_labels, 
                    validation_data=(validation_padded, validation_labels),
                    callbacks=[checkpoint],
                    epochs=epochs)
model.load_weights(checkpoint_path)

# 8단계 시각화
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 9))
plt.plot(np.arange(1, epochs+1), history.history['loss'])
plt.plot(np.arange(1, epochs+1), history.history['val_loss'])
plt.title('Loss / Val Loss', fontsize=20)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(['loss', 'val_loss'], fontsize=15)
plt.show()

plt.figure(figsize=(12, 9))
plt.plot(np.arange(1, epochs+1), history.history['acc'])
plt.plot(np.arange(1, epochs+1), history.history['val_acc'])
plt.title('Acc / Val Acc', fontsize=20)
plt.xlabel('Epochs')
plt.ylabel('Acc')
plt.legend(['acc', 'val_acc'], fontsize=15)
plt.show()
```
