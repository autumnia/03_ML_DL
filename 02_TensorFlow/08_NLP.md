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

```
