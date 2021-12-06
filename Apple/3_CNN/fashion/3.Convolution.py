import tensorflow as tf
import requests
requests.packages.urllib3.disable_warnings()
import ssl
import matplotlib.pyplot as plt
import numpy as np

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    # Legacy Python that doesn't verify HTTPS certificates by default
    pass
else:
    # Handle target environment that doesn't support HTTPS verification
    ssl._create_default_https_context = _create_unverified_https_context

( trainX, trainY ), ( testX, testY )= tf.keras.datasets.fashion_mnist.load_data()

# 데이터 전처리

# 0 ~ 255까지의 숫자를 0 ~ 1사이로 압축시키자.(optional)
# 속도가 빨라지고 정확도도 좋아진다.
trainX = trainX / 255.0
testX = testX / 255.0

# trinaX, Y, testX, Y 모두 출력해보면 파이썬 리스트가 아니라 numpy array 자료형이다.
# numpy array 자료의 shape를 변경하자.
''' trainX.reshape( (60000, 28, 28, 1) ) '''
# 원래는 (60000, 28, 28)인 형태인데 여기다 원소 하나하나 괄호를 쳐주는 형태이다.
# (60000, 28, 28) -> (60000, 28, 28, 1)
# 하나의 데이터(리스트 안에 원소 1개)가 28개 있고 그게 28개 있고(사진 하나), 이게 전체 6만개 세트가 있는 형태

# 이를 범용적으로 사용하려면?
# trainX.shape()
# : (60000, 28, 28) -> trainX[0]
trainX = trainX.reshape( ( trainX.shape[0], trainX.shape[1], trainX.shape[2], 1 ) )
testX = testX.reshape( ( testX.shape[0], testX.shape[1], testX.shape[2], 1 ) )

model = tf.keras.Sequential([
  tf.keras.layers.Conv2D( 32, (3, 3), padding='same', activation='relu', input_shape=(28, 28, 1) ),
  # Convolution layer 추가하기
  # Conv2D( 내가 원하는 수의 이미지 복사본, ( n, n : kernel 사이즈 ), padding(테두리), AF, input_shape   )
  # input_shape가 없으면 ndim 에러가 난다.

  # Conv2D layer 사용법(흑백 데이터)
  # Conv2D는 4차원 데이터를 필요로 한다.(60000, 28, 28, 1)
  # 샘플로 넣은 데이터는 (60000, 28, 28)이라 input_shape를 넣지 않으면 ndim 에러가 난다.
  # input_shape는 하나의 데이터 shape(여기선 28 * 28)
  # [ [ 0, 0, 0, ... 0 ],        [ [ [0], [0], [0], ... [0] ],
  #          ...            =>
  #   [ 0, 0, 0, ... 0 ] ]         [ [0], [0], [0], ... [0] ]
  # 하나의 input_shape인 28 * 28을 3차원으로 늘려서 넣어야 한다(28, 28, 1) 이게 6만개 있으니까 4차원이다.
  # ( 28, 28, 1 ) 하나의 리스트 데이터가 28개 있고 그게 또 28개 있다.
  # 그런데 이번 문제에서 input_shape는 위와같이 되어있지 않다. -> numpy로 모양을 맞줘추자.

  # Conv2D layer 사용법(컬러 데이터)
  # 흑백 데이터는 하나의 픽셀에 하나의 숫자가 담긴 모양을 하고 있지만,
  # 컬러 데이터는 이런 모양을 띄고 있다.
  # [ [ [ 0, 0, 0 ], [ 0, 0, 0 ] ... [ 0, 0, 0 ] ]
  #   [ [ 0, 0, 0 ], [ 0, 0, 0 ] ... [ 0, 0, 0 ] ]
  #                     ...
  #   [ [ 0, 0, 0 ], [ 0, 0, 0 ] ... [ 0, 0, 0 ] ] ]
  # 하나의 픽셀에 R, G, B 세개의 숫자가 담긴 모양을 하고 있다.
  # 이때는 input_shape=( 28, 28, 3 )인 모양이 되어야 한다.

  # tf.keras.layers.Dense(128, input_shape=(28, 28), activation='relu'),

  # Convolution을 적용한 다음이 가장 중요 => 이미지의 중요한 부분을 중앙으로 모으자.(Pooling layer)
  # 2D 사진을 maxpooling을 적용해서 이미지 사이즈를 줄여주고, 중요한 정보들을 가운데로 모은다.
  # ( 2, 2 ) 는 pooling size 2 * 2 행렬 사이즈
  tf.keras.layers.MaxPooling2D( ( 2, 2 ) ),
  tf.keras.layers.Flatten(),

  # Convolution layer 구성 순서
  # Conv - Pooling * 여러번 가능 하지만 원하는 결과에 따라 Flatten을 적용해줘야한다.
  # 그리고 Flatten - Dense - 출력

  # 이 문제에서 원하는 결과는 1차원 데이터이기 때문에 위와같이 레이어를 구성하면 4차원의 데이터다.
  # Conv - Pooling은 여러번 구성해도 괜찮지만 원하는 결과를 얻기위해 Flatten을 적용 후 Dense 레이어로 넘어가자.


  tf.keras.layers.Dense(64, activation='relu'),
  # tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(10, activation='softmax'),
  ])

model.summary()

model.compile( loss='sparse_categorical_crossentropy', optimizer = 'adam', metrics=['accuracy'] )
model.fit(trainX, trainY, validation_data=(testX, testY), epochs=5)

'''
Epoch 1/5
1875/1875 [==============================] - 9s 5ms/step - loss: 0.4232 - accuracy: 0.8524
Epoch 2/5
1875/1875 [==============================] - 9s 5ms/step - loss: 0.2814 - accuracy: 0.8989
Epoch 3/5
1875/1875 [==============================] - 10s 5ms/step - loss: 0.2401 - accuracy: 0.9130
Epoch 4/5
1875/1875 [==============================] - 9s 5ms/step - loss: 0.2086 - accuracy: 0.9231
Epoch 5/5
1875/1875 [==============================] - 9s 5ms/step - loss: 0.1834 - accuracy: 0.9317
epoch을 5번만 돌려도 accuracy가 높이 올라간다.
'''


score = model.evaluate( testX, testY )
# evaluate 함수는 모델을 만들고 학습까지 완료한 상태에서 모델을 평가해주는 함수.
# evaluate( testX, testY )를 적는데 파라미터로 trainX, trainY는 쓰면 안된다.
# 왜? 학습용 데이터를 적용하면 그 동안 너무 많이 봤기 때문에 답안을 외울 수도 있다.
# 컴퓨터가 처음 보는 데이터를 넣어야한다.

print(score)
# [0.2955975830554962, 0.8962000012397766]
# test 데이터의 [loss, accuracy]
# 그런데 마지막 epoch의 accuracy와 차이가 있다. 약 93% -> 89%
# 이를 overfitting이라 한다.
# overfitting이란?
# 학습용 데이터를 외운 것.
# 기존의 똑같은 문제를 몇 번 반복했기 때문에 traning 데이터셋을 외워 accuracy를 높였기 때문에,
# 새로운 데이터를 가져다주면 잘 못푸는 현상을 말한다.

# overfitting을 해결하는 방법
# eopch 1회 끝날 때 마다 새로운 데이터를 테스트하는 방법
# model.fit(trainX, trainY, validation_data=(testX, testY), eopchs=5)

'''
Epoch 1/5
1875/1875 [==============================] - 9s 5ms/step - loss: 0.3891 - accuracy: 0.8630 - val_loss: 0.3132 - val_accuracy: 0.8863
Epoch 2/5
1875/1875 [==============================] - 9s 5ms/step - loss: 0.2642 - accuracy: 0.9045 - val_loss: 0.2808 - val_accuracy: 0.8981
Epoch 3/5
1875/1875 [==============================] - 9s 5ms/step - loss: 0.2229 - accuracy: 0.9183 - val_loss: 0.2607 - val_accuracy: 0.9038
Epoch 4/5
1875/1875 [==============================] - 9s 5ms/step - loss: 0.1949 - accuracy: 0.9293 - val_loss: 0.2537 - val_accuracy: 0.9105
Epoch 5/5
1875/1875 [==============================] - 9s 5ms/step - loss: 0.1707 - accuracy: 0.9367 - val_loss: 0.2496 - val_accuracy: 0.9114
313/313 [==============================] - 1s 2ms/step - loss: 0.2496 - accuracy: 0.9114
[0.24959604442119598, 0.9114000201225281]
'''
# 실제 epochs를 많이 늘리면 overfittin이 이뤄지는 즉, 더이상 학습이 잘 이뤄지지 않는 구간이 온다.
# validation accuracy를 확인하면서 overfitting이 이뤄지기 전에 멈춰서 모델을 뽑을 수 있다.
# val_accuracy를 높일 방법을 찾자.
# Dense layer 추가?
# Conv + Pooling 추가?
# 고민해보자.
