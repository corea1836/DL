# 손실함수(Loss function)
# 손실함수는 NN의 예측이 얼마나 잘 맞는지 측정하는 역할을 한다.
# 손실함수로부터 얻어진 손실값(Loss value)은 훈련 과정에서 NN이 얼마나 잘 훈련되었는지 확인하는 지표가 된다.

# MSE(Mean Squared Error)로 예제 진행

# 1. NN 구성하기
import tensorflow as tf
from tensorflow import keras
import numpy as np

# 하나의 입력을 받고 세개의 출력 노드를 갖는 NN를 구성

model = tf.keras.models.Sequential([
  tf.keras.layers.Dense(3, input_shape=[1])
])

#--------------------------------------------------------------------
# 2. NN 컴파일하기

# 컴파일이란? 모델 구성 후, 모델을 훈련하기 전에 손실함수와 옵티마이저를 지정해주는 과정

model.compile(loss='mse')

# 이번 예제에서는 compile()메서드의 loss파라미터를 이용해 손실 함수를 'mse'로 지정함.
# mse(Mean Squared Error)는 수식을 이용해 평균 제곱 오차를 계산하는 방식임.
# MSE = 1/n ∑(y - y(hat)) ** 2
# MSE는 예측치와 관측치의 차이인 오차의 제곱에 비례해서 손실 함수로부터 계산되는 손실값이 커진다.

#--------------------------------------------------------------------

# 3. NN 예측하기

pred = model.predict([0])
# print(pred)
# [[0. 0. 0.]]
# 임의로 생성된 모델의 가중치 값(W)이 있지만, 입력이 0이므로 예측값도 모두 0을 출력한다.

#--------------------------------------------------------------------

# 4. NN 손실 계산하기

# evaluate() 메서드는 예측값과 관측값 사이의 손실값을 반환한다.

model.evaluate([0], [[0, 1, 0]])
# 1/1 [==============================] - 0s 90ms/step - loss: 0.3333
# 모델의 손실함수를 MSE로 지정했기 때문에 0.3333이 계산된다.
# loss = (0 - 0)**2 + (1 - 0)**2 + (0 - 0)**2 / 3
# : 0.3333
