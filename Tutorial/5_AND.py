# AND 로직

# AND 연산은 논리연산의 한 종류로 두 상태가 모두 참일 때 (True, 1) 참이고,
# 둘 중 하나라도 거짓이라면(False, 0) 거짓이 되는 연산이다.

# TensorFlow를 이용해 두 입력값에 대해 AND 논리 연산 결과를 출력하는 신경망을 구현함.
# 두개의 입력값을 받고, 하나의 값을 출력하는 인공신경망을 구축한다.

import tensorflow as tf
from tensorflow import keras
import numpy as np

tf.random.set_seed(0)

# tf.random 모듈의 set_seed()함수를 사용해 랜덤 시드를 설정함.

# 1 훈련 데이터 준비하기

x_train = [ [ 0, 0 ], [ 0, 1 ], [ 1, 0 ], [ 1, 1 ] ]
y_train = [ [ 0 ], [ 0 ], [ 0 ], [ 1 ] ]

# x_train, y_train은 각각 훈련에 사용할 입력값, 출력값임

#------------------------------------------------------------------------

# 2. NN 구성하기

model = tf.keras.models.Sequential([
  tf.keras.layers.Dense(units=3, input_shape=[2], activation='relu'),
  tf.keras.layers.Dense(units=1)
])

# 입력은 2개 -> Hidden layer은 3개(활성화함수 : ReLu(Rectified Linear Unit)) -> 출력은 1개로 구성

#------------------------------------------------------------------------

# 3. NN 컴파일하기

model.compile(loss='mse', optimizer='Adam')

# 손실함수로 mse, 옵티마이저로 Adam으로 설정

#------------------------------------------------------------------------

# 4. NN 훈련하기

pred_before_training = model.predict(x_train)
print('Before Traning : \n', pred_before_training)

history = model.fit(x_train, y_train, epochs=1000, verbose=0)

pred_after_training = model.predict(x_train)
print('After Traning : \n', pred_after_training)

'''
Before Traning : 
 [[0.        ]
 [0.6210649 ]
 [0.06930891]
 [0.6721569 ]]
After Traning : 
 [[-0.00612798]
 [ 0.00896955]
 [ 0.00497075]
 [ 0.99055475]]
'''

# predict() 메서드를 이용해 NN의 예측값을 얻음
# Model 클래스의 fit() 메서드는 모델을 훈련하고, 훈련 진행상황과 현재의 손실값을 반환
# 모델 훈련 전후로 입력 데이터에 대한 NN의 예측값을 출력함.

#------------------------------------------------------------------------

# 5. 손실값 확인하기

import matplotlib.pyplot as plt

loss = history.history['loss']
plt.plot(loss)
plt.xlabel('Epoch', labelpad=15)
plt.ylabel('Loss', labelpad=15)

plt.show()

plt.style.use('default')
plt.rcParams['figure.figsize'] = (6, 4)
plt.rcParams['font.size'] = 14

plt.plot(pred_before_training, 's-', markersize=10, label='pred_before_training')
plt.plot(pred_after_training, 'd-', markersize=10, label='pred_after_training')
plt.plot(y_train, 'o-', markersize=10, label='y_train')

plt.xticks(np.arange(4), labels=['[0, 0]', '[0, 1]', '[1, 0]', '[1, 1]'])
plt.xlabel('Input (x_train)', labelpad=15)
plt.ylabel('Output (y_train)', labelpad=15)

plt.legend()
plt.show()

# Matplotlib 라이브러리를 사용해 훈련 전후의 입력값과 출력값을 나타냄.
# 1000회 훈련이 이루어지면 네가지 경우의 0과 1 입력에 대해 1% 미만의 오차로 AND연산을 수행할 수 있음.