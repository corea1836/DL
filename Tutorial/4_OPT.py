# 옵티마이저(Optimizer)는 손실 함수를 통해 얻은 손실값으로부터 모델을 업데이트하는 방식
# TensorFlow는 SGD, Adam, RMSprop와 같은 다양한 옵티마이저를 제공함.

import tensorflow as tf
from tensorflow import keras
import numpy as np

# 1. NN 모델 구성하기

tf.random.set_seed(0)
# tf.random 모듈의 set_seed() 함수를 사용해서 랜덤 시드를 설정

model = tf.keras.models.Sequential([
  tf.keras.layers.Dense(units=3, input_shape=[1])
])

# 1개의 input ----- 3개의 output

#------------------------------------------------------------------------

# 2. NN 컴파일
model.compile(loss='mse', optimizer='SGD')

# loss function : mse
# optimizer : SGD(Stohastic Gradient Descent, 확률적 경사하강법)

#------------------------------------------------------------------------

# 3. NN 훈련하기

# model.fit([1], [ [ 0, 1, 0 ] ], epochs=1)
# model.evaluate([1], [ [ 0, 1, 0 ] ])

'''
1/1 [==============================] - 0s 198ms/step - loss: 1.0738
1/1 [==============================] - 0s 92ms/step - loss: 1.0453
'''
# fit() 메서드는 훈련 진행 상황과 손실값을 반환함.
# 1회 epoch 이후, evaluate() 메서드를 사용해서 손실값을 확인하면 1.0738 -> 1.0453으로 감소했음을 확인

history = model.fit([1], [ [ 0, 1, 0 ] ], epochs=100)

'''
Epoch 1/100
1/1 [==============================] - 0s 1ms/step - loss: 1.0738
Epoch 2/100
1/1 [==============================] - 0s 1ms/step - loss: 1.0453
Epoch 3/100
1/1 [==============================] - 0s 982us/step - loss: 1.0176
.
.
.
Epoch 98/100
1/1 [==============================] - 0s 1ms/step - loss: 0.0794
Epoch 99/100
1/1 [==============================] - 0s 1ms/step - loss: 0.0773
Epoch 100/100
1/1 [==============================] - 0s 2ms/step - loss: 0.0753
'''
# 100회 훈련시 줄어든 시간과 손실값을 확인할 수 있다.

#------------------------------------------------------------------------

# 4. 손실값 시각화하기

import matplotlib.pyplot as plt

plt.style.use('default')
plt.rcParams['figure.figsize'] = (4, 3)
plt.rcParams['font.size'] = 12

loss = history.history['loss']
plt.plot(loss)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()

# fit() 메서드는 History 객체를 반환한다.
# History 객체의 history 속성은 훈련 과정의 손실값(loss values)과 지표(metrics)를 포함한다.
# 이번 예제에서는 컴파일과정에 지표를 지정하지 않았기 때문에 없는걸로 가정한다.
# 훈련 과정의 손실값을 Matplolib을 이용해 그래프로 나타내면 감소하는 경향을 확인할 수 있다.