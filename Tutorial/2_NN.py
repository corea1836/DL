import tensorflow as tf
from tensorflow import keras
import numpy as np

# 1. Neural Network 구성하기

# tf.keras는 TensorFlow의 하이레벨을 구현하기 위한 Keras API 모듈이다.
# tf.keras 모듈의 Sequential클래스는 Neural Network의 각 층을 순서대로 쌓을 수 있도록 한다.
# tf.keras.layer 모듈의 Dense클래스는 완전 연결된 하나의 뉴런층을 구현한다.
# units는 뉴런 또는 출력 노드의 개수이다.(양수로 설정)
# input_shape는 입력 데이터의 형태를 결정한다.

# Dense(units=1)
# input ----- output


model = keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])

'''
model = tf.keras.models.Sequential([
  tf.keras.layers.Dense(1, input_shape=[1])
])
'''


# ----------------------------------------------------------------------

# 2. Neural Network 컴파일하기

# NN 모델을 컴파일 하는 과정에서는, 
# 모델의 학습에 필요한 손실함수(loss function)와 옵티마이저(optimizer)를 결정한다.
# 손실함수는 NN의 예측이 얼마나 잘 맞는지 측정하는 역할을 하고,
# 옵티마지너는 더 개선된 예측값을 출력하도록 최적화 하는 알고리즘이다.
# 이번 예제에서는 LF : mse, opt : SGD를 사용함.

model.compile(loss='mean_squared_error', optimizer='sgd')

# ----------------------------------------------------------------------

# 3. NN 훈련하기

# Sequential클래스의 fit()매서드는 주어진 입출력 데이터에 대해 지정한 횟수만큼 NN을 훈련한다.
# 훈련이 이루어질 때마다, NN은 주어진 입력에 대해 주어진 출력값에 더 가까운 값을 출력하게 된다.
# (xs) -> input ----- output -> (ys)

xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
ys = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=float)

model.fit(xs, ys, epochs=500)

# xs, ys는 NN 훈련에 사용할 입력과 출력 데이터이며, y = 2x - 1의 관계를 갖는 것처럼 보인다.
# epoch는 주어진 데이터를 한 번 훈련하는 단위이다.

# ----------------------------------------------------------------------

# 4. NN 예측하기
# Sequential클래스의 predict() 메서드를 사용하면 특정 입력에 대해 NN이 예측하는 값을 얻을 수 있다.
# (5.0) -> input ----- output -> (??)

pred = model.predict([5.0])
print(pred)
# [[8.995782]] 약 8.99이므로 2x - 1에 출력하도록 훈련되었다.
# 오차가 생기는 이유 : 6개라는 적은 양의 입출력 데이터를 훈련에 사용했고, 
# 모든 x에 대해 입출력의 관계가 2x - 1이 아닐 가능성이 있기 때문에.
