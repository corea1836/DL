# 뉴런층의 출력 확인하기
# 특정 입력 데이터에 대해 각 뉴런층이 출력하는 값을 확인하는 방법

import tensorflow as tf
import numpy as np

tf.random.set_seed(0)

# 1. 훈련 데이터 준비하기
x_train = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
y_train = np.array([[0], [1], [1]])

# 2. 뉴런층 만들기
input_layer = tf.keras.layers.InputLayer(input_shape=(3,))
hidden_layer = tf.keras.layers.Dense(units=4, activation='relu')
output_layer = tf.keras.layers.Dense(units=2, activation='softmax')

# 입력층 : 길이 3을 갖는 벡터
# 은닉층 : 네 개의 뉴런 노드 + AF(relu)
# 출력층 : 두 개의 뉴런 노드 + AF(softmax)

# 3. NN 구성하기
model = tf.keras.Sequential([
  input_layer,
  hidden_layer,
  output_layer
  ])

# 4. NN 컴파일하기
model.compile(loss='mse', optimizer='Adam')

# 5. 은닉층의 출력값 확인하기
intermediate_layer_model = tf.keras.Model(inputs=model.input, outputs=model.layers[0].output)
intermediate_output = intermediate_layer_model(x_train)

print('======== Inputs ========')
print(x_train)

print('\n======== Weights of Hidden Layer ========')
print(hidden_layer.get_weights()[0])

print('\n======== Outputs of Hidden Layer ========')
print(intermediate_output)

'''
======== Inputs ========
[[1 0 0]
 [0 1 0]
 [0 0 1]]

======== Weights of Hidden Layer ========
[[-0.3851872  -0.54333335  0.0655309   0.1134268 ]
 [-0.15428883  0.5699866  -0.01254469  0.9223561 ]
 [ 0.36428273 -0.6936733   0.38850498  0.30073535]]

======== Outputs of Hidden Layer ========
tf.Tensor(
[[0.         0.         0.0655309  0.1134268 ]
 [0.         0.5699866  0.         0.9223561 ]
 [0.36428273 0.         0.38850498 0.30073535]], shape=(3, 4), dtype=float32)
'''

# 입력 데이터 [ 1, 0, 0 ]의 경우를 살펴보면,
# 입력층 첫번째 보드인 입력 1에 시냅스 가중치[-0.3851872 -0.54333335 0.0655309 0.1134268 ]가 곱해진다.
# 다음으로 은닉층의 활성화함수인 ReLU가 적용되어 0보다 작은 값은 0이되고, 0보다 큰 값은 그대로 출력값이 된다.

# 6. 출력층의 출력값 확인하기
pred = model.predict(x_train)

print('\n======== Outputs of Output Layer ========')
print(pred)

'''
======== Outputs of Output Layer ========
[[0.4551601  0.5448399 ]
 [0.18469976 0.8153002 ]
 [0.4541021  0.5458979 ]]
'''
# 출력층의 출력값은 전체 NN 신경망의 출력값이다.
# 전체 신경망의 출력값은 Model 클래스의 predict() 메서드를 사용해 얻을 수 있다.
# 세 개의 값을 갖는 세 개의 입력 데이터 백터에 대해 두 개의 값을 찾는 벡터 세 개를 출력한다.
# test input case는 3개였고, 각 input당 입력 값이 3개 -> 2개의 값을 출력하는 출력 layer의 test output case 는 3개