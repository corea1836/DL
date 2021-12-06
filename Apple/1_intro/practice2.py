import tensorflow as tf

# 데이터가 많은 경우 딥러닝 돌리는 법

train_x = [1, 2, 3, 4, 5, 6, 7]
train_y = [3, 5, 7, 9, 11 ,13 ,15]

# Q : x를 어떻게 하면 y가 나올까?
# A : x에 2를 곱하고 1을 더하면 된다. -> 이걸 DL로 추론해보자

# 딥러닝 순서
# 1. 모델 만들기
# 2. 학습 시키기

# 1. 예측 모델 만들기(수식)
a = tf.Variable(0.1)
b = tf.Variable(0.1)
# 실제 변수 초기값은 randomize해서 무작위로 넣는다.

# 예측_y = train_x * a + b
# a와 b를 컴퓨터가 구하도록 시키자(식을 2차함수로 쓸 수도 있고, 지금 같은 경우는 직선으로 풀 수 있을것 같아 1차 함수로 구현)

# 2. 경사하강법을 이용해 최적화 시키기
opt = tf.keras.optimizers.Adam(learning_rate=0.01)

def 손실함수():
    예측_y = train_x * a + b
    return tf.keras.losses.mse(train_y, 예측_y)

for i in range(2900):
    opt.minimize(손실함수, var_list=[a, b])
    print(a.numpy(), b.numpy())

