import tensorflow as tf
# tensorflow에서 딥러능은 어떻게 하는것인가?

# 딥러닝으로 간단한 수학문제 풀어보기
키 = [170, 180, 175, 160]
실발 = [260, 270, 265, 255]

# Q : 키와 신발사이즈는 어떤 관련이 있을까?
# A : Linear Regression을 이용하자

# y(신발사이즈) = ax(키) + b
# x에 키를 대입하면 y(신발사이즈)를 알 수 있도록 a와 b를 구하면 된다.
# 실발 = a * 키 + b 
# y = ax + b, 일차 함수를 찾는 식임. 이와 유사한 작업을 딥러닝에서 하면 된다.

# ------------------------------------------------------------------------

키 = 170
신발 = 260

# 신발 = 키 * a + b
# 일차함수처럼 a, b를 DL로 추정하는 것임.

# 1. a와 b의 정의가 필요하다.
#    초기값은 임의의 값으로 세팅하고 a, b가 좋은 결과가 나올 때 까지 학습을 시킨다.
a = tf.Variable(0.1)
b = tf.Variable(0.2)

# 2. 경사 하강법으로 a, b를 학습시킨다.
opt = tf.keras.optimizers.Adam(learning_rate=0.1)

def 손실함수():
    예측값 = 키 * a + b
    return tf.square(260 - 예측값)

for i in range(300):
    opt.minimize(손실함수, var_list=[a, b])
    print(a.numpy(), b.numpy())
# 1.5198832 1.6198832

print(170 * 1.52 + 1.62)
# 260.02