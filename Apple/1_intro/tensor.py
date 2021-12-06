# 기본 텐서 만들기
import tensorflow as tf

# 텐서 = tf.constant(3)
# tf.Tensor(3, shape=(), dtype=int32)

#텐서 = tf.constant( [3, 4, 5] )
# tf.Tensor([3 4 5], shape=(3,), dtype=int32)

#print(텐서)

# 텐서 = tensorflow에서 가장 기본적인 단위(숫자, 리스트를 담을 수 있다.)

# ------------------------------------------------------------

'''
tensorflow가 필요한 이유
ex) (x1 * w1) + (x2 * w2) + (x3 * w3)...
    이런 뉴럴네트워크를 계산할 때 행렬을 통해 쉽게 계산하게 해준다.(행렬의 곱 이용)
    숫자 여러개를 한 번에 계산하기 편하다.
    행렬로 인풋/w값이 저장 가능하고, 그럼 노드 계산하기가 편하다.
'''

텐서1 = tf.constant( [3, 4, 5] )
텐서2 = tf.constant( [6, 7, 8] )

# print(텐서1 + 텐서2)
# print(tf.add(텐서1, 텐서2))
# tf.Tensor([ 9 11 13], shape=(3,), dtype=int32)

# -----------------------------------------------------------

# tensor로 행렬 표현하기

텐서3 = tf.constant( [ [1, 2], 
                      [3, 4] ] )
# 리스트 안의 리스트를 콤마(,)로 상하배치가 가능하다.
# 비슷한 모양을 지닌 텐서끼리 사칙연산도 가능하다.

# 사칙연산
# +, tf.add()
# -, tf.subtract()
# /, tf.devide()
# *, matmul() 
# tf.multiply() -> 각 element 끼리의 곱(행렬의 곱셈이 아님)

# -----------------------------------------------------------

# 0만 담긴 텐서 만들기(1차원)
# 텐서4 = tf.zeros( 10 )
# tf.Tensor([0. 0. 0. 0. 0. 0. 0. 0. 0. 0.], shape=(10,), dtype=float32)

# 0만 담긴 텐서 만들기(2차원)
# 텐서4 = tf.zeros( [2, 2] )
# tf.Tensor( 
# [[0. 0.]
# [0. 0.]], shape=(2, 2), dtype=float32)

# 텐서 모양 지정
텐서4 = tf.zeros( [2, 2, 3] )
'''
tf.Tensor(
[[[0. 0. 0.]
  [0. 0. 0.]]

 [[0. 0. 0.]
  [0. 0. 0.]]], shape=(2, 2, 3), dtype=float32)
'''
# [2, 2, 3] 뒤에서 부터 읽자 (3개의 데이터(0)를 담은 리스트를 2개 세트로 생성해주고, 그걸 또 2개 생성해줘)
# print(텐서4)

# -----------------------------------------------------------

# tensor의 shape(= 몇 차원의 data 인지)

print(텐서1.shape)
# (3,) -> 자료가 3개 들어있다.
print(텐서3.shape)
# (2, 2) -> 2행 2열의 data이다.
텐서5 = tf.constant( [ [1, 2, 3],
                      [4, 5, 6] ] )
print(텐서5.shape)
# (2, 3) -> 뒤에서부터 해석하자.(3개의 데이터가 리스트에 들어있고 그게 2개 있다.)

# -----------------------------------------------------------

# tensor의 datatype
print(텐서1)
# tf.Tensor([3 4 5], shape=(3,), dtype=int32)
# dtype = int(정수)

텐서6 = tf.constant( [3.0, 2, 3] )
print(텐서6)
# tf.Tensor([3. 2. 3.], shape=(3,), dtype=float32)
# dtype = float(실수)
# DL은 tensor 자료를 만들 때 float으로 주로 만든다.

텐서7 = tf.constant( [3, 4, 5], tf.float32 )
print(텐서7)
# tf.Tensor([3. 4. 5.], shape=(3,), dtype=float32)

텐서8 = tf.constant( [3, 4, 5])
print(tf.cast(텐서8, float))
# tf.Tensor([3. 4. 5.], shape=(3,), dtype=float32)

# -----------------------------------------------------------

# Variable tensor(Weight를 저장하는 tensor)

# tf.constant() -> 고정된 숫자로 tensor을 만들 때(상수)

# tf.Varialbe() -> Variable(변수, 변하는 숫자) = Weight(새로운 W로 갱신)
w = tf.Variable(1.0)
print(w)
# <tf.Variable 'Variable:0' shape=() dtype=float32, numpy=1.0>

# tensor을 constant로 만들면 사칙연산은 가능하지만 원소를 변경하는 것은 불가능하다.
# 변동되는 W값을 저장하고 싶으면 tf.Variable()로 만들어야한다.

print(w.numpy())
# 1.0
# w에 실제 저장된 값을 쓰거나, 불러오고 싶다면 변수.numpy()

w.assign(2)
print(w.numpy())
# 2.0
# 변수에 값 수정하기 변수.assign()

w = tf.Variable( [[1, 2], [3, 4]], tf.float32 )
print(w.numpy())




