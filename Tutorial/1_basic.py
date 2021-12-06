'''
텐서란? 
1. 다차원 배열(Multi-dimensional Array)
2. 백터와 행렬을 일반화한 것으로, 3차원 이상으로 확장할 수 있다.
3. TensorFlow의 주요 객체이며 TensorFlow 작업은 주로 텐서의 연산으로 이루어진다.
: 텐서를 정의하고 연산을 수행하는 프레임워크
'''
import tensorflow as tf

# tf.rank()
# 텐서 객체의 랭크는 차원의 수(n-dimension)이다.
# tf.rank()는 텐서의 랭크를 반환한다.

scalar = tf.constant(1)
vector = tf.constant([1, 2, 3])
matrix = tf.constant([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
tensor = tf.constant([[[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                    [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                    [[1, 2, 3], [4, 5, 6], [7, 8, 9]]])

print(tf.rank(scalar))
print(tf.rank(vector))
print(tf.rank(matrix))
print(tf.rank(tensor))

'''
tf.Tensor(0, shape=(), dtype=int32)
tf.Tensor(1, shape=(), dtype=int32)
tf.Tensor(2, shape=(), dtype=int32)
tf.Tensor(3, shape=(), dtype=int32)
'''

#-----------------------------------------------------------------------

# tf.constant()
# tf.constant()는 상수 텐서를 만든다.

a = tf.constant(1)
b = tf.constant([2])
c = tf.constant([[1, 2], [3, 4]])
d = tf.constant([[[1, 2, 3], [4, 5, 6]],
                [[1, 2, 3], [4, 5, 6]]]
)

print(a)
print(b)
print(c)
print(d)

'''
tf.Tensor(1, shape=(), dtype=int32)
tf.Tensor([2], shape=(1,), dtype=int32)
tf.Tensor(
[[1 2]
 [3 4]], shape=(2, 2), dtype=int32)
 [[[1 2 3]
  [4 5 6]]
tf.Tensor(
[[[1 2 3]
  [4 5 6]]

 [[1 2 3]
  [4 5 6]]], shape=(2, 2, 3), dtype=int32)
'''


#-----------------------------------------------------------------------

# tf.zeros()
# tf.zeros()는 모든 요소가 0인 텐서를 만든다.
# ()안에는 만들어질 텐서의 형태(shape)를 입력한다.

a = tf.zeros(1)
b = tf.zeros([2])
c = tf.zeros([2, 3])

print(a)
print(b)
print(c)

'''
tf.Tensor([0.], shape=(1,), dtype=float32)
tf.Tensor([0. 0.], shape=(2,), dtype=float32)
tf.Tensor(
[[0. 0. 0.]
 [0. 0. 0.]], shape=(2, 3), dtype=float32)
'''

#-----------------------------------------------------------------------

# tf.ones()
# tf.ones()는 모든 요소가 1인 텐서를 만든다.
# ()안에는 만들어질 텐서의 형태(shape)를 입력한다.

a = tf.ones(3)
b = tf.ones([4])
c = tf.ones([2, 2, 2])

print(a)
print(b)
print(c)

'''
tf.Tensor([1. 1. 1.], shape=(3,), dtype=float32)
tf.Tensor([1. 1. 1. 1.], shape=(4,), dtype=float32)
tf.Tensor(
[[[1. 1.]
  [1. 1.]]

 [[1. 1.]
  [1. 1.]]], shape=(2, 2, 2), dtype=float32)
'''

#-----------------------------------------------------------------------

# tf.range()
# tf.range()는 파이선 range와 비슷하게, 주어진 범위와 간격을 갖는 숫자 시퀀스를 만든다.

a = tf.range(0, 3)
b = tf.range(1, 5, 2)

print(a)
print(b)

'''
tf.Tensor([0 1 2], shape=(3,), dtype=int32)
tf.Tensor([1 3], shape=(2,), dtype=int32)
'''

#-----------------------------------------------------------------------

# tf.linspace()
# tf.linspace()는 numpy.linspace()와 비슷하게 주어진 범위를 균일한 간격으로 나누는 시퀀스를 반환한다.

a = tf.linspace(0, 1, 3)
b = tf.linspace(0, 3, 10)

print(a)
print(b)

'''
tf.Tensor([0.  0.5 1. ], shape=(3,), dtype=float64)
tf.Tensor(
[0.         0.33333333 0.66666667 1.         1.33333333 1.66666667
 2.         2.33333333 2.66666667 3.        ], shape=(10,), dtype=float64)
'''

#-----------------------------------------------------------------------

# tensor의 사칙연산


a = tf.add(1, 2)
b = tf.subtract(10, 5)
c = tf.square(3)
d = tf.reduce_sum([1, 2, 3])
e = tf.reduce_mean([1, 2, 3])

print(a)
print(b)
print(c)
print(d)
print(e)

'''
tf.Tensor(3, shape=(), dtype=int32)
tf.Tensor(5, shape=(), dtype=int32)
tf.Tensor(9, shape=(), dtype=int32)
tf.Tensor(6, shape=(), dtype=int32)
tf.Tensor(2, shape=(), dtype=int32)
'''

#-----------------------------------------------------------------------

# Numpy 호환성
# 텐서는 Numpy Array와 비슷하지만 GPU, TPU와 같은 가속기에서 사용할 수 있고,
# 텐서의 값은 변경할 수 없다.

# 텐서 -> NUmpy Array

a = tf.constant([1, 2, 3])

print(a)
print(a.numpy())

'''
tf.Tensor([1 2 3], shape=(3,), dtype=int32)
[1 2 3]
'''

# Numpy Array -> 텐서
import numpy as np

b = np.ones(3)

print(b)
print(tf.multiply(b, 3))

'''
[1. 1. 1.]
tf.Tensor([3. 3. 3.], shape=(3,), dtype=float64)
'''