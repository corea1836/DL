# 뉴런층의 속성 확인하기

# tf.keras.layers.Layer는 NN의 모든 레이어 객체가 상속하는 클래스이다.
# tf.keras.layers.Layer의 다양한 속성(Attribute)을 이용해 각 레이어에 대한 정보를 확인할 수 있다.

import tensorflow as tf
tf.random.set_seed(0)

# 1) 뉴런층의 이름과 자료형

# 1. 뉴런층 만들기
input_layer = tf.keras.layers.InputLayer(input_shape=(3, ))
hidden_layer = tf.keras.layers.Dense(units=4, activation='relu')
output_layer = tf.keras.layers.Dense(units=2, activation='softmax')

# 2. 모델 구성하기
model = tf.keras.Sequential([
  input_layer,
  hidden_layer,
  output_layer
])

# 3. 모델 컴파일하기
model.compile(loss='mse', optimizer='Adam')

# 4. 뉴런층의 이름과 자료형
print(input_layer.name, input_layer.dtype)
print(hidden_layer.name, hidden_layer.dtype)
print(output_layer.name, output_layer.dtype)

'''
input_1 float32
dense float32
dense_1 float32
'''
# name은 뉴런층의 이름
# dtype은 뉴런층의 연산과 Weight값에 사용되는 자료형

print(model.layers[0].name)
print(model.layers[1].name)
# print(model.layers[2].name)

'''
dense
dense_1
Traceback (most recent call last):
  File "/Users/jamiehong/Documents/tensorflow/Tutorial/6_Layer.py", line 41, in <module>
    print(model.layers[2].name)
IndexError: list index out of range
'''
# model.layers는 구성한 NN 모델의(입력층을 제외한) 뉴런층 레이어 객체를 리스트 형태로 반환한다.
# model.layers[0]은 모델의 첫번째 뉴런층인 hidden layer이다.
# model.layers[1]은 모델의 두번째 뉴런층인 output_layer이다.
# 입력층을 제외하고 레이어는 2개(hidden, output)를 구성하였기 때문에 index : 2는 out of range가 된다.

'''
print(model.layers)
# [<keras.layers.core.Dense object at 0x7fe6abd24280>, <keras.layers.core.Dense object at 0x7fe6abd24610>]
'''

#---------------------------------------------------------------------------------------------------------

# 2) 뉴런층의 입력과 출력

print(input_layer.input)
print(input_layer.output)

print(hidden_layer.input)
print(hidden_layer.output)

print(hidden_layer.input.shape)
print(hidden_layer.output.shape)

print(output_layer.input)
print(output_layer.output)

'''
KerasTensor(type_spec=TensorSpec(shape=(None, 3), dtype=tf.float32, name='input_1'), name='input_1', description="created by layer 'input_1'")
KerasTensor(type_spec=TensorSpec(shape=(None, 3), dtype=tf.float32, name='input_1'), name='input_1', description="created by layer 'input_1'")
KerasTensor(type_spec=TensorSpec(shape=(None, 3), dtype=tf.float32, name='input_1'), name='input_1', description="created by layer 'input_1'")
KerasTensor(type_spec=TensorSpec(shape=(None, 4), dtype=tf.float32, name=None), name='dense/Relu:0', description="created by layer 'dense'")
(None, 3)
(None, 4)
KerasTensor(type_spec=TensorSpec(shape=(None, 4), dtype=tf.float32, name=None), name='dense/Relu:0', description="created by layer 'dense'")
KerasTensor(type_spec=TensorSpec(shape=(None, 2), dtype=tf.float32, name=None), name='dense_1/Softmax:0', description="created by layer 'dense_1'")
'''

# input은 뉴런층의 입력 텐서이다.
# output은 뉴련층의 출력 텐서이다.
# hidden layer의 입력과 출력형태(shape)를 출력해보면,
# 입력 텐서는 길이 3의 형태, 출력 텐서는 길이 4의 형태를 가진다.
# ex) (None, 3)은 길이 3의 벡터의 시퀀스 형태임

#---------------------------------------------------------------------------------------------------------

# 3) 뉴런층의 활성화함수(activation)

print(hidden_layer.activation)
print(hidden_layer.activation.__name__)
print(output_layer.activation)
print(output_layer.activation.__name__)

'''
<function relu at 0x7f9869434310>
relu
<function softmax at 0x7f9868faf310>
softmax
'''

# activation은 뉴런 노드의 활서화함수(Activation function)를 나타낸다.
# hidden layer = Relu
# output layer = SoftMax

#---------------------------------------------------------------------------------------------------------

# 4) 뉴런층의 가중치

print(hidden_layer.weights)
print(output_layer.weights)

'''
[<tf.Variable 'dense/kernel:0' shape=(3, 4) dtype=float32, numpy=
array([[-0.3851872 , -0.54333335,  0.0655309 ,  0.1134268 ],
       [-0.15428883,  0.5699866 , -0.01254469,  0.9223561 ],
       [ 0.36428273, -0.6936733 ,  0.38850498,  0.30073535]],
      dtype=float32)>, <tf.Variable 'dense/bias:0' shape=(4,) dtype=float32, numpy=array([0., 0., 0., 0.], dtype=float32)>]
[<tf.Variable 'dense_1/kernel:0' shape=(4, 2) dtype=float32, numpy=
array([[ 0.11082816, -0.55741405],
       [ 0.7298498 ,  0.5545671 ],
       [ 0.29023337,  0.0607245 ],
       [-0.971118  ,  0.74701834]], dtype=float32)>, <tf.Variable 'dense_1/bias:0' shape=(2,) dtype=float32, numpy=array([0., 0.], dtype=float32)>]
'''

# weights를 사용해 각 뉴런층의 새닙스 가중치에 대한 정보를 얻을 수 있다.

#---------------------------------------------------------------------------------------------------------

# 5) get_weights()의 메서드

print(hidden_layer.get_weights())
print(output_layer.get_weights())

'''
[array([[-0.3851872 , -0.54333335,  0.0655309 ,  0.1134268 ],
       [-0.15428883,  0.5699866 , -0.01254469,  0.9223561 ],
       [ 0.36428273, -0.6936733 ,  0.38850498,  0.30073535]],
      dtype=float32), array([0., 0., 0., 0.], dtype=float32)]
[array([[ 0.11082816, -0.55741405],
       [ 0.7298498 ,  0.5545671 ],
       [ 0.29023337,  0.0607245 ],
       [-0.971118  ,  0.74701834]], dtype=float32), array([0., 0.], dtype=float32)]
'''

# get_weights() 메서드를 사용하면 시냅스 가중치를 Numpy Array 형태로 얻을 수 있다.
