import tensorflow as tf
import requests
requests.packages.urllib3.disable_warnings()
import ssl
import matplotlib.pyplot as plt

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    # Legacy Python that doesn't verify HTTPS certificates by default
    pass
else:
    # Handle target environment that doesn't support HTTPS verification
    ssl._create_default_https_context = _create_unverified_https_context

( trainX, trainY ), ( testX, testY )= tf.keras.datasets.fashion_mnist.load_data()
# 구글이 호스팅해주는 데이터셋 중 하나(= 텐서플로우 라이브러리에서 기본적으로 제공하는 데이터셋 중 하나)
# 안에는 ( ( trainX : input_data, trainY : answer ), ( testX, testY ) ) 이렇게 튜플로 묶인
# 두 개의 데이터셋 묶음이 들어있다.

# print(trainX[0])
# 예시를 위해 trainX의 첫째 이미지 확인
'''
[[  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
    0   0   0   0   0   0   0   0   0   0]
    .
    .
    .
  [  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
     0   0   0   0   0   0   0   0   0   0]]
'''
# 이미지 한개는 28 * 28 -> 가로 28줄인 데이터가 세로로도 28줄 있다.
'''
그림으로 보면
가로 : 28개
[0, 0, 0, ... 0, 0, 0]
[0, 0, 0, ... 0, 0, 0]
.
. 세로 : 28개
.
[0, 0, 0, ... 0, 0, 0]

하나의 큰 행렬로 구성되어 있다.
행렬 안의 숫자들은 각각 생상정보를 띄고 있는 것(ex. 0이면 black)
'''
# print(trainX.shape)
# shape는 numpy array나 tensor 자료형에서 쓸 수 있다. 이런 자료형에 몇개나 들어있는지 궁금하면 쓰자.
# (60000, 28, 28) -> [28개의 숫자가 들어있는 리스트]가 28개 있다. = 행렬(이미지를 숫자화) 하나가 6만개 있다.

# print(trainY)
# [9 0 0 ... 3 0 5]
# Y는 정답이 하나의 리스트에 들어있다.(lable)
# class_names = ['T-shirt/top', 'Trouser', .... ] 정답은 이런식으로 10가지 의류 카테고리로 분류된다.
# 어떤 사진이 T-shirt라면 0으로 마킹 되어있다.
# 정답을 정수로 치환해서 리스트로 만들어 놓은게 Y

# 이미지를 파이썬으로 띄워보는법
'''
import matplotlib.pyplot as plt
plt.imshow(trainX[50])
plt.gray()
plt.colorbar()
plt.show()
'''


# 딥러닝 순서
# 1. 모델 만들기
model = tf.keras.Sequential([
  tf.keras.layers.Dense(128, input_shape=(28, 28), activation='relu'),
# NN은 activation function을 적용해야 한다.
# relu -> 음수값은 다 0으로 만들어준다.
# convolution layer에서 자주 쓴다.(이미지는 0 ~ 255까지의 정수이므로 음수가 나오면 안된다.)

  tf.keras.layers.Dense(64, activation='relu'),
  tf.keras.layers.Flatten(),
  # (None, 28, 10)
  # 2, 3차원 행렬을 1차원으로 압축해주는 Flatten 레이어(결과가 1차원 이므로)
  tf.keras.layers.Dense(10, activation='softmax'),
# output(원하는 예측결과)은 T-shirt/Trouser/Pullover...일 확률
# 확률예측문제라면 
# 1. 일단 마지막 레이어 노드수를 카테고리 갯수만큼 설정하자.
# [ 0.2  0.4  0.1  ... ] -> Trouser일 확률이 40% 정도다 라는걸 알 수 있다.
# 2. loss 함수는 cross entropy를 쓰자.

# 확률을 예측하고 싶을 때 sigmoid / softmax 등을 쓰는데 둘의 차이점
# sigmoid : 결과를 0 ~ 1로 압축
#           binary 예측문제에서 사용(정답이 0인지 1인지 예측하고 싶을 때 : 대학원 붙는다 / 안붙는다)
#           마지막 노드 갯수는 1개로 설정
# softmax : 결과를 0 ~ 1로 압축
#           카테고리 예측문제에서 사용(여러 카테고리가 있는데 어떤 카테고리에 속할 확률이 높은지 예측하고 싶을 때)
#           마지막 노드 갯수는 카테고리 갯수로 설정
#           softmax 함수는 카테고리 확률을 다 더하면 1이 나온다.
])

# 2. compile 하기
model.compile( loss='sparse_categorical_crossentropy', optimizer = 'adam', metrics=['accuracy'] )
# (여러 카테고리가 있을 때) 카테고리 예측문제에서 쓰는 loss function
# categorical_crossentropy -> 답안(trainY)이 원핫인코딩이 되어있을 때 사용
# sparse_categorical_crossentropy -> 답안이 정수로 인코딩 되어있을 때 사용

# 2-1. 모델 아웃라인 출력해보기(모델을 잘 짰는지 한 번에 볼 수 있는 법)
# 학습시엔 layers.Dense에 input_shape가 없어도 keras가 알아서 판단해준다.
# But, summary를 보고싶으면 input_shape(집어넣을 데이터 하나의 모양)를 지정해줘야 한다.
model.summary()
'''
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
dense (Dense)                (None, 28, 128)           3712      
_________________________________________________________________
dense_1 (Dense)              (None, 28, 64)            8256      
_________________________________________________________________
dense_2 (Dense)              (None, 28, 10)            650       
=================================================================
Total params: 12,618
Trainable params: 12,618
Non-trainable params: 0

해석하는 법
Output Shape : (None, 28, 128) -> 이 레이어를 거치고 나면 이런 shape의 노드들이 남는다.
                (128개의 데이터)리스트가 28개 있고, 이게 None만큼(아직 모르지만 많다. input수만큼)
                 2d Dense layer(3d도 만들 수 있다.)
Param : 이 레이어에 있는 학습가능한 w 갯수

Output 모양이 조금 이상하다. (None, 28, 10) -> 내가 원하는건 1차원 리스트
-----------------------------------------------------------------
Flatten() 이후

flatten (Flatten)            (None, 1792)              0         
_________________________________________________________________
dense_2 (Dense)              (None, 10)                17930     
'''
# exit()

# 3. fit 하기
model.fit(trainX, trainY, epochs=5)
# 이대로 학습하면 accuracy가 조금 낮다.

#----------------------------------------------------------------------------------------------------------------

# 이미지를 딥러닝 모델에 넣을 때는 숫자 행렬(2차원)로 변환해서 넣는데,
# 이때 결과는 1차원 행렬을 원하기 때문에 한줄로 쭉 늘여셔(Faltten레이어) 결과를 얻는다.
# 이건 이미지를 해체해서 딥러닝을 돌리는 것과 같다.
# 이렇게 하면 예측모델의 응용력도 없어진다.
# 같은 픽셀의 이미지로 학습해도 조금만 위치를 변경하면 전에 학습한 가중치가 쓸모 없어진다.

#----------------------------------------------------------------------------------------------------------------

# 해결책 : convolutional layer(이미지를 이미지 그 자체로 분석하자, 일렬로 해체하지 말고 - flatten layer에 대한 대책)

# convolutional layer이란?
# 1. 이미지에서 중요한 정보를 추려서 복사본 레이어를 20 ~ 30개 만든다.
# 2. 각 복사본에는 이지미의 중요한 feature, 특성(가로줄 강조, 세로줄 강조, 동그란 부분 강조 등등)이 담겨있다.
# 3. 이걸로 학습을 하자 : feature extraction

# 전통적인 ML 사물 인식문제에서는 feature extraction이 꼭 필요하다. = 전통적인 사물인식(자동차) ML기법(사전 가이드가 필요하다.)
# feature extraction과 유사한 작업이 convolutional layer(각각 다른 특성이 강조된 이미지 복사본 20 ~ 30장)

#----------------------------------------------------------------------------------------------------------------

# Convolutional layer(feature map 만들기)
# 5*5의 이미지를 3*3으로 압축시킬 때 영역의 처음부터 끝까지(2차원 순회하듯이) 3*3 영역 중 가장 중요한 부분을 압축해서
# 새로운 3*3이미지로 쓴다.(ex. 중요한 색상을 뽑아서 만든다.)

# 중요한 이미지를 뽑는 과정
# 1. kernel을 디자인 한다.
# ex. 세로선 강조된 feature map을 만들고 싶다.
#     shapen kernel : 선명도 업
#     gaussian blur kernel : 블로어 효과
# 2. 이런 특성이 강조된 이미지들로 NN이 학습을 할 수 있다.(각 요소별 특성 파악을 쉽게 할 수 있다.)

# But, 단순 convolutional의 문제점 : feature의 위치
# ex. 사진 왼쪽 하단에 위치한 자동차 사진만 학습하면, 
# 사진에서 차의 위치가 변하면 예측확률이 현저히 떨어진다.

#----------------------------------------------------------------------------------------------------------------

# 해결책 : Pooling layer(Down sampling)
# 이미지의 크기를 축소하자.(그냥 이미지의 크기만 축소하는게 아니라 이미지의 중요한 부분은 추려서 유지한채로 가운데로 모아서 축소하자.)
# Max Pooling : 최대값만 추리자.
# Average Pooling : 평균값으로 추리자.

# 장점 : translation invariance(응용력 있는 모델)이 된다. = 이미지가 어디에 있던간에 잘 인식한다.
#       Convolutional + Pooling layer를 도입하면 특징추출 + 특징을 가운데로 모아준다.
#       차를 예로들면 차가 사진의 오른쪽 하단에 있던, 왼쪽 상단에 있던 인식을 잘한다.(결국 feature들이 가운데로 모아준다.)

# 즉 이러한 특성을 적용하면 Convolution Neural Network(CNN)이 된다.

# CNN의 일반적인 구성법
# Input -> ( Convolutional layer + Pooling )-> Nueral Network
# Filter(kernel)을 내가 정할 수도 있지만 keras가 알아서 해준다.