# (Kaggle) 개 고양이 구분 AI 만들기
# 1. 데이터 준비와 개발환경 세팅
import os
import tensorflow as tf
import shutil
# 폴더에 이미지 갯수 세기(os 이용)
# print( len( os.listdir('./content/train/')) )
# 25000

# 이미지 전처리(이미지를 [[0, 0, 0 ... 0], [0, 0, 0 ... 0] ~ ] 숫자로 변환)
# 1) opencv 라이브러리로 이미지 숫자화 하기(반복문 필요)
# 2) tf.keras로 처리하기
  # 2-1) 폴더의 사진을 개 폴더, 고양이 폴더로 나눠야함.(수작업 vs os함수 이용)
    # 2-1-1) 일단 train안의 모든 파일명 출력

os.mkdir('../content/dataset')
os.mkdir('../content/dataset/cat')
os.mkdir('../content/dataset/dog')

for i in os.listdir('../content/train/'):
      # 2-1-2) 파일명에 cat이 들어있으면 cat 폴더로
  if 'cat' in i:
    shutil.copyfile( '../content/train/' + i, '../content/dataset/cat/' + i )
        # 2-1-3) 파일명에 dog이 들어있으면 dog 폴더로
  if 'dog' in i:
    shutil.copyfile( '../content/train/' + i, '../content/dataset/dog/' + i )
'''
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
  './content/dataset/',
  image_size=(64, 64),
  batch_size=32,
  subset='training',
  validation_split=0.2,
  seed=1234
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
  './content/dataset/',
  image_size=(64, 64),
  batch_size=32,
  subset='validation',
  validation_split=0.2,
  seed=1234
)

print(train_ds)

import matplotlib.pyplot as plt

for i, ans in train_ds.take(1):
  print(i[0])
  print(ans)
  break
 # plt.imshow( i[0].numpy().astype('uint8') )
 # plt.show()

'''