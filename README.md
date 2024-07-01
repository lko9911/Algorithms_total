## 1. 이미지 전처리 🐬
- 사진에서 객체만 추출하기
- 추출된 데이터 분류및 데이터 셋 작업

딥러닝 : YOLO / 사용자 지정 : grabCut 

## 2. 색상 알고리즘 🛩️
- 색상 등록 (표준, 사진과 이미지)
- 색상을 판단할 사진 입력 (전처리x)
- 색상을 판단할 사진 입력 (grabCut 적용 이미지)
<br>


# 모델링 계획 :star:

### 목표  
- 객체 탐지 (YOLO-spp 재학습)  
- 탐지된 식물 영역에 대해 식물의 여러 특성 및 품종 예측 (수치화)

### 1단계 : 데이터셋 준비 ✔️
- 이미지 데이터의 저장 형태
<pre><code>/dataset
    /species_1
        - img1.jpg
        - img2.jpg
        ...
    /species_2
        - img1.jpg
        - img2.jpg
        ...
    ...
</code></pre>
- 특성 레이블 데이터
<pre><code>/dataset/crack_labels.csv</code></pre>
![화면 캡처 2024-07-01 141804](https://github.com/lko9911/Algorithms_total/assets/160494158/29bc8ea7-bc19-4e75-bb50-d5da28bd9d66)

### 2단계 : YOLO 모델 재학습 ✔️
준비물 : 'cfg', 'weights', 'obj.names', 'obj.data' 어노테이션 파일

- YOLO 설정 파일 수정
<pre><code>convolutional]
filters=21  # (classes + 5) * 3

[yolo]
classes=2</code></pre>

- YOLO 모델 학습
<pre><code>./darknet detector train data/obj.data cfg/yolov3.cfg yolov3.conv.74
</code></pre>

### 3단계 : 품종 예측 모델 (CNN) ✔️
- 데이터셋 불러오기
<pre><code>import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# 특성 레이블 로드
labels_df = pd.read_csv('dataset/crack_labels.csv')

# 이미지 경로와 특성 레이블을 분리
image_paths = labels_df['image_path'].values
crack_levels = labels_df['crack_level'].values
color_intensities = labels_df['color_intensity'].values
leaf_sizes = labels_df['leaf_size'].values

# 데이터 전처리
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_generator = datagen.flow_from_directory(
    'dataset',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='training')

validation_generator = datagen.flow_from_directory(
    'dataset',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='validation')
</code></pre>
- 데이터 학습 (CNN모델 사용)
<pre><code>import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.models import Model

# 입력 레이어
input_layer = Input(shape=(224, 224, 3))

# 합성곱 레이어
x = Conv2D(32, (3, 3), activation='relu')(input_layer)
x = MaxPooling2D((2, 2))(x)
x = Conv2D(64, (3, 3), activation='relu')(x)
x = MaxPooling2D((2, 2))(x)
x = Conv2D(128, (3, 3), activation='relu')(x)
x = MaxPooling2D((2, 2))(x)
x = Flatten()(x)

# 품종 예측 출력 레이어
num_species_classes = len(train_generator.class_indices)
output_species = Dense(num_species_classes, activation='softmax', name='species_output')(x)

# 특성 예측 출력 레이어 1: 껍질 갈라짐 정도
num_crack_levels = len(set(crack_levels))
output_crack = Dense(num_crack_levels, activation='softmax', name='crack_output')(x)

# 특성 예측 출력 레이어 2: 채도
output_color = Dense(1, activation='linear', name='color_output')(x)

# 특성 예측 출력 레이어 3: 잎 크기
output_leaf = Dense(1, activation='linear', name='leaf_output')(x)

# 모델 정의
model = Model(inputs=input_layer, outputs=[output_species, output_crack, output_color, output_leaf])

# 모델 컴파일
model.compile(optimizer='adam',
              loss={'species_output': 'categorical_crossentropy',
                    'crack_output': 'categorical_crossentropy',
                    'color_output': 'mean_squared_error',
                    'leaf_output': 'mean_squared_error'},
              metrics={'species_output': 'accuracy', 'crack_output': 'accuracy', 'color_output': 'mae', 'leaf_output': 'mae'})

model.summary()

# 모델 학습
history = model.fit(
    train_generator,
    epochs=10,
    validation_data=validation_generator
) 
</code></pre>

### 4단계 : YOLO + 특성 분석 결합 ✔️
- 탐지된 식물 영역에 대해 특성 분석을 수행
<pre><code>import cv2 as cv
import numpy as np
from tensorflow.keras.preprocessing import image

# YOLO 모델 로드
net = cv.dnn.readNet("yolov3.weights", "yolov3.cfg")
with open("obj.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# 이미지 로드
image_path = 'path_to_your_image.png'
img = cv.imread(image_path)
height, width, channels = img.shape

# 이미지 전처리
blob = cv.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
net.setInput(blob)
outs = net.forward(net.getUnconnectedOutLayersNames())

# 물체 검출
class_ids = []
confidences = []
boxes = []

for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.5:
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)
            x = int(center_x - w / 2)
            y = int(center_y - h / 2)
            boxes.append([x, y, w, h])
            confidences.append(float(confidence))
            class_ids.append(class_id)

# Non-maximum suppression
indexes = cv.dnn.NMSBoxes(boxes, confidences, score_threshold=0.5, nms_threshold=0.4)

# 각 객체에 대해 특성 예측
for i in range(len(boxes)):
    if i in indexes:
        x, y, w, h = boxes[i]
        roi = img[y:y+h, x:x+w]
        roi_resized = cv.resize(roi, (224, 224))
        img_array = image.img_to_array(roi_resized)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        # 특성 예측
        species_prediction, crack_prediction = model.predict(img_array)
        species_class = np.argmax(species_prediction, axis=1)[0]
        crack_level = np.argmax(crack_prediction, axis=1)[0]

        # 결과 표시
        label = f"Species: {species_class}, Crack Level: {crack_level}"
        color = (0, 255, 0)
        cv.rectangle(img, (x, y), (x + w, y + h), color, 2)
        cv.putText(img, label, (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

# 결과 이미지 파일로 저장
cv.imwrite('predicted_objects.jpg', img)

# 결과 이미지 출력
cv.imshow('Object detection and prediction', img)
cv.waitKey(0)
cv.destroyAllWindows()</code></pre>

### 5단계 : 모델 평가 ✔️
- 혼돈행렬 리포트
- k 교차 검증
- ROC 곡선

### 데이터 입력 
<pre><code>from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# 학습된 모델 파일 경로
model_path = 'path_to_your_trained_model.h5'

# 모델 로드
model = load_model(model_path)

# 입력 이미지 경로
image_path = 'path_to_your_image.jpg'

# 이미지 불러오기 및 전처리
img = image.load_img(image_path, target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0) / 255.0  # 모델에 맞는 입력 형식으로 정규화

# 예측 수행
predictions = model.predict(img_array)

# 예측 결과 해석
species_prediction = np.argmax(predictions[0])
crack_level_prediction = np.argmax(predictions[1])
color_intensity_prediction = predictions[2]
leaf_size_prediction = predictions[3]

print(f'Species Prediction: {species_prediction}')
print(f'Crack Level Prediction: {crack_level_prediction}')
print(f'Color Intensity Prediction: {color_intensity_prediction}')
print(f'Leaf Size Prediction: {leaf_size_prediction}')</code></pre>
