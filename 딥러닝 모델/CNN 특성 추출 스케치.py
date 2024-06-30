import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# CNN 기본 모델 (파라미터 정의 x)
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(3, activation='softmax')  # 분류할 특징의 개수에 맞게 출력층 설정 (예: 수피 색, 갈라짐 정도, 처짐 정도)
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 데이터 준비
(x_train, y_train),(x_test, y_test) = 데이터셋 로드
...
...
...

# 모델 학습
model.fit(x_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

# 학습된 모델을 사용하여 특징 추출하는 함수 정의
def extract_features(image_path):
    # 이미지를 적절한 형식으로 전처리 (예시로는 더미 처리)
    image = preprocess_image(image_path)
    # 전처리된 이미지를 모델에 입력하여 특징 추출
    features = model.predict(image)
    return features

# 이미지 전처리 함수 (YOLO 코드 사용)
def preprocess_image(image_path):
    # 이미지 로드 및 사이즈 조정
    image = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
    image = tf.keras.preprocessing.image.img_to_array(image)
    image = np.expand_dims(image, axis=0)
    return image

# 입력된 사진의 특징 추출
input_image_path = 'path_to_your_input_image.jpg'
extracted_features = extract_features(input_image_path)
print("Extracted features:", extracted_features)


