import os
import shutil
import cv2
import numpy as np

# 데이터셋 디렉토리 생성
dataset_dir = 'C:/dataset'
classes = ['class1', 'class2']
for cls in classes:
    os.makedirs(os.path.join(dataset_dir, cls), exist_ok=True)

# 이미지 복사 함수 정의
def copy_image(source_path, save_path):
    try:
        shutil.copy(source_path, save_path)
        print(f"Copied image from {source_path} to {save_path}")
    except Exception as e:
        print(f"Error copying image from {source_path}: {e}")

# 이미지 전처리 함수 정의
def preprocess_image(image_path, size=(32, 32)):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Image not found or unable to read at {image_path}")
    image = cv2.resize(image, size)
    image = image.astype('float32') / 255.0  # 정규화
    return image

# 예시로 사용할 이미지 경로 리스트
image_paths = {
    'class1': ['C:/dataset_test/picture/토끼.jpg', 'C:/dataset_test/picture/피카츄.jpg'],
    'class2': ['C:/dataset_test/picture/그림.jpg'],
}

# 데이터셋 생성
for cls, paths in image_paths.items():
    for idx, source_path in enumerate(paths):
        image_save_path = os.path.join(dataset_dir, cls, f"{cls}_{idx+1}.jpg")
        copy_image(source_path, image_save_path)

        # 이미지 전처리
        image = preprocess_image(image_save_path)
        cv2.imwrite(image_save_path, (image * 255).astype(np.uint8))

# 데이터셋 로드 및 저장 예시
def load_dataset(dataset_dir, classes, image_size=(32, 32)):
    images = []
    labels = []
    for cls_idx, cls in enumerate(classes):
        cls_dir = os.path.join(dataset_dir, cls)
        for image_name in os.listdir(cls_dir):
            image_path = os.path.join(cls_dir, image_name)
            try:
                image = preprocess_image(image_path, size=image_size)
                images.append(image)
                labels.append(cls_idx)
            except ValueError as e:
                print(e)
    return np.array(images), np.array(labels)

# 데이터셋 로드
images, labels = load_dataset(dataset_dir, classes)

# 데이터셋을 npz 파일로 저장
np.savez('dataset_test.npz', images=images, labels=labels)
