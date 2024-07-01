# 코드 설명 ⭐

### 1. 라이브러리
<pre><code>
import os
import urllib.request
import cv2
import numpy as np
</code></pre>

### 2. 데이터셋 디렉토리 생성

<pre><code>dataset_dir = 'path/to/dataset'
classes = ['class1', 'class2', 'class3', 'class4', 'class5', 'class6', 'class7', 'class8', 'class9', 'class10']
for cls in classes:
    os.makedirs(os.path.join(dataset_dir, cls), exist_ok=True)
</code></pre>

### 3. 이미지 다운로드 함수 정의
<pre><code>def download_image(url, save_path):
    try:
        urllib.request.urlretrieve(url, save_path)
        print(f"Downloaded image from {url} to {save_path}")
    except Exception as e:
        print(f"Error downloading image from {url}: {e}")</code></pre>

### 4. 이미지 전처리 함수 정의 및 URL (파일 경로) 리스트
<pre><code>def preprocess_image(image_path, size=(32, 32)):
    image = cv2.imread(image_path)
    image = cv2.resize(image, size)
    image = image.astype('float32') / 255.0  # 정규화
    return image

image_urls = {
    'class1': ['url1', 'url2', 'url3'],  # 실제 이미지 URL을 여기에 추가
    'class2': ['url1', 'url2', 'url3'],
    'class3': ['url1', 'url2', 'url3'],
    # 추가 클래스 및 이미지 URL 추가 가능
}</code></pre>

### 5. 데이터셋 생성및 저장
<pre><code>for cls, urls in image_urls.items():
    for idx, url in enumerate(urls):
        image_save_path = os.path.join(dataset_dir, cls, f"{cls}_{idx+1}.jpg")
        download_image(url, image_save_path)

        # 이미지 전처리
        image = preprocess_image(image_save_path)
        cv2.imwrite(image_save_path, (image * 255).astype(np.uint8))

def load_dataset(dataset_dir, classes, image_size=(32, 32)):
    images = []
    labels = []
    for cls_idx, cls in enumerate(classes):
        cls_dir = os.path.join(dataset_dir, cls)
        for image_name in os.listdir(cls_dir):
            image_path = os.path.join(cls_dir, image_name)
            image = preprocess_image(image_path, size=image_size)
            images.append(image)
            labels.append(cls_idx)
    return np.array(images), np.array(labels)

images, labels = load_dataset(dataset_dir, classes)

np.savez('cifar10_like_dataset.npz', images=images, labels=labels)</code></pre>
