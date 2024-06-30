import os
import urllib.request

# 데이터셋을 저장할 디렉토리 생성
dataset_dir = '/path/to/dataset'
os.makedirs(dataset_dir, exist_ok=True)

# 이미지 다운로드 함수 정의
def download_image(url, save_path):
    try:
        urllib.request.urlretrieve(url, save_path)
        print(f"Downloaded image from {url} to {save_path}")
    except Exception as e:
        print(f"Error downloading image from {url}: {e}")

# 데이터셋 이미지 다운로드
image_urls = [
    "https://example.com/image1.jpg",
    "https://example.com/image2.jpg",
    "https://example.com/image3.jpg",
    # 추가 이미지 URL 추가 가능
]

for idx, url in enumerate(image_urls):
    image_save_path = os.path.join(dataset_dir, f"image_{idx+1}.jpg")
    download_image(url, image_save_path)
