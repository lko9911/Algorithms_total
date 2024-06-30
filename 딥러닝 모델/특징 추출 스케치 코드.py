import numpy as np
import cv2 as cv
from sklearn.metrics.pairwise import cosine_similarity

# 특징 추출 함수 (fast RNN 구현)
def extract_features(image):
    # fast RNN 코드
    feature_vector = 특징 정량화
    return feature_vector

# 데이터셋 로딩 (실험 셋)
dataset = {
    'rose': extract_features(cv.imread('rose.jpg')),
    'tulip': extract_features(cv.imread('tulip.jpg')),
    'sunflower': extract_features(cv.imread('sunflower.jpg')),
    # 신품종 등록 예시
    'new_species_1': extract_features(cv.imread('new_species_1.jpg')),
}

# 입력 이미지
input_image = cv.imread('input_image.jpg')
input_feature = extract_features(input_image)

# 유사도 계산 및 가장 유사한 항목 추출
max_similarity = -1
best_match = None

for plant, feature_vector in dataset.items():
    similarity = cosine_similarity([input_feature], [feature_vector])[0][0]
    if similarity > max_similarity:
        max_similarity = similarity
        best_match = plant

# 결과 출력
print(f"입력된 사진의 식물 종류 추정 결과: {best_match}, 유사도: {max_similarity}")
