# 학습 데이터 준비 - color 사전 추가

import cv2
import numpy as np

color_dict = {}

while True:
    
    # image_path = 'C:/test.jpg'

    image_path = input('파일 경로 입력 :')
    label_input = input('해당 색상의 이름 :')
    
    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        print("이미지 로드 실패")
        break
    
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    color_dict[label_input] = tuple(image_rgb.mean(axis=(0, 1)).astype(int))
    print("추가된 사전:", color_dict)
    
    user_input = input("추가 여부 (y/n): ").lower()
    if user_input != 'y':
        break

print("최종 추가된 사전:", color_dict)

# 입력한 사진 색상 정보 출력 알고리즘 (전처리 x, 이미지의 평균값으로 계산)

def find_closest_color(rgb_value):
    min_distance = float('inf')
    closest_color = None

    for color_name, known_rgb in color_dict.items():

        distance = np.linalg.norm(np.array(rgb_value) - np.array(known_rgb))
        if distance < min_distance:
            min_distance = distance
            closest_color = color_name

    return closest_color

while True:
    
    # image_path = 'C:/test.jpg'

    image_path = input('파일 경로 입력 :')
    
    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        print("이미지 로드 실패")
        break
    
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    closest_color = find_closest_color(tuple(image_rgb.mean(axis=(0, 1)).astype(int)))

    print(f"주어진 사진의 가장 가까운 색상 이름은 '{closest_color}'입니다.")
    print(tuple(image_rgb.mean(axis=(0, 1)).astype(int)))
    
    user_input = input("추가 실행 여부 (y/n): ").lower()
    if user_input != 'y':
        break
