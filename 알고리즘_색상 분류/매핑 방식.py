# 색상 데이터 (모델링 사용x)

import webcolors

color_map = {
    (255, 0, 0): 'red',
    (0, 255, 0): 'green',
    (0, 0, 255): 'blue',
    # 추가 색상 데이터
}

image_rgb = (100, 150, 200) # 예시 데이터

def find_closest_color(rgb_value, color_map):
    min_distance = float('inf')
    closest_color = None

    for known_rgb, color_name in color_map.items():
        # 유클리드 거리 계산
        distance = sum((x - y) ** 2 for x, y in zip(rgb_value, known_rgb))

        if distance < min_distance:
            min_distance = distance
            closest_color = color_name

    return closest_color

closest_name = find_closest_color(image_rgb, color_map)

print("입력 이미지 색상:", image_rgb)
print("가장 가까운 색상 이름:", closest_name)
