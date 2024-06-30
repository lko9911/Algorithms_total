## 코드 설명 :star:

### 1. 라이프러리 임포트 

<pre>
<code>import cv2
import numpy as np</code>
</pre>

### 2. 학습 데이터(색상 기준표) 준비 : 색상 사전 제작 자동화
<pre>
<code>color_dict = {}
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

print("최종 추가된 사전:", color_dict)</code>
</pre>

### 3. 입력한 사진 색상 정보 출력 알고리즘 
### 3-1. 전처리 x, 이미지의 평균값으로 계산

<pre>
<code>def find_closest_color(rgb_value):
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
        break</code>
</pre>

### 3-2. 이미지 전처리 적용 색상 분석 (grabCut 적용)
### 3-2-1. grabCut 알고리즘
<pre><code># 이미지 로드 및 복사본 생성
img = cv.imread('C:/test.jpg')
img_show = np.copy(img)

# 마스크 초기화
mask = np.zeros((img.shape[0], img.shape[1]), np.uint8)
mask[:, :] = cv.GC_PR_BGD

# 브러시 크기 및 색상 설정
BrushSiz = 9
LColor, RColor = (255, 0, 0), (0, 0, 255)

# 페인팅 함수 정의
def painting(event, x, y, flags, param):
    if event == cv.EVENT_LBUTTONDOWN or (event == cv.EVENT_MOUSEMOVE and flags == cv.EVENT_FLAG_LBUTTON):
        cv.circle(img_show, (x, y), BrushSiz, LColor, -1)
        cv.circle(mask, (x, y), BrushSiz, cv.GC_FGD, -1)
    elif event == cv.EVENT_RBUTTONDOWN or (event == cv.EVENT_MOUSEMOVE and flags == cv.EVENT_FLAG_RBUTTON):
        cv.circle(img_show, (x, y), BrushSiz, RColor, -1)
        cv.circle(mask, (x, y), BrushSiz, cv.GC_BGD, -1)
    cv.imshow('Painting', img_show)

# 윈도우 생성 및 콜백 함수 설정
cv.namedWindow('Painting')
cv.setMouseCallback('Painting', painting)

# 이미지 보여주기
while True:
    if cv.waitKey(1) == ord('q'):
        break

# GrabCut 적용
background = np.zeros((1, 65), np.float64)
foreground = np.zeros((1, 65), np.float64)

cv.grabCut(img, mask, None, background, foreground, 5, cv.GC_INIT_WITH_MASK)
mask2 = np.where((mask == cv.GC_BGD) | (mask == cv.GC_PR_BGD), 0, 1).astype('uint8')
grab = img * mask2[:, :, np.newaxis]

# 결과 이미지 보여주기
cv.imshow('Grab cut image', grab)
cv.waitKey(0)
cv.destroyAllWindows()</code></pre>

![원본](https://github.com/lko9911/Algorithms_total/assets/160494158/0ba93a7d-a6ea-47ae-84ad-35a838ee54c5)
![grab](https://github.com/lko9911/Algorithms_total/assets/160494158/2d1fcc7a-0061-4375-bb10-0a7f76f3d19c)
