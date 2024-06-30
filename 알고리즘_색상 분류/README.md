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

### 3-2-2. YOLO v3 (spp) 알고리즘

<pre><code>import cv2 as cv
import numpy as np

# YOLO 모델 로드
net = cv.dnn.readNet("C:/DL/opencv/yolov3-spp.weights", "C:/DL/opencv/yolov3-spp.cfg")

# 클래스 이름 로드
with open("C:/DL/opencv/coco_names.txt", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# 이미지 로드
image_path = 'C:/test.jpg'
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
        if confidence > 0.5:  # 신뢰도 임계값
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)
            x = int(center_x - w / 2)
            y = int(center_y - h / 2)
            boxes.append([x, y, w, h])
            confidences.append(float(confidence))
            class_ids.append(class_id)

# Non-maximum suppression 적용하여 중복 박스 제거
indexes = cv.dnn.NMSBoxes(boxes, confidences, score_threshold=0.5, nms_threshold=0.4)

# 각 객체의 RGB 값 계산 및 출력
for i in range(len(boxes)):
    if i in indexes:
        x, y, w, h = boxes[i]
        label = str(classes[class_ids[i]])
        color = (0, 255, 0)
        cv.rectangle(img, (x, y), (x + w, y + h), color, 2)
        roi = img[y:y+h, x:x+w]
        mean_rgb = cv.mean(roi)[:3][::-1]  # BGR을 RGB로 변환
        text = f"{label}: ({mean_rgb[0]}, {mean_rgb[1]}, {mean_rgb[2]})"
        cv.putText(img, text, (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

# 결과 이미지 파일로 저장
cv.imwrite('detected_objects.jpg', img)

# 결과 이미지 출력
cv.imshow('Object detection by YOLO v.3', img)
cv.waitKey(0)
cv.destroyAllWindows()</code></pre>

![detected_objects](https://github.com/lko9911/Algorithms_total/assets/160494158/4d2b536e-d068-418a-acd7-3393b30d9b74)
