import cv2 as cv
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

# 각 객체의 RGB 값 추출
for i in range(len(boxes)):
    if i in indexes:
        x, y, w, h = boxes[i]
        roi = img[y:y+h, x:x+w]
        roi_colors = roi.reshape(-1, 3)  # ROI 내 모든 픽셀의 BGR 값 추출

        # RGB 값 범주화 및 통계 계산
        rgb_statistics = {}
        for color in roi_colors:
            rgb = tuple(color)
            rgb_category = tuple([c // 50 * 50 for c in rgb])  # 50 단위로 RGB 범주화
            if rgb_category in rgb_statistics:
                rgb_statistics[rgb_category] += 1
            else:
                rgb_statistics[rgb_category] = 1

        # 결과 출력
        print(f"Detected object {i+1}:")
        for rgb_category, count in rgb_statistics.items():
            print(f"RGB: {rgb_category} -> Count: {count}")

# 결과 이미지에 사각형 그리기 및 출력
for i in range(len(boxes)):
    if i in indexes:
        x, y, w, h = boxes[i]
        color = (0, 255, 0)
        cv.rectangle(img, (x, y), (x + w, y + h), color, 2)

# 결과 이미지 출력
#cv.imshow('Object detection by YOLO v.3', img)
#cv.waitKey(0)
#cv.destroyAllWindows()
