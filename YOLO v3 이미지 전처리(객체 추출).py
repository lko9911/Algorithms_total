# YOLO v3을 이용한 이미지에서 객체 추출하기

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

# 선택한 객체의 ROI 추출
selected_rois = []
for i in range(len(boxes)):
    if i in indexes:
        x, y, w, h = boxes[i]
        roi = img[y:y+h, x:x+w]
        selected_rois.append(roi)

# selected_rois를 다른 모델의 입력으로 사용하여 처리
for roi in selected_rois:
    # 예시: 각 ROI에 대해 여기서 추가적인 처리를 수행할 수 있음
    # 예를 들어, 이미지 분류 모델에 넣어서 클래스 분류를 수행하는 등의 작업이 가능
    pass

# 예시: selected_rois를 다른 모델의 입력으로 사용하여 예측 수행
# 예를 들어, 이미지 분류 모델을 사용하여 각 ROI에 대해 클래스를 예측하는 등의 작업을 수행할 수 있음

# 처리된 결과 출력 또는 저장
# write 입력

# 결과 이미지 출력
cv.imshow('Object detection by YOLO v.3', img)
cv.waitKey(0)
cv.destroyAllWindows()
