import cv2 as cv
import numpy as np

# 이미지 로드 및 복사본 생성
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
cv.destroyAllWindows()
