import cv2
import numpy as np
from ultralytics import YOLO
from tqdm import tqdm
from constants import *
from utils import create_directory
from DatasetStat import DatasetStat

def detect_table_contour(image):
    # 그레이스케일 변환
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 이진화
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # 윤곽선 검출
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 가장 큰 윤곽선 선택 (표라고 가정)
    table_contour = max(contours, key=cv2.contourArea)
    
    return table_contour

def crop_table(image, contour):
    x, y, w, h = cv2.boundingRect(contour)
    return image[y:y+h, x:x+w]

def detect_cells(image, model):
    results = model(image)
    return results[0].boxes.xyxy.cpu().numpy()

def save_all_contour(image):
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 이진화
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    cv2.imshow('a',binary)
    cv2.waitKey(0)
    # 윤곽선 찾기
    contours, _ = cv2.findContours(gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    # 원본 이미지 복사
    contour_image = image.copy()

    # 모든 윤곽선 그리기
    cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 2)

    # 결과 표시
    cv2.imshow('Alllll Contours', contour_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # 결과 저장
    cv2.imwrite('all_contours.png', contour_image)
def main():
    # YOLO 모델 로드 (사전에 학습된 모델 필요)
    model = YOLO('last(2).pt')
    
    # 이미지 로드
    image = cv2.imread('ocr/college/ocr2.jpg')
    save_all_contour(image)
    # 표 윤곽선 검출
    table_contour = detect_table_contour(image)
    # 원본 이미지와 같은 크기의 빈 이미지 생성
    contour_image = np.zeros(image.shape[:2], dtype=np.uint8)

    # 빈 이미지에 윤곽선 그리기
    cv2.drawContours(contour_image, [table_contour], 0, (255), 2)

    # 윤곽선 이미지 저장
    cv2.imwrite("./tee.png", contour_image)    # 표 영역 추출
    table_image = crop_table(image, table_contour)
    
    # 셀 검출
    cells = detect_cells(image, model)
    
    # 결과 시각화
    for cell in cells:
        x1, y1, x2, y2 = map(int, cell[:4])
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    # 결과 출력
    cv2.imshow('Detected Cells', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
