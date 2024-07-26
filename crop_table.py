import cv2
import numpy as np
from sympy import im
import torch
from PIL import Image
from transformers import pipeline
import os
from config import *

def preprocess_image(image_path):
    # 이미지 읽기
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    
    # PNG 파일의 알파 채널 처리
    if img.shape[2] == 4:
        # 알파 채널이 있는 경우, 배경을 흰색으로 설정
        alpha = img[:,:,3]
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        white_background = np.ones_like(img, dtype=np.uint8) * 255
        alpha_factor = alpha[:,:,np.newaxis].astype(np.float32) / 255.0
        img = (1 - alpha_factor) * white_background + alpha_factor * img
        img = img.astype(np.uint8)
    
    # 그레이스케일로 변환
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 가우시안 블러로 노이즈 제거
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # 이진화
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # 모폴로지 연산으로 노이즈 제거 및 선 강화
    kernel = np.ones((3,3), np.uint8)
    morphed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    # 결과 이미지 저장
    preprocessed_path = os.path.splitext(image_path)[0] + '_preprocessed.png'
    cv2.imwrite(preprocessed_path, morphed)
    
    return Image.fromarray(morphed), preprocessed_path

def detect_tables(image):
    detector = pipeline("object-detection", model="microsoft/table-transformer-detection")
    results = detector(image)
    return results

def expand_box(box, image_size, expand_ratio=0.05):
    width, height = image_size
    x1, y1, x2, y2 = box['xmin'], box['ymin'], box['xmax'], box['ymax']
    
    box_width = x2 - x1
    box_height = y2 - y1
    
    expand_x = box_width * expand_ratio
    expand_y = box_height * expand_ratio
    
    x1 = max(0, x1 - expand_x)
    y1 = max(0, y1 - expand_y)
    x2 = min(width, x2 + expand_x)
    y2 = min(height, y2 + expand_y)
    
    return {'xmin': x1, 'ymin': y1, 'xmax': x2, 'ymax': y2}

def crop_and_save_tables(image, detections, output_folder, expand_ratio, original_format):
    os.makedirs(output_folder, exist_ok=True)
    for i, detection in enumerate(detections):
        box = expand_box(detection['box'], image.size, expand_ratio)
        x1, y1, x2, y2 = box['xmin'], box['ymin'], box['xmax'], box['ymax']
        cropped_table = image.crop((x1, y1, x2, y2))
        output_path = f"{output_folder}/.{original_format}"
        cropped_table.save(output_path)
import sys

input_image = sys.argv[1]
output_folder = sys.argv[2]

# 파일 확장자 확인
_, file_extension = os.path.splitext(input_image)
original_format = file_extension[1:].lower()  # 확장자에서 점(.) 제거

if original_format not in ['jpg', 'jpeg', 'png', 'bmp', 'tiff']:
    print("지원되지 않는 이미지 형식입니다. jpg, jpeg, png, bmp, tiff 형식을 사용해주세요.")
    exit()

# 사용자로부터 확장 비율 입력 받기
expand_ratio = float(0.05)

try:
    # 이미지 전처리
    preprocessed_image, preprocessed_path = preprocess_image(input_image)
    
    # 전처리된 이미지로 테이블 감지
    detections = detect_tables(input_image)
    
    # 원본 이미지 로드 (크롭을 위해)
    original_image = Image.open(input_image)
    
    # 감지된 테이블 크롭 및 저장 (원본 이미지에서)
    crop_and_save_tables(original_image, detections, output_folder, expand_ratio, original_format)
    
    print(f"감지된 테이블 수: {len(detections)}")
    print(f"크롭된 테이블 이미지가 {output_folder} 폴더에 저장되었습니다.")
    print(f"바운딩 박스가 {expand_ratio*100}% 확장되었습니다.")

except Exception as e:
    print(f"오류가 발생했습니다: {str(e)}") 