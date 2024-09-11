import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
def resize_image(image, scale_factor):
    width = int(image.shape[1] * scale_factor)
    height = int(image.shape[0] * scale_factor)
    dim = (width, height)
    resized = cv2.resize(image, dim, interpolation=cv2.INTER_CUBIC)
    return resized
def preprocess_image(image_path, scale_factor=1.0):
    # 1. 이미지 읽기
    image = cv2.imread(image_path)
    
    # 2. 리사이징 (필요한 경우)
    if scale_factor != 1.0:
        image = resize_image(image, scale_factor)
    
    # 3. 그레이스케일로 변환
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 4. 노이즈 제거 (가우시안 블러 또는 미디안 필터)
    denoised = cv2.GaussianBlur(gray, (5, 5), 0)
    #denoised = cv2.medianBlur(gray, 1)
    
    # 5. 대비 향상 (CLAHE)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(denoised)
    

    # 7. 적응적 이진화
    binary = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    
    # 8. 모폴로지 연산
    kernel = np.ones((3,3), np.uint8)
    morph = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
    morph = cv2.dilate(morph, kernel, iterations=1)
    morph = cv2.erode(morph, kernel, iterations=1)
    
    # 9. 3채널로 변환 (YOLO 모델 입력을 위해)
    result = cv2.cvtColor(denoised, cv2.COLOR_GRAY2BGR)
    
    return result
def remove_noise(image):
    return cv2.medianBlur(image, 3)
def enhance_edges(image):
    edges = cv2.Canny(image, 100, 200)
    return cv2.addWeighted(image, 0.8, edges, 0.2, 0)

def sharpen_image(image):
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    sharpened = cv2.filter2D(image, -1, kernel)
    return sharpened

# 모델 로드
model_path = "last(2).pt"
model = YOLO(model_path)

# 이미지 경로
image_path = "OCR/college/ocr3.jpg"

# 이미지 전처리
preprocessed_image = preprocess_image(image_path)

# 전처리된 이미지 저장 (선택사항)
cv2.imwrite("preprocessed_image.jpg", preprocessed_image)

# 전처리된 이미지로 YOLO 모델 실행
results = model(preprocessed_image, conf=0.8)

# 결과 시각화 (라벨 없이)
for r in results:
    im_array = r.plot(labels=False, line_width=2)
    im = Image.fromarray(im_array[..., ::-1])  # BGR to RGB
    im.show()  # 이미지 표시
    im.save("result_no_labels.jpg")  # 결과 이미지 저장

# 결과 분석
for r in results:
    print(r.boxes)  # 바운딩 박스 정보 출력
    print(r.boxes.xyxy)  # 박스 좌표
    print(r.boxes.conf)  # 신뢰도 점수
    print(r.boxes.cls)  # 클래스 레이블
