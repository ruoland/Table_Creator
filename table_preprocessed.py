import cv2
import numpy as np
import os

def preprocess_image(input_path, output_path, threshold=265):
    # 파일 존재 여부 확인
    if not os.path.isfile(input_path):
        print(f"Error: File does not exist: {input_path}")
        return

    # 이미지 읽기
    image = cv2.imread(input_path)
    enhanced_image = preprocess_table_image(image)
    cv2.imwrite(input_path+'_.png', enhanced_image)
    scale_factor = 2
    
    # 원본 이미지의 크기 가져오기
    height, width = image.shape[:2]
    
    # 새로운 크기 계산
    new_height = int(height * scale_factor)
    new_width = int(width * scale_factor)
    
    # 이미지 리사이즈
    resized_img = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)
    hist = cv2.equalizeHist(gray)  # 히스토그램 평활화
    binary = cv2.adaptiveThreshold(hist, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    edge = cv2.Canny(hist, 100, 200)
    blurred = cv2.GaussianBlur(hist, (7, 7), 0)
# 결과 저장
    cv2.imwrite(output_path, blurred)
def preprocess_table_image(image):
    # 그레이스케일 변환
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 이진화
    adaptive_thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 5)
    
    return adaptive_thresh

def create_imperfect_table(width, height, rows, cols):
    image = np.ones((height, width), dtype=np.uint8) * 255
    
    # 기본 선 그리기
    for i in range(rows + 1):
        y = int(i * height / rows)
        cv2.line(image, (0, y), (width, y), 0, 1)
    
    for i in range(cols + 1):
        x = int(i * width / cols)
        cv2.line(image, (x, 0), (x, height), 0, 1)
    
    # 선 두께 변화
    for i in range(rows + 1):
        y = int(i * height / rows)
        thickness = np.random.randint(1, 4)
        cv2.line(image, (0, y), (width, y), 0, thickness)
    
    # 끊어진 선
    for i in range(cols + 1):
        x = int(i * width / cols)
        for j in range(0, height, 20):
            if np.random.rand() < 0.2:  # 20% 확률로 선을 끊음
                cv2.line(image, (x, j), (x, min(j+10, height)), 255, 1)
    
    # 돌출부 추가
    for _ in range(200):  # 50개의 돌출부 추가
        x = np.random.randint(0, width)
        y = np.random.randint(0, height)
        square_width = np.random.randint(2, 10)  # 돌출부 크기
        square_height = np.random.randint(2, 10)  # 돌출부 크기
        if image[y, x] == 0:  # 선 위에 있는 경우에만 돌출부 추가
            cv2.rectangle(image, (x, y), (x+square_width, y+square_height), 0, -1)
    
    # 노이즈 추가
    noise = np.random.randint(0, 255, (height, width), dtype=np.uint8)
    image = cv2.addWeighted(image, 0.9, noise, 0.1, 0)
    
    return image

# 사용 예
imperfect_table = create_imperfect_table(500, 400, 5, 4)
cv2.imwrite('imperfect_table_with_protrusions.png', imperfect_table)

def batch_preprocess(input_dir, output_dir, threshold=128):
    # 입력 디렉토리가 존재하는지 확인
    if not os.path.isdir(input_dir):
        print(f"Error: Input directory does not exist: {input_dir}")
        return

    # 출력 디렉토리가 없으면 생성
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 입력 디렉토리의 모든 이미지 처리
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)
            preprocess_image(input_path, output_path, threshold)
            print(f"Processed: {filename}")

# 사용 예
input_directory = r"D:\Projects\OCR-LEARNIGN-PROJECT\OCR-PROJECT_OLD\OCR\college"  # 원시 문자열 사용
output_directory = r"D:\Projects\OCR-LEARNIGN-PROJECT\OCR-PROJECT_OLD\OCR_preprocessed\college"  # 원시 문자열 사용
batch_preprocess(input_directory, output_directory)
