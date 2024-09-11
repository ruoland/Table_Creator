import cv2
import easyocr
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from difflib import SequenceMatcher

def string_similarity(a, b):
    return SequenceMatcher(None, a, b).ratio()

def post_process_text(text, known_words, similarity_threshold=0.8):
    for known_word in known_words:
        if string_similarity(text, known_word) > similarity_threshold:
            return known_word
    return text

# 알려진 단어 목록
known_words = ['창체', '국어', '수학', '영어', '과학', '사회', '체육', '음악', '미술', '기술가정', '월', '화', '수', '목', '금']

def create_grid(image_shape, rows, cols):
    height, width = image_shape[:2]
    row_height = height / rows
    col_width = width / cols
    return row_height, col_width

def get_center_point(bbox):
    top_left, _, bottom_right, _ = bbox
    center_x = (top_left[0] + bottom_right[0]) / 2
    center_y = (top_left[1] + bottom_right[1]) / 2
    return center_x, center_y

def get_grid_position(center_point, row_height, col_width):
    x, y = center_point
    row = int(y / row_height)
    col = int(x / col_width)
    return row, col

def detect_and_recognize_text(image_path, font_path, rows=7, cols=6):
    # 이미지 로드
    image = cv2.imread(image_path)
    original_image = image.copy()
    
    # 그리드 생성
    row_height, col_width = create_grid(image.shape, rows, cols)
    
    reader = easyocr.Reader(['ko', 'en'], gpu=False)
    results = reader.readtext(image, paragraph=False, detail=1, contrast_ths=0.1, adjust_contrast=0.8)
    
    recognized_texts = []
    
    # OpenCV 이미지를 PIL 이미지로 변환
    pil_image = Image.fromarray(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_image)
    
    # 폰트 설정
    grid = [[[] for _ in range(cols)] for _ in range(rows)]
    for (bbox, text, prob) in results:
        # 박스 좌표 추출
        (top_left, top_right, bottom_right, bottom_left) = bbox
        top_left = tuple(map(int, top_left))
        bottom_right = tuple(map(int, bottom_right))
        
        # 후처리 적용
        processed_text = post_process_text(text, known_words)
        
        # 중심점 계산 및 그리드 위치 확인
        center_point = get_center_point(bbox)
        row, col = get_grid_position(center_point, row_height, col_width)
        
        recognized_texts.append((top_left, bottom_right, processed_text, prob, row, col))
        
        # 사각형 그리기
        draw.rectangle([top_left, bottom_right], outline="green", width=2)
        
        # 인식된 텍스트, 인식률, 행/열 정보 표시
        display_text = f"{processed_text} ({prob:.2f}) [{row},{col}]"
        grid[row][col].append((processed_text, prob, top_left[0], top_left[1]))

    # PIL 이미지를 OpenCV 이미지로 변환
    result_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    
    return result_image, recognized_texts, grid
import cv2
import easyocr
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from difflib import SequenceMatcher


def print_low_confidence_words(grid):
    print("\n인식률이 1.00 미만인 단어들:")
    low_confidence_words = []
    for row_number, row in enumerate(grid):
        for col_number, cell in enumerate(row):
            for text, prob, _, _ in cell:
                if prob < 0.90:
                    low_confidence_words.append((text, prob, row_number, col_number))
    
    # 인식률을 기준으로 정렬 (낮은 순서대로)
    low_confidence_words.sort(key=lambda x: x[1])
    
    for text, prob, row, col in low_confidence_words:
        print(f"텍스트: {text}, 인식률: {prob:.2f}, 위치: [{row},{col}]")

def print_formatted_results(grid):
    for row_number, row in enumerate(grid):
        print(f"줄 {row_number + 1}:")
        line_text = []
        for col_number, cell in enumerate(row):
            if cell:
                for text, prob, x, y in sorted(cell, key=lambda item: item[2]):  # x 좌표로 정렬
                    print(f"  텍스트: {text}, 인식률: {prob:.2f}, 위치: [{row_number},{col_number}]")
                    line_text.append(text)
        if line_text:
            print(f"  전체 텍스트: {' '.join(line_text)}\n")
        else:
            print("  <빈 줄>\n")

# 이미지와 폰트 경로
image_path = 'C:\\Users\\admin\\OneDrive\\OCR-PROJECT\\OCR\\young\\ocr4.png'
font_path = 'C:\\Users\\admin\\OneDrive\\OCR-PROJECT\\NanumGothic.ttf'

# 텍스트 영역 표시 및 인식
result_image, recognized_texts, grid = detect_and_recognize_text(image_path, font_path)

# 결과 이미지 저장
cv2.imwrite('text_recognized_with_prob_and_position.jpg', result_image)

# 포맷에 맞춰 결과 출력
print("\nOCR 결과 (줄별로 그룹화):")
print_formatted_results(grid)

# 빈 셀 출력
print("빈 셀 위치:")
empty_cells = [(row, col) for row in range(len(grid)) for col in range(len(grid[0])) if not grid[row][col]]
for row, col in empty_cells:
    print(f"[{row},{col}]")
    
print_low_confidence_words(grid)

# 결과 이미지 표시 (선택적)
cv2.imshow('Recognized Text', result_image)
cv2.waitKey(0)
cv2.destroyAllWindows()