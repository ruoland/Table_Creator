from PIL import Image, ImageDraw
from paddleocr import PaddleOCR
import json

def paddle_ocr(img_path):
    ocr = PaddleOCR(lang="korean")
    result = ocr.ocr(img_path, cls=False)
    return result[0]

def parse_ocr_result(ocr_result):
    results = []
    for item in ocr_result:
        coordinates = item[0]
        text = item[1][0]
        confidence = item[1][1]
        
        x_coords = [coord[0] for coord in coordinates]
        y_coords = [coord[1] for coord in coordinates]
        
        results.append({
            "text": text,
            "confidence": confidence,
            "coordinates": {
                "x_min": min(x_coords),
                "x_max": max(x_coords),
                "y_min": min(y_coords),
                "y_max": max(y_coords)
            }
        })
    
    return results

def save_to_json(data, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    folder = "OCR/"
    type = "College/"
    results = "results/"
    input_image = folder+ type+"ocr2.jpg"
    output_image = results +type+"orc2.jpg"

    # 사용자 입력 받기
    vertical_threshold = int(input("세로 방향 병합 임계값(픽셀)을 입력하세요: "))

    # OCR 실행
    ocr_data = paddle_ocr(input_image)
    
    # OCR 결과 파싱
    parsed_data = parse_ocr_result(ocr_data)
   
    print(ocr_data)
