from paddleocr import PaddleOCR
import json
import numpy as np

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

def paddle_ocr(img_path):
    ocr = PaddleOCR(lang="korean")
    result = ocr.ocr(img_path, cls=False)
    return result[0]

def parse_paddle_result(ocr_result):
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
        json.dump(data, f, ensure_ascii=False, indent=2, cls=NumpyEncoder)

def main():
    image_path = "OCR7.png"  # 이미지 경로를 적절히 수정해주세요
    
    # PaddleOCR 실행
    paddle_data = paddle_ocr(image_path)
    paddle_results = parse_paddle_result(paddle_data)
    save_to_json(paddle_results, 'paddle_ocr_results.json')
    
    print("PaddleOCR 결과가 paddle_ocr_results.json 파일에 저장되었습니다.")

if __name__ == "__main__":
    main()