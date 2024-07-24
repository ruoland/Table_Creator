import easyocr
import json
from difflib import SequenceMatcher
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

def string_similarity(a, b):
    return SequenceMatcher(None, a, b).ratio()

def post_process_text(text, known_words, similarity_threshold=0.8):
    for known_word in known_words:
        if string_similarity(text, known_word) > similarity_threshold:
            return known_word
    return text

def easy_ocr(img_path, known_words):
    reader = easyocr.Reader(['ko', 'en'], gpu=False)
    results = reader.readtext(img_path, paragraph=False, detail=1, contrast_ths=0.1, adjust_contrast=0.8)
    
    parsed_results = []
    for (bbox, text, prob) in results:
        (top_left, _, bottom_right, _) = bbox
        processed_text = post_process_text(text, known_words)
        parsed_results.append({
            "text": processed_text,
            "confidence": prob,
            "coordinates": {
                "x_min": top_left[0],
                "x_max": bottom_right[0],
                "y_min": top_left[1],
                "y_max": bottom_right[1]
            }
        })
    
    return parsed_results

def save_to_json(data, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2, cls=NumpyEncoder)

def main():
    image_path = "OCR7.png"  # 이미지 경로를 적절히 수정해주세요
    known_words = ['창체', '국어', '수학', '영어', '과학', '사회', '체육', '음악', '미술', '기술가정', '월', '화', '수', '목', '금']
    
    # EasyOCR 실행
    easy_results = easy_ocr(image_path, known_words)
    save_to_json(easy_results, 'easy_ocr_results.json')
    
    print("EasyOCR 결과가 easy_ocr_results.json 파일에 저장되었습니다.")

if __name__ == "__main__":
    main()