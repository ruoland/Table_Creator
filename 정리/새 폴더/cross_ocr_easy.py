import easyocr
from utils import save_json, draw_boxes
from difflib import get_close_matches
def easy_ocr(image_path):
    reader = easyocr.Reader(['ko'])  # Korean language
    result = reader.readtext(image_path)
    
    parsed_result = []
    for item in result:
        coordinates = item[0]
        text = item[1]
        confidence = float(item[2])  # Convert to Python float
        
        parsed_result.append({
            "text": text,
            "confidence": confidence,
            "coordinates": {
                "x_min": int(min(coord[0] for coord in coordinates)),
                "x_max": int(max(coord[0] for coord in coordinates)),
                "y_min": int(min(coord[1] for coord in coordinates)),
                "y_max": int(max(coord[1] for coord in coordinates))
            }
        })
    
    return parsed_result
from PIL import Image, ImageEnhance, ImageFilter
# 사전 정의된 단어 목록
defined_words = ["국어", "수학", "영어", "사회", "과학", "한문", "일어", "체육", "음악", "미술", "기술", "가정", "도덕", "창제"]

def correct_text(text, word_list):
    # 가장 유사한 단어를 찾음
    matches = get_close_matches(text, word_list, n=1, cutoff=0.6)
    return matches[0] if matches else text
# 결과 출력 및 텍스트 교정

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage: python cross_ocr_easy.py <image_path> <output_json_path>")
        sys.exit(1)
    
        # 이미지 파일 경로
    image_path = sys.argv[1]
    output_path = sys.argv[2]
    # 이미지 열기
    img = Image.open(image_path)

    # 이미지 전처리: 그레이스케일 변환, 샤프닝 필터 적용
    img = img.convert('L')  # 그레이스케일 변환
    img = img.filter(ImageFilter.SHARPEN)  # 샤프닝 필터 적용

    # 임시 파일로 저장
    temp_path = './temp/temp_image.png'
    img.save(temp_path)

    
    result = easy_ocr(temp_path)
    save_json(result, output_path)
    corrected_results = []
    for (bbox, text, prob) in result:
        corrected_text = correct_text(text, defined_words)
        corrected_results.append((bbox, corrected_text, prob))
        print(f"Change Text: {text}, Corrected Text: {corrected_text}, Probability: {prob}")

    # Draw boxes for EasyOCR results
    easy_boxes_image = "cross_image_easy_ocr_boxes.png"
    draw_boxes(image_path, result, easy_boxes_image, color="green")
    
    print(f"EasyOCR results saved to {output_path}")
    print(f"EasyOCR boxes drawn on {easy_boxes_image}")
    