from paddleocr import PaddleOCR
import json
from PIL import Image, ImageEnhance, ImageFilter
from utils import save_json, draw_boxes

def paddle_ocr(image_path, vertical_threshold=10):
    ocr = PaddleOCR(lang="korean")
    # 이미지 파일 경로
    # 이미지 열기
    img = Image.open(image_path)

    # 이미지 전처리: 그레이스케일 변환, 샤프닝 필터 적용
    img = img.convert('L')  # 그레이스케일 변환
    img = img.filter(ImageFilter.SHARPEN)  # 샤프닝 필터 적용

    # 임시 파일로 저장
    temp_path = 'temp/heh.png'
    img.save(temp_path)

    result = ocr.ocr(temp_path, cls=False)
    if result == None:
        print('인식한 문자가 없습니다.')
        return None
    print(result)
    parsed_result = parse_ocr_result(result[0])
    merged_result = merge_boxes(parsed_result, vertical_threshold)
    final_result, _, _ = identify_rows_and_columns(merged_result)
    
    return final_result

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

def boxes_overlap(box1, box2, vertical_threshold):
    vertical_overlap = (box1['coordinates']['y_max'] >= box2['coordinates']['y_min'] - vertical_threshold and
                        box2['coordinates']['y_max'] >= box1['coordinates']['y_min'] - vertical_threshold)
    
    horizontal_overlap = (box1['coordinates']['x_min'] <= box2['coordinates']['x_max'] and
                          box2['coordinates']['x_min'] <= box1['coordinates']['x_max'])

    return vertical_overlap and horizontal_overlap

def merge_boxes(ocr_data, vertical_threshold):
    merged = []
    sorted_data = sorted(ocr_data, key=lambda x: (x['coordinates']['y_min'], x['coordinates']['x_min']))
    
    for item in sorted_data:
        if not merged:
            merged.append(item)
        else:
            merged_item = None
            for m in merged:
                if boxes_overlap(m, item, vertical_threshold):
                    merged_item = m
                    break
            
            if merged_item:
                merged_item['coordinates']['x_min'] = min(merged_item['coordinates']['x_min'], item['coordinates']['x_min'])
                merged_item['coordinates']['x_max'] = max(merged_item['coordinates']['x_max'], item['coordinates']['x_max'])
                merged_item['coordinates']['y_min'] = min(merged_item['coordinates']['y_min'], item['coordinates']['y_min'])
                merged_item['coordinates']['y_max'] = max(merged_item['coordinates']['y_max'], item['coordinates']['y_max'])
                
                if item['coordinates']['x_min'] < merged_item['coordinates']['x_min']:
                    merged_item['text'] = f"{item['text']} {merged_item['text']}"
                else:
                    merged_item['text'] += f" {item['text']}"
                
                merged_item['confidence'] = min(merged_item['confidence'], item['confidence'])
            else:
                merged.append(item)
    
    return merged

def identify_rows_and_columns(merged_data, row_threshold=10):
    sorted_data = sorted(merged_data, key=lambda x: x['coordinates']['y_min'])
    
    rows = []
    current_row = []
    for item in sorted_data:
        if not current_row or abs(item['coordinates']['y_min'] - current_row[-1]['coordinates']['y_min']) <= row_threshold:
            current_row.append(item)
        else:
            rows.append(sorted(current_row, key=lambda x: x['coordinates']['x_min']))
            current_row = [item]
    
    if current_row:
        rows.append(sorted(current_row, key=lambda x: x['coordinates']['x_min']))
    
    max_columns = max(len(row) for row in rows)
    
    for row_idx, row in enumerate(rows):
        for col_idx, item in enumerate(row):
            item['row'] = row_idx + 1
            item['column'] = col_idx + 1
    
    return [item for row in rows for item in row], len(rows), max_columns

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage: python cross_ocr_paddle.py <image_path> <output_json_path>")
        sys.exit(1)
    
    image_path = sys.argv[1]
    output_path = sys.argv[2]
    
    result = paddle_ocr(image_path)
    save_json(result, output_path)
    
    # Draw boxes for PaddleOCR results
    paddle_boxes_image = "cross_image_paddle_ocr_boxes.png"
    draw_boxes(image_path, result, paddle_boxes_image, color="blue")
    
    print(f"PaddleOCR 결과가 {output_path}에 저장되었습니다.")
    print(f"PaddleOCR 박스가 {paddle_boxes_image}에 그려졌습니다")