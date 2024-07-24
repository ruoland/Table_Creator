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

def boxes_overlap(box1, box2, vertical_threshold):
    # 세로 방향 겹침 확인
    vertical_overlap = (box1['coordinates']['y_max'] >= box2['coordinates']['y_min'] - vertical_threshold and
                        box2['coordinates']['y_max'] >= box1['coordinates']['y_min'] - vertical_threshold)
    
    # 가로 방향 겹침 확인 (조금이라도 겹치면 True)
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
                # Merge boxes
                merged_item['coordinates']['x_min'] = min(merged_item['coordinates']['x_min'], item['coordinates']['x_min'])
                merged_item['coordinates']['x_max'] = max(merged_item['coordinates']['x_max'], item['coordinates']['x_max'])
                merged_item['coordinates']['y_min'] = min(merged_item['coordinates']['y_min'], item['coordinates']['y_min'])
                merged_item['coordinates']['y_max'] = max(merged_item['coordinates']['y_max'], item['coordinates']['y_max'])
                
                # 텍스트 병합 (가로 위치에 따라 순서 결정)
                if item['coordinates']['x_min'] < merged_item['coordinates']['x_min']:
                    merged_item['text'] = f"{item['text']} {merged_item['text']}"
                else:
                    merged_item['text'] += f" {item['text']}"
                
                merged_item['confidence'] = min(merged_item['confidence'], item['confidence'])
            else:
                merged.append(item)
    
    return merged

def draw_boxes(image_path, ocr_data, output_path):
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)

    for item in ocr_data:
        coords = item['coordinates']
        draw.rectangle(
            [
                (coords['x_min'], coords['y_min']),
                (coords['x_max'], coords['y_max'])
            ],
            outline="red",
            width=2
        )

    image.save(output_path)
def identify_rows_and_columns(merged_data, row_threshold=10):
    # Sort by y_min to group into rows
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
    
    # Assign row and column numbers
    for row_idx, row in enumerate(rows):
        for col_idx, item in enumerate(row):
            item['row'] = row_idx + 1
            item['column'] = col_idx + 1
    
    return [item for row in rows for item in row]
def print_low_confidence_words(ocr_data, threshold=0.9):
    low_confidence_words = [item for item in ocr_data if item['confidence'] < threshold]
    low_confidence_words.sort(key=lambda x: x['confidence'])  # 인식률 오름차순 정렬
    
    print(f"\n인식률이 {threshold:.2f} 미만인 단어들:")
    for item in low_confidence_words:
        print(f"텍스트: {item['text']}, 인식률: {item['confidence']:.2f}, "
              f"위치: [{int(item['coordinates']['x_min'])},{int(item['coordinates']['y_min'])}]")

if __name__ == "__main__":
    input_image = "OCR/OCR4.png"
    output_image = "results/OCR4_with_merged_boxes.png"

    # 사용자 입력 받기
    vertical_threshold = int(input("세로 방향 병합 임계값(픽셀)을 입력하세요: "))

    # OCR 실행
    ocr_data = paddle_ocr(input_image)
    
    # OCR 결과 파싱
    parsed_data = parse_ocr_result(ocr_data)
    
   # 박스 병합
    merged_data = merge_boxes(parsed_data, vertical_threshold)
    
    # 행과 열 식별
    merged_data_with_positions = identify_rows_and_columns(merged_data)
    
    # JSON 파일로 저장
    save_to_json(merged_data_with_positions, 'results/ocr_results_merged_with_positions.json')
    
    print("행과 열 정보가 포함된 병합된 OCR 결과가 JSON 파일로 변환되었습니다.")
    # 병합된 박스 그리기
    draw_boxes(input_image, merged_data, output_image)
    
    print(f"병합된 박스가 그려진 이미지가 {output_image}로 저장되었습니다.")

    # 인식률이 낮은 단어들만 출력
    print_low_confidence_words(merged_data)

# 결과 확인 (선택사항)
with open('results/ocr_results_merged.json', 'r', encoding='utf-8') as f:
    loaded_data = json.load(f)
    print(json.dumps(loaded_data[:2], ensure_ascii=False, indent=2))