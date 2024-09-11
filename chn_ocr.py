from PIL import Image, ImageDraw
from paddleocr import PaddleOCR
import json
from timetable import make_timetable

from sympy import im
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

from config import *

import os

import re
from typing import List, Dict

def preprocess_image(image_path, scale_factor=4):
    # 이미지 로드
    image = Image.open(image_path)
    
    # 이미지 확대
    width, height = image.size
    image = image.resize((width * scale_factor, height * scale_factor), Image.LANCZOS)
    
    # 대비 향상
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(1.5)
    
    # 노이즈 제거 및 선명화
    image = image.filter(ImageFilter.MedianFilter(size=3))
    image = image.filter(ImageFilter.UnsharpMask(radius=2, percent=150))
    
    return image

#리사이즈 전처리 버전
def paddle_ocr_preprocess(img_path):
    preprocessed_image = preprocess_image(img_path)
    ocr = PaddleOCR(lang="korean", use_angle_cls=True, use_gpu=False)
    result = ocr.ocr(np.array(preprocessed_image), cls=False)
    return result[0]

#일반 버전
def paddle_ocr(img_path):
    # OCR 수행
    ocr = PaddleOCR(lang="korean")
    result = ocr.ocr(img_path)
    return result[0]

def postprocess_ocr_result(ocr_result, confidence_threshold=0.7):
    processed_result = []
    for item in ocr_result:
        if item[1][1] >= confidence_threshold:
            processed_result.append(item)
        else:
            # 낮은 신뢰도 결과에 대한 처리 (예: 로깅 또는 사용자 확인 요청)
            print(f"{item[1][0]} 의 신뢰도가 낮습니다: {item[1][1]})")
    
    return processed_result
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
    horizontal_overlap = (box1['coordinates']['x_min'] - vertical_threshold <= box2['coordinates']['x_max'] and
                          box2['coordinates']['x_min'] - vertical_threshold <= box1['coordinates']['x_max'])

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
    
    # Find the maximum number of columns
    max_columns = max(len(row) for row in rows)
    
    # Assign row and column numbers
    for row_idx, row in enumerate(rows):
        for col_idx, item in enumerate(row):
            item['row'] = row_idx + 1
            item['column'] = col_idx + 1
    
    return [item for row in rows for item in row], len(rows), max_columns

def print_low_confidence_words(ocr_data, threshold=0.9):
    low_confidence_words = [item for item in ocr_data if item['confidence'] < threshold]
    low_confidence_words.sort(key=lambda x: x['confidence'])  # 인식률 오름차순 정렬
    
    print(f"\n인식률이 {threshold:.2f} 미만인 단어들:")
    for item in low_confidence_words:
        print(f"텍스트: {item['text']}, 인식률: {item['confidence']:.2f}, "
              f"위치: [{int(item['coordinates']['x_min'])},{int(item['coordinates']['y_min'])}]")


def identify_time_slots(ocr_data: List[Dict]) -> List[Dict]:
    # 시간 패턴 정의
    time_pattern = r'\d{1,2}:\d{2}\s*-\s*\d{1,2}:\d{2}|\d{1,2}시\s*\d{2}분'
    
    # x 좌표를 기준으로 정렬
    sorted_data = sorted(ocr_data, key=lambda x: x['coordinates']['x_min'])
    
    time_slots = []
    for item in sorted_data:
        text = item['text']
        if re.search(time_pattern, text) or '교시' in text:
            time_slots.append(item)
    
    # y 좌표로 정렬하여 순서 유지
    time_slots.sort(key=lambda x: x['coordinates']['y_min'])
    
    return time_slots

def extract_schedule(ocr_data: List[Dict], time_slots: List[Dict]) -> List[Dict]:
    schedule = []
    for i, time_slot in enumerate(time_slots):
        y_min = time_slot['coordinates']['y_min']
        y_max = time_slots[i+1]['coordinates']['y_min'] if i+1 < len(time_slots) else float('inf')
        
        row_items = [item for item in ocr_data if y_min <= item['coordinates']['y_min'] < y_max]
        row_items.sort(key=lambda x: x['coordinates']['x_min'])
        
        schedule.append({
            'time': time_slot['text'],
            'items': row_items
        })
    
    return schedule

# 메인 함수
def process_timetable(ocr_data: List[Dict]) -> List[Dict]:
    time_slots = identify_time_slots(ocr_data)
    schedule = extract_schedule(ocr_data, time_slots)
    return schedule

if __name__ == "__main__":
    input_path = IMAGES_YOUNG[4]
    file_name = os.path.basename(input_path)

    output_image = RESULT_COLLEGE + file_name
    input_image = input_path

    vertical_threshold = 15
    
    # OCR 실행 및 후처리
    ocr_data = paddle_ocr_preprocess(input_image)
    processed_ocr_data = postprocess_ocr_result(ocr_data)
    
    # OCR 결과 파싱
    parsed_data = parse_ocr_result(ocr_data)
    
    # 박스 병합
    merged_data = merge_boxes(parsed_data, vertical_threshold)
    
    # 행과 열 식별
    merged_data_with_positions = identify_rows_and_columns(merged_data)
    
    results_file = (f'results/ocr_results-{file_name}.json')
    
    # JSON 파일로 저장
    save_to_json(merged_data_with_positions, results_file)
    
    print("행과 열 정보가 포함된 병합된 OCR 결과가 JSON 파일로 변환되었습니다.")
    # 병합된 박스 그리기
    draw_boxes(input_image, merged_data, output_image)
    
    print(f"병합된 박스가 그려진 이미지가 {output_image}로 저장되었습니다.")
    file = open(results_file, 'r', encoding='UTF-8')
    make_timetable(file.read())
    file.close()
    
    
    # OCR 결과를 이용하여 시간표 처리
    processed_schedule = process_timetable(merged_data)

    # 결과 출력
    for row in processed_schedule:
        print(f"시간: {row['time']}")
        for item in row['items']:
            print(f"  - {item['text']}")
        print()
    # 인식률이 낮은 단어들만 출력
    print_low_confidence_words(merged_data)