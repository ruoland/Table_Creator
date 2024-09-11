import json
import numpy as np
from PIL import Image, ImageDraw
from collections import defaultdict

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)
import json
from PIL import Image, ImageDraw

def numpy_to_python(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj

def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_json(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=numpy_to_python)
def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_json(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def boxes_overlap(box1, box2, threshold=0.5):
    # Calculate the intersection area
    x_left = max(box1['coordinates']['x_min'], box2['coordinates']['x_min'])
    y_top = max(box1['coordinates']['y_min'], box2['coordinates']['y_min'])
    x_right = min(box1['coordinates']['x_max'], box2['coordinates']['x_max'])
    y_bottom = min(box1['coordinates']['y_max'], box2['coordinates']['y_max'])

    if x_right < x_left or y_bottom < y_top:
        return False

    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # Calculate the area of both boxes
    box1_area = (box1['coordinates']['x_max'] - box1['coordinates']['x_min']) * (box1['coordinates']['y_max'] - box1['coordinates']['y_min'])
    box2_area = (box2['coordinates']['x_max'] - box2['coordinates']['x_min']) * (box2['coordinates']['y_max'] - box2['coordinates']['y_min'])

    # Calculate the overlap ratio
    overlap_ratio = intersection_area / min(box1_area, box2_area)

    return overlap_ratio > threshold

def create_timetable_matrix(ocr_results, x_threshold, y_threshold):
    # OCR 결과를 정렬
    sorted_results = sorted(ocr_results, key=lambda x: (x['coordinates']['y_min'], x['coordinates']['x_min']))
    
    # 행과 열 식별
    rows = []
    current_row = []
    last_y = None
    for result in sorted_results:
        if last_y is None or result['coordinates']['y_min'] - last_y > y_threshold:
            if current_row:
                rows.append(sorted(current_row, key=lambda x: x['coordinates']['x_min']))
            current_row = [result]
            last_y = result['coordinates']['y_min']
        else:
            current_row.append(result)
    if current_row:
        rows.append(sorted(current_row, key=lambda x: x['coordinates']['x_min']))
    
    # 행렬 생성
    matrix = []
    for row in rows:
        matrix_row = []
        for item in row:
            matrix_row.append(item['text'])
        matrix.append(matrix_row)
    
    return matrix
def merge_boxes(boxes, confidence_threshold=0.8):
    merged = []
    for box in boxes:
        if box['confidence'] < confidence_threshold:
            continue  # 낮은 신뢰도의 결과 제외
        
        overlap = False
        for merged_box in merged:
            if boxes_overlap(box, merged_box):
                # 겹치는 경우, 신뢰도가 높은 결과 선택
                if box['confidence'] > merged_box['confidence']:
                    merged_box.update(box)
                overlap = True
                break
        
        if not overlap:
            merged.append(box)
    
    return merged

def validate_results(merged_boxes, original_boxes):
    validated = []
    for box in merged_boxes:
        if any(original_box['text'] == box['text'] for original_box in original_boxes):
            validated.append(box)
        else:
            print(f"Warning: Unexpected text '{box['text']}' found in merged results.")
    return validated

def draw_boxes(image_path, results, output_path, color="red"):
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)
    
    for result in results:
        coords = result['coordinates']
        x_min, y_min = coords['x_min'], coords['y_min']
        x_max, y_max = coords['x_max'], coords['y_max']
        
        draw.rectangle([x_min, y_min, x_max, y_max], outline=color, width=2)
    
    image.save(output_path)
    print(f"Image with boxes saved to {output_path}")
def group_by_rows(boxes, row_threshold=10):
    if not boxes:  # 빈 리스트인 경우 처리
        print("Warning: No text boxes detected in the image.")
        return []

    sorted_boxes = sorted(boxes, key=lambda x: x['coordinates']['y_min'])
    rows = defaultdict(list)
    current_row = 0
    last_y = sorted_boxes[0]['coordinates']['y_min']

    for box in sorted_boxes:
        if abs(box['coordinates']['y_min'] - last_y) > row_threshold:
            current_row += 1
        rows[current_row].append(box)
        last_y = box['coordinates']['y_min']

    return [sorted(row, key=lambda x: x['coordinates']['x_min']) for row in rows.values()]

def create_coordinate_grid(boxes, x_threshold=20, y_threshold=20):
    if not boxes:
        print("Warning: No text boxes detected in the image.")
        return []

    # Sort boxes by y_min and then by x_min
    sorted_boxes = sorted(boxes, key=lambda x: (x['coordinates']['y_min'], x['coordinates']['x_min']))

    grid = []
    current_row = []
    last_y = sorted_boxes[0]['coordinates']['y_min']

    for box in sorted_boxes:
        if abs(box['coordinates']['y_min'] - last_y) > y_threshold:
            if current_row:
                grid.append(current_row)
                current_row = []
            last_y = box['coordinates']['y_min']
        
        if not current_row or abs(box['coordinates']['x_min'] - current_row[-1]['coordinates']['x_min']) > x_threshold:
            current_row.append(box)

    if current_row:
        grid.append(current_row)

    return grid

def create_timetable_from_grid(grid):
    timetable = []
    for row in grid:
        timetable_row = [box['text'] for box in row]
        timetable.append(timetable_row)
    return timetable

def process_ocr_to_timetable(ocr_results, x_threshold=20, y_threshold=20):
    if not ocr_results:
        print("Warning: OCR results are empty.")
        return []

    merged_boxes = merge_boxes(ocr_results)
    grid = create_coordinate_grid(merged_boxes, x_threshold, y_threshold)
    timetable = create_timetable_from_grid(grid)
    matrix = create_timetable_matrix(ocr_results, x_threshold, y_threshold)
    return timetable, matrix

def create_timetable(rows):
    if not rows:  # 빈 행 리스트인 경우 처리
        print("Warning: No rows detected in the image.")
        return []

    timetable = []
    for row in rows:
        timetable_row = [box['text'] for box in row]
        timetable.append(timetable_row)
    return timetable
def save_timetable_as_html(timetable, matrix, output_path):
    html_content = "<html><body>"
    html_content += "<h1>Timetable</h1>"
    
    # 기존 시간표 형식 출력 (수정된 부분)
    html_content += "<h2>Parsed Format</h2>"
    html_content += "<ul>"
    for entry in timetable:
        html_content += f"<li>{entry}</li>"
    html_content += "</ul>"
    
    # 행렬 형식 출력
    html_content += "<h2>Matrix Format</h2>"
    html_content += "<table border='1'>"
    for row in matrix:
        html_content += "<tr>"
        for cell in row:
            html_content += f"<td>{cell}</td>"
        html_content += "</tr>"
    html_content += "</table>"
    
    html_content += "</body></html>"
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)
