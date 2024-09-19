import os
import random
from tqdm import tqdm
import numpy as np
import logging
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from concurrent.futures import ProcessPoolExecutor
import itertools
from opti_calc import *
from dataset_draw import *
from dataset_constant import *
from dataset_utils import *
from dataset_io import *


# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
def generate_random_resolution():
    """랜덤한 이미지 해상도 생성 (여백 포함)"""
    width = random.randint(MIN_IMAGE_WIDTH, MAX_IMAGE_WIDTH)
    height = random.randint(MIN_IMAGE_HEIGHT, MAX_IMAGE_HEIGHT)
    margin_left = random.randint(MIN_MARGIN, MAX_MARGIN)
    margin_right = random.randint(MIN_MARGIN, MAX_MARGIN)
    margin_top = random.randint(MIN_MARGIN, MAX_MARGIN)
    margin_bottom = random.randint(MIN_MARGIN, MAX_MARGIN)
    width = width - (width % 32) + margin_left + margin_right
    height = height - (height % 32) + margin_top + margin_bottom
    return (width, height), (margin_left, margin_top, margin_right, margin_bottom)
def create_table(image_width, image_height, margins, title_height, has_gap=False):
    #print(f"Debug: create_table - image_width = {image_width}, image_height = {image_height}")
     
    margin_left, margin_top, margin_right, margin_bottom = margins
    table_width = image_width - margin_left - margin_right
    table_height = image_height - margin_top - margin_bottom - title_height  # 제목 높이 고려
    
    cols = random.randint(MIN_COLS, MAX_COLS)
    rows = random.randint(MIN_ROWS, MAX_ROWS)
    
    cell_width = min(max(table_width // cols, BASE_CELL_WIDTH), MAX_CELL_WIDTH)
    cell_height = min(max(table_height // rows, BASE_CELL_HEIGHT), MAX_CELL_HEIGHT)
    
    gap = random.randint(1, 3) if has_gap else 0
    
    cells = [
        [margin_left + col * cell_width + (gap if has_gap else 0),
         margin_top + title_height + row * cell_height + (gap if has_gap else 0),  # 제목 높이 추가
         margin_left + (col + 1) * cell_width - (gap if has_gap else 0),
         margin_top + title_height + (row + 1) * cell_height - (gap if has_gap else 0),  # 제목 높이 추가
         row, col, False]
        for row in range(rows)
        for col in range(cols)
    ]
    
    cells = merge_cells(cells, rows, cols)
    
    table_bbox = [margin_left, margin_top + title_height,  # 제목 높이 추가
                  margin_left + cols * cell_width, 
                  margin_top + title_height + rows * cell_height]  # 제목 높이 추가
    
    return cells, table_bbox, gap, rows, cols

def merge_cells(cells, rows, cols):
    """셀 병합 로직"""
    merged_cells = cells.copy()
    num_merges = random.randint(1, min(rows, cols) // 2)
    merged_areas = []

    for _ in range(num_merges):
        attempts = 0
        while attempts < 10:
            # 병합할 셀 영역 선택
            start_row = random.randint(0, rows - 2)
            start_col = random.randint(0, cols - 2)
            merge_rows = random.randint(2, min(3, rows - start_row))
            merge_cols = random.randint(2, min(3, cols - start_col))
            
            merge_area = (start_row, start_col, start_row + merge_rows, start_col + merge_cols)
            
            if not any(is_overlapping(merge_area, area) for area in merged_areas):
                # 병합 실행
                new_cell_info = [
                    cells[start_row * cols + start_col][0],
                    cells[start_row * cols + start_col][1],
                    cells[(start_row + merge_rows - 1) * cols + (start_col + merge_cols - 1)][2],
                    cells[(start_row + merge_rows - 1) * cols + (start_col + merge_cols - 1)][3],
                    start_row, start_col, True,  # True는 병합된 셀을 의미
                    start_row, start_col,  # 병합 시작 위치
                    start_row + merge_rows - 1, start_col + merge_cols - 1  # 병합 끝 위치
                ]
                for r in range(start_row, start_row + merge_rows):
                    for c in range(start_col, start_col + merge_cols):
                        merged_cells[r * cols + c] = None
                
                merged_cells[start_row * cols + start_col] = new_cell_info
                merged_areas.append(merge_area)
                break
            
            attempts += 1

    return [cell for cell in merged_cells if cell is not None]

def is_overlapping(area1, area2):
    """두 영역이 겹치는지 확인"""
    return not (area1[2] <= area2[0] or area1[0] >= area2[2] or
                area1[3] <= area2[1] or area1[1] >= area2[3])
def add_title_to_image(draw, image_width, margin_top, bg_color):
    # bg_color 타입 확인 및 처리
    if isinstance(bg_color, int):
        bg_color = (bg_color, bg_color, bg_color)  # 그레이스케일을 RGB로 변환
    elif not isinstance(bg_color, tuple) or len(bg_color) != 3:
        raise ValueError(f"Invalid bg_color format: {bg_color}")

    title = generate_random_title()
    
    # 폰트 크기를 이미지 너비의 2%에서 5% 사이로 설정
    font_size = random.randint(int(image_width * 0.02), int(image_width * 0.1))
    font_size = max(MIN_TITLE_SIZE, min(font_size, MAX_TITLE_SIZE))  # MIN_TITLE_SIZE와 MAX_TITLE_SIZE 사이로 제한
    
    font = ImageFont.truetype(random.choice(FONTS), font_size)
    
    # 텍스트 크기 계산
    left, top, right, bottom = font.getbbox(title)
    text_width = right - left
    text_height = bottom - top
    
    # 제목이 너무 길면 줄바꿈
    if text_width > image_width * 0.9:  # 이미지 너비의 90%를 넘으면 줄바꿈
        words = title.split()
        half = len(words) // 2
        title = ' '.join(words[:half]) + '\n' + ' '.join(words[half:])
        left, top, right, bottom = font.getbbox(title)
        text_width = right - left
        text_height = bottom - top
    
    # 텍스트 위치 계산 (가운데 정렬)
    x = (image_width - text_width) // 2
    y = margin_top + random.randint(3, 30)  # 상단 여백에 랜덤 추가
    
    text_color = get_line_color(bg_color)    
    #print(f"Debug: bg_color = {bg_color}, text_color = {text_color}")  # 디버깅용 출력
    # 텍스트 그리기
    draw.multiline_text((x, y), title, font=font, fill=text_color, align='center')
    
    return y + text_height + random.randint(5, 40)  # 제목 아래 20-40픽셀의 추가 랜덤 여백


def get_contrasting_color(bg_color):
    # bg_color가 정수인 경우 RGB로 변환
    if isinstance(bg_color, int):
        r = (bg_color >> 16) & 255
        g = (bg_color >> 8) & 255
        b = bg_color & 255
    # bg_color가 문자열인 경우 (예: "#RRGGBB")
    elif isinstance(bg_color, str) and bg_color.startswith("#"):
        r, g, b = int(bg_color[1:3], 16), int(bg_color[3:5], 16), int(bg_color[5:7], 16)
    # bg_color가 튜플이나 리스트인 경우 (예: (R, G, B))
    elif isinstance(bg_color, (tuple, list)) and len(bg_color) == 3:
        r, g, b = bg_color
    else:
        raise ValueError("Invalid bg_color format")

    # 밝기 계산 (YIQ 색상 공간의 Y 값 사용)
    brightness = (r * 299 + g * 587 + b * 114) / 1000
    
    # 밝기에 따라 검은색 또는 흰색 반환
    return (0, 0, 0) if brightness > 128 else (255, 255, 255)
def generate_coco_annotations(cells, table_bbox, rows, cols, image_id, gap):
    """COCO 형식의 주석 생성 (테이블과 셀만)"""
    coco_annotations = []
    annotation_id = 1

    # Cell annotations
    for cell_info in cells:
        is_merged = cell_info[6] if len(cell_info) > 6 else False
        category_id = 2 if is_merged else 1
        category_name = "merged_cell" if is_merged else "normal_cell"
        width = cell_info[2] - cell_info[0]
        height = cell_info[3] - cell_info[1]
        row = cell_info[4]
        col = cell_info[5]
        cell_annotation = {
            "id": annotation_id,
            "image_id": image_id,
            "category_id": category_id,
            "bbox": [cell_info[0], cell_info[1], width, height],
            "area": width * height,
            "iscrowd": 0,
            "category_name": category_name,
            "row": row,
            "col": col,
            "is_merged": is_merged
        }
        
        if is_merged:
            cell_annotation["merge_start"] = {"row": cell_info[7], "col": cell_info[8]}
            cell_annotation["merge_end"] = {"row": cell_info[9], "col": cell_info[10]}
        coco_annotations.append(cell_annotation)
        annotation_id += 1

    # Table annotation
    table_width = table_bbox[2] - table_bbox[0]
    table_height = table_bbox[3] - table_bbox[1]
    table_annotation = {
        "id": annotation_id,
        "image_id": image_id,
        "category_id": 3,  # 테이블 카테고리 ID
        "bbox": [table_bbox[0], table_bbox[1], table_width, table_height],
        "area": table_width * table_height,
        "iscrowd": 0,
        "category_name": "table",
        "rows": rows,
        "cols": cols,
        "cell_count": len(cells)
    }
    coco_annotations.append(table_annotation)

    return coco_annotations

def generate_image_and_labels(image_id, resolution, margins, bg_mode, has_gap, is_imperfect=False):
    try:
        image_width, image_height = resolution
        
        bg_color = BACKGROUND_COLORS['light']['white'] if bg_mode == 'light' else BACKGROUND_COLORS['dark']['black']
        #print(f"Debug: generate_image_and_labels - bg_color = {bg_color}, type = {type(bg_color)}")
        
        img = Image.new('RGB', (image_width, image_height), color=bg_color)
        draw = ImageDraw.Draw(img)
        
        title_height = add_title_to_image(draw, image_width, margins[1], bg_color)
        #print(f"Debug: After add_title_to_image - title_height = {title_height}")
        
        cells, table_bbox, gap, rows, cols = create_table(image_width, image_height, margins, title_height, has_gap)
        #print(f"Debug: After create_table - rows = {rows}, cols = {cols}")
        
        line_widths, has_outer_lines = draw_table(draw, cells, table_bbox, bg_color, has_gap, is_imperfect)
        #print(f"Debug: After draw_table - has_outer_lines = {has_outer_lines}")
        
        add_content_to_cells(draw, cells, random.choice(FONTS), bg_color)
        #print(f"Debug: After add_content_to_cells")
        
        
        if is_imperfect:
            img = apply_imperfections(img, cells)
        
        coco_annotations = generate_coco_annotations(cells, table_bbox, rows, cols, image_id, gap)
        
        image_stats = {
            'image_id': image_id,
            'bg_mode': bg_mode,
            'has_gap': has_gap,
            'is_imperfect': is_imperfect,
            'gap_size': gap,
            'num_cells': len(cells),
            'rows': rows,
            'cols': cols,
            'has_outer_lines': has_outer_lines,
            'image_width': image_width,
            'image_height': image_height,
            'table_width': table_bbox[2] - table_bbox[0],
            'table_height': table_bbox[3] - table_bbox[1],
            'margins': margins,
            'avg_cell_width': np.mean([cell[2] - cell[0] for cell in cells]),
            'avg_cell_height': np.mean([cell[3] - cell[1] for cell in cells]),
            'min_cell_width': min([cell[2] - cell[0] for cell in cells]),
            'min_cell_height': min([cell[3] - cell[1] for cell in cells]),
            'max_cell_width': max([cell[2] - cell[0] for cell in cells]),
            'max_cell_height': max([cell[3] - cell[1] for cell in cells]),
            'avg_line_width': np.mean(line_widths),
            'min_line_width': min(line_widths),
            'max_line_width': max(line_widths),
            'num_merged_cells': sum(1 for cell in cells if len(cell) > 6 and cell[6]),
            'merged_cells_info': [
                {
                    'start_row': cell[4],
                    'start_col': cell[5],
                    'end_row': cell[4] + ((cell[3] - cell[1]) // ((table_bbox[3] - table_bbox[1]) // rows)),
                    'end_col': cell[5] + ((cell[2] - cell[0]) // ((table_bbox[2] - table_bbox[0]) // cols))
                }
                for cell in cells if len(cell) > 6 and cell[6]
            ]
        }
        
        return img, coco_annotations, image_stats
    except Exception as e:
        logger.error(f"Error generating image {image_id}: {str(e)}")
        print(f"Debug: Error occurred in generate_image_and_labels - {str(e)}")
        import traceback
        print(traceback.format_exc())
        return None, None, None

def process_dataset(output_dir, num_images, imperfect_ratio, start_id, subset):
    dataset_info = []
    coco_annotations = []
    
    for i in range(num_images):
        image_id = start_id + i
        
        resolution, margins = generate_random_resolution()
        
        img, annotations, image_stats = generate_image_and_labels(
            image_id,
            resolution,
            margins,
            random.choice(['light', 'dark']),
            random.choice([True, False]),
            random.random() < imperfect_ratio
        )
        
        if img is not None:
            image_stats['subset'] = subset
            img_path = os.path.join(output_dir, subset, 'images', f"{image_stats['image_id']:06d}.png")
            img.save(img_path)
            dataset_info.append(image_stats)
            coco_annotations.extend(annotations)
    
    return dataset_info, coco_annotations

def batch_dataset(output_dir, total_num_images, batch_size=1000, imperfect_ratio=0.3, train_ratio=0.8, max_workers=None):
    """배치 단위로 데이터셋 생성"""
    os.makedirs(output_dir, exist_ok=True)
    
    for subset in ['train', 'val']:
        os.makedirs(os.path.join(output_dir, subset, 'images'), exist_ok=True)
    
    results = {'train': [], 'val': []}
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for subset in ['train', 'val']:
            subset_total = int(total_num_images * (train_ratio if subset == 'train' else 1-train_ratio))
            
            futures = []
            start_id = 1
            for start in range(0, subset_total, batch_size):
                end = min(start + batch_size, subset_total)
                future = executor.submit(
                    process_dataset,
                    output_dir=output_dir, 
                    num_images=end - start, 
                    imperfect_ratio=imperfect_ratio,
                    start_id=start_id,
                    subset=subset
                )
                futures.append(future)
                start_id += batch_size
            
            all_dataset_info = []
            all_coco_annotations = []
            
            for future in tqdm(futures, desc=f"Processing {subset} dataset"):
                batch_dataset_info, batch_coco_annotations = future.result()
                all_dataset_info.extend(batch_dataset_info)
                all_coco_annotations.extend(batch_coco_annotations)
            
            # 각 서브셋의 결과를 즉시 저장
            save_subset_results(output_dir, subset, all_dataset_info, all_coco_annotations)
            
            results[subset] = all_dataset_info
    
    print("Dataset generation completed.")
    
    # 실제 생성된 파일 수 확인
    train_files = len([f for f in os.listdir(os.path.join(output_dir, 'train', 'images')) if f.endswith('.png')])
    val_files = len([f for f in os.listdir(os.path.join(output_dir, 'val', 'images')) if f.endswith('.png')])
    
    print(f"Actual files generated - Train: {train_files}, Validation: {val_files}")
    
    return results

def save_subset_results(output_dir, subset, dataset_info, coco_annotations):
    """서브셋 결과 저장"""
    # 데이터셋 정보 저장
    with open(os.path.join(output_dir, f'{subset}_dataset_info.json'), 'w') as f:
        json.dump(dataset_info, f, indent=4)
    
    # COCO 형식 주석 저장
    coco_format = {
        "images": [{"id": info['image_id'], "file_name": f"{info['image_id']:06d}.png", 
                    "width": info['image_width'], "height": info['image_height']} 
                   for info in dataset_info],
        "annotations": coco_annotations,
        "categories": [
            {"id": 1, "name": "normal_cell"},
            {"id": 2, "name": "merged_cell"},
            {"id": 3, "name": "table"}
        ]
    }
    with open(os.path.join(output_dir, f'{subset}_annotations.json'), 'w') as f:
        json.dump(coco_format, f)
        
def save_dataset_info(output_dir, dataset_info, coco_annotations):
    """전체 데이터셋 정보 저장"""
    import json
    
    # 데이터셋 정보 저장
    with open(os.path.join(output_dir, 'dataset_info.json'), 'w') as f:
        json.dump(dataset_info, f, indent=4)
    
    # COCO 형식 주석 저장
    for subset in ['train', 'val']:
        # 전역 annotation_id 카운터 초기화
        annotation_id = 1
        
        filtered_annotations = []
        for ann in coco_annotations[subset]:
            if ann['category_name'] in ['normal_cell', 'merged_cell', 'row', 'column', 'table']:
                # bbox 유효성 검사
                image_info = next(img for img in dataset_info[subset] if img['image_id'] == ann['image_id'])
                if validate_bbox(ann['bbox'], image_info['image_width'], image_info['image_height']):
                    ann['id'] = annotation_id
                    annotation_id += 1
                    filtered_annotations.append(ann)
        
        # 카테고리 ID 재매핑
        category_id_map = {'normal_cell': 1, 'merged_cell': 2, 'row': 3, 'column': 4, 'table': 5}
        for ann in filtered_annotations:
            ann['category_id'] = category_id_map[ann['category_name']]
        
        coco_format = {
            "images": [{"id": info['image_id'], "file_name": f"{info['image_id']:06d}.png", 
                        "width": info['image_width'], "height": info['image_height']} 
                       for info in dataset_info[subset]],
            "annotations": filtered_annotations,
            "categories": [
                {"id": 1, "name": "normal_cell"},
                {"id": 2, "name": "merged_cell"},
                {"id": 3, "name": "table"}
            ]

        }
        with open(os.path.join(output_dir, f'{subset}_annotations.json'), 'w') as f:
            json.dump(coco_format, f)

def validate_bbox(bbox, image_width, image_height):
    """bbox 유효성 검사"""
    x, y, width, height = bbox
    if x < 0 or y < 0 or x + width > image_width or y + height > image_height:
        return False
    return True

def save_dataset_info_simple(output_dir, dataset_info, coco_annotations):
    """간소화된 데이터셋 정보 저장"""
    import json
    
    # 데이터셋 정보 저장
    with open(os.path.join(output_dir, 'dataset_info.json'), 'w') as f:
        json.dump(dataset_info, f, indent=4)
    
    # COCO 형식 주석 저장
    for subset in ['train', 'val']:
        # 이미지 정보를 딕셔너리로 변환
        image_info_dict = {info['image_id']: info for info in dataset_info[subset]}
        
        # 전역 annotation_id 카운터 초기화
        annotation_id = 1
        
        filtered_annotations = []
        batch_size = 1000  # 배치 크기 설정
        
        for i in range(0, len(coco_annotations[subset]), batch_size):
            batch = coco_annotations[subset][i:i+batch_size]
            
            for ann in batch:
                if ann['category_name'] in ['normal_cell', 'merged_cell', 'table']:
                    # bbox 유효성 검사
                    image_info = image_info_dict.get(ann['image_id'])
                    if image_info and validate_bbox(ann['bbox'], image_info['image_width'], image_info['image_height']):
                        new_ann = {
                            'id': annotation_id,
                            'image_id': ann['image_id'],
                            'category_id': {'normal_cell': 1, 'merged_cell': 2, 'table': 3}[ann['category_name']],
                            'bbox': ann['bbox'],
                            'area': ann['area'],
                            'iscrowd': ann['iscrowd'],
                            'category_name': ann['category_name']
                        }
                        
                        # 셀 정보 추가
                        if ann['category_name'] in ['normal_cell', 'merged_cell']:
                            new_ann.update({
                                'row': ann.get('row', -1),
                                'col': ann.get('col', -1),
                                'is_merged': ann.get('is_merged', False)
                            })
                            if new_ann['is_merged']:
                                new_ann.update({
                                    'merge_start': ann.get('merge_start', {}),
                                    'merge_end': ann.get('merge_end', {})
                                })
                        
                        # 테이블 정보 추가
                        if ann['category_name'] == 'table':
                            new_ann.update({
                                'rows': ann.get('rows', -1),
                                'cols': ann.get('cols', -1),
                                'cell_count': ann.get('cell_count', -1)
                            })
                        
                        filtered_annotations.append(new_ann)
                        annotation_id += 1
            
            # 메모리 해제를 위해 처리된 배치 삭제
            del batch
        
        coco_format = {
            "images": [{"id": info['image_id'], "file_name": f"{info['image_id']:06d}.png", 
                        "width": info['image_width'], "height": info['image_height']} 
                       for info in dataset_info[subset]],
            "annotations": filtered_annotations,
            "categories": [
                {"id": 1, "name": "normal_cell"},
                {"id": 2, "name": "merged_cell"},
                {"id": 3, "name": "table"}
            ]
        }
        
        with open(os.path.join(output_dir, f'{subset}_annotations.json'), 'w') as f:
            json.dump(coco_format, f)
        
        # 메모리 해제
        del filtered_annotations
        del coco_format

if __name__ == "__main__":
    output_dir = 'yolox_table_dataset_simple233-0919-2' # 데이터셋 출력될 경로
    num_images = 60000 # 표를 최대 몇개까지?
    imperfect_ratio = 0.2 # 20% 확률로 불완전한 표로 생성됨(셀의 선이 없거나, 돌출부가 있거나 노이즈 등 추가된 표)
    train_ratio = 0.8 # 80% 학습, 20% 검증
    batch_size = 1000 # 한번에 몇개까지 만들지

    print(f"Starting dataset generation: {num_images} images, batch size: {batch_size}")
    
    dataset_info = batch_dataset(
        output_dir=output_dir, 
        total_num_images=num_images, 
        batch_size=batch_size,
        imperfect_ratio=imperfect_ratio, 
        train_ratio=train_ratio
    )

    if dataset_info:
        print("Dataset generation completed.")
        print(f"Total images generated (according to dataset_info): {sum(len(info) for info in dataset_info.values())}")
        print(f"Train images (according to dataset_info): {len(dataset_info['train'])}")
        print(f"Validation images (according to dataset_info): {len(dataset_info['val'])}")
        
        # 실제 생성된 파일 수 다시 확인
        train_files = len([f for f in os.listdir(os.path.join(output_dir, 'train', 'images')) if f.endswith('.png')])
        val_files = len([f for f in os.listdir(os.path.join(output_dir, 'val', 'images')) if f.endswith('.png')])
        print(f"Actual files in folders - Train: {train_files}, Validation: {val_files}")
        
        print(f"Dataset saved in: {output_dir}")
    else:
        print("Dataset generation failed.")

    # 추가적인 디버깅 정보
    print("\nAdditional debugging information:")
    print(f"Total expected images: {num_images}")
    print(f"Expected train images: {int(num_images * train_ratio)}")
    print(f"Expected validation images: {num_images - int(num_images * train_ratio)}")
