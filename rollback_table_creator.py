import os
import random
from tqdm import tqdm
import numpy as np
import logging
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from concurrent.futures import ProcessPoolExecutor
import itertools
from opti_calc import *
from rollback_draw import *
from rollback_constant import *
from rollback_utils import *
from rollback_files import *

# 전역 설정
MIN_IMAGE_WIDTH, MAX_IMAGE_WIDTH = 640, 1280
MIN_IMAGE_HEIGHT, MAX_IMAGE_HEIGHT = 480, 960
MIN_COLS, MAX_COLS = 2, 10
MIN_ROWS, MAX_ROWS = 2, 10
BASE_CELL_WIDTH, BASE_CELL_HEIGHT = 50, 30

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def generate_random_resolution():
    width = random.randint(MIN_IMAGE_WIDTH, MAX_IMAGE_WIDTH)
    height = random.randint(MIN_IMAGE_HEIGHT, MAX_IMAGE_HEIGHT)
    width = width - (width % 32)
    height = height - (height % 32)
    return (width, height)

def create_table(image_width, image_height, has_gap=False):
    # 기존 create_table 함수 그대로 사용
    cols = random.randint(MIN_COLS, MAX_COLS)
    rows = random.randint(MIN_ROWS, MAX_ROWS)
    
    cell_width = max(image_width // cols, BASE_CELL_WIDTH)
    cell_height = max(image_height // rows, BASE_CELL_HEIGHT)
    total_height = rows * cell_height
    if total_height > image_height:
        cell_height = image_height // rows
    gap = random.randint(1, 3) if has_gap else 0
    
    cells = [
        [col * cell_width + (gap if has_gap else 0),
         row * cell_height + (gap if has_gap else 0),
         (col + 1) * cell_width - (gap if has_gap else 0),
         (row + 1) * cell_height - (gap if has_gap else 0),
         row, col, False]  # False는 병합되지 않은 셀을 의미
        for row in range(rows)
        for col in range(cols)
    ]
    
    cells = merge_cells(cells, rows, cols)
    
    return cells, [0, 0, image_width, image_height], gap, rows, cols

def merge_cells(cells, rows, cols):
    merged_cells = cells.copy()
    num_merges = random.randint(1, min(rows, cols) // 2)
    merged_areas = []

    for _ in range(num_merges):
        attempts = 0
        while attempts < 10:
            start_row = random.randint(0, rows - 2)
            start_col = random.randint(0, cols - 2)
            merge_rows = random.randint(2, min(3, rows - start_row))
            merge_cols = random.randint(2, min(3, cols - start_col))
            
            merge_area = (start_row, start_col, start_row + merge_rows, start_col + merge_cols)
            
            if not any(is_overlapping(merge_area, area) for area in merged_areas):
                new_cell_info = [
                    cells[start_row * cols + start_col][0],
                    cells[start_row * cols + start_col][1],
                    cells[(start_row + merge_rows - 1) * cols + (start_col + merge_cols - 1)][2],
                    cells[(start_row + merge_rows - 1) * cols + (start_col + merge_cols - 1)][3],
                    start_row, start_col, True  # True는 병합된 셀을 의미
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
    return not (area1[2] <= area2[0] or area1[0] >= area2[2] or
                area1[3] <= area2[1] or area1[1] >= area2[3])
    
def generate_coco_annotations(cells, table_bbox, rows, cols, image_id, gap):
    coco_annotations = []
    annotation_id = 1

    # Cell annotations
    cell_annotations = {}
    for i, cell_info in enumerate(cells):
        is_merged = cell_info[6] if len(cell_info) > 6 else False
        category_id = 2 if is_merged else 1
        category_name = "merged_cell" if is_merged else "normal_cell"
        width = cell_info[2] - cell_info[0]
        height = cell_info[3] - cell_info[1]
        cell_annotation = {
            "id": annotation_id,
            "image_id": image_id,
            "category_id": category_id,
            "bbox": [cell_info[0], cell_info[1], width, height],
            "area": width * height,
            "iscrowd": 0,
            "category_name": category_name
        }
        coco_annotations.append(cell_annotation)
        cell_annotations[i] = annotation_id
        annotation_id += 1

    # Row categories
    for row in range(rows):
        row_cells = [cell for cell in cells if cell[4] == row]
        if row_cells:
            min_x = min(cell[0] for cell in row_cells)
            max_x = max(cell[2] for cell in row_cells)
            y = min(cell[1] for cell in row_cells)
            max_y = max(cell[3] for cell in row_cells)
            height = max_y - y
            cell_ids = [cell_annotations[cells.index(cell)] for cell in row_cells]
            coco_annotations.append({
                "id": annotation_id,
                "image_id": image_id,
                "category_id": 3,  # Row category
                "bbox": [min_x - gap, y - gap, max_x - min_x + 2*gap, height + 2*gap],
                "area": (max_x - min_x + 2*gap) * (height + 2*gap),
                "iscrowd": 0,
                "category_name": "row",
                "cell_ids": cell_ids
            })
            annotation_id += 1

    # Column categories
    for col in range(cols):
        col_cells = [cell for cell in cells if cell[5] == col]
        if col_cells:
            x = min(cell[0] for cell in col_cells)
            max_x = max(cell[2] for cell in col_cells)
            min_y = min(cell[1] for cell in col_cells)
            max_y = max(cell[3] for cell in col_cells)
            width = max_x - x
            cell_ids = [cell_annotations[cells.index(cell)] for cell in col_cells]
            coco_annotations.append({
                "id": annotation_id,
                "image_id": image_id,
                "category_id": 4,  # Column category
                "bbox": [x - gap, min_y - gap, width + 2*gap, max_y - min_y + 2*gap],
                "area": (width + 2*gap) * (max_y - min_y + 2*gap),
                "iscrowd": 0,
                "category_name": "column",
                "cell_ids": cell_ids
            })
            annotation_id += 1

    # Table label
    table_width = table_bbox[2] - table_bbox[0]
    table_height = table_bbox[3] - table_bbox[1]
    coco_annotations.append({
        "id": annotation_id,
        "image_id": image_id,
        "category_id": 5,  # Table category
        "bbox": [table_bbox[0], table_bbox[1], table_width, table_height],
        "area": table_width * table_height,
        "iscrowd": 0,
        "category_name": "table"
    })

    return coco_annotations

def generate_image_and_labels(image_id, resolution, bg_mode, has_gap, is_imperfect=False):
    try:
        image_width, image_height = resolution
        cells, table_bbox, gap, rows, cols = create_table(image_width, image_height, has_gap)
        
        bg_color = random.choice(list(BACKGROUND_COLORS[bg_mode].values()))
        img = Image.new('L', (image_width, image_height), color=bg_color)
        draw = ImageDraw.Draw(img)
        
        line_widths, has_outer_lines = draw_table(draw, cells, table_bbox, bg_color, has_gap, is_imperfect)
        
        add_content_to_cells(draw, cells, random.choice(FONTS), bg_color)
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
                    'end_row': cell[4] + ((cell[3] - cell[1]) // (table_bbox[3] // rows)),
                    'end_col': cell[5] + ((cell[2] - cell[0]) // (table_bbox[2] // cols))
                }
                for cell in cells if len(cell) > 6 and cell[6]
            ]
        }
        
        return img, coco_annotations, image_stats
    except Exception as e:
        logger.error(f"Error generating image {image_id}: {str(e)}")
        return None, None, None

def process_dataset(output_dir, num_images, imperfect_ratio=0.3, global_counter=None, subset='train'):
    os.makedirs(os.path.join(output_dir, subset, 'images'), exist_ok=True)
    
    dataset_info = []
    all_coco_annotations = []
    
    with ProcessPoolExecutor() as executor:
        futures = [
            executor.submit(generate_image_and_labels, 
                            next(global_counter), 
                            generate_random_resolution(),
                            random.choice(['light', 'dark']), 
                            random.choice([True, False]), 
                            random.random() < imperfect_ratio)
            for _ in range(num_images)
        ]
        
    for future in tqdm(futures, total=num_images, desc=f"{subset} 이미지 표 만드는 중..."):
        img, coco_annotations, image_stats = future.result()
        if img is not None:
            image_stats['subset'] = subset
            
            # 이미지 저장
            img_path = os.path.join(output_dir, subset, 'images', f"{image_stats['image_id']:06d}.png")
            img.save(img_path)
            
            dataset_info.append(image_stats)
            all_coco_annotations.extend(coco_annotations)
    
    return dataset_info, all_coco_annotations

def batch_dataset(output_dir, total_num_images, batch_size=1000, imperfect_ratio=0.3, train_ratio=0.8):
    os.makedirs(output_dir, exist_ok=True)
    
    for subset in ['train', 'val']:
        os.makedirs(os.path.join(output_dir, subset, 'images'), exist_ok=True)
    
    all_dataset_info = {'train': [], 'val': []}
    all_coco_annotations = {'train': [], 'val': []}
    global_counter = itertools.count(1)
    
    for subset in ['train', 'val']:
        subset_total = int(total_num_images * (train_ratio if subset == 'train' else 1-train_ratio))
        
        for start in range(0, subset_total, batch_size):
            end = min(start + batch_size, subset_total)
            batch_dataset_info, batch_coco_annotations = process_dataset(
                output_dir=output_dir, 
                num_images=end - start, 
                imperfect_ratio=imperfect_ratio,
                global_counter=global_counter,
                subset=subset
            )
            
            all_dataset_info[subset].extend(batch_dataset_info)
            all_coco_annotations[subset].extend(batch_coco_annotations)
    
    # 데이터셋 정보 및 COCO 주석 저장
    save_dataset_info_simple(output_dir, all_dataset_info, all_coco_annotations)
    return all_dataset_info
def save_dataset_info(output_dir, dataset_info, coco_annotations):
    import json
    
    # 데이터셋 정보 저장
    with open(os.path.join(output_dir, 'dataset_info.json'), 'w') as f:
        json.dump(dataset_info, f, indent=4)
    
    # COCO 형식 주석 저장
    for subset in ['train', 'val']:
        coco_format = {
            "images": [{"id": info['image_id'], "file_name": f"{info['image_id']:06d}.png", 
                        "width": info['image_width'], "height": info['image_height']} 
                       for info in dataset_info[subset]],
            "annotations": coco_annotations[subset],
            "categories": [
                {"id": 1, "name": "normal_cell"},
                {"id": 2, "name": "merged_cell"},
                {"id": 3, "name": "row"},
                {"id": 4, "name": "column"},
                {"id": 5, "name": "table"}
            ]
        }
        with open(os.path.join(output_dir, f'{subset}_annotations.json'), 'w') as f:
            json.dump(coco_format, f)
def validate_bbox(bbox, image_width, image_height):
    x, y, width, height = bbox
    if x < 0 or y < 0 or x + width > image_width or y + height > image_height:
        print(f"Warning: Invalid bbox {bbox} for image size {image_width}x{image_height}")
        return False
    return True
 
def save_dataset_info_simple(output_dir, dataset_info, coco_annotations):
    import json
    
    # 데이터셋 정보 저장
    with open(os.path.join(output_dir, 'dataset_info.json'), 'w') as f:
        json.dump(dataset_info, f, indent=4)
    
    # COCO 형식 주석 저장
    for subset in ['train', 'val']:
        # 필요한 카테고리만 필터링
        filtered_annotations = [
            ann for ann in coco_annotations[subset]
            if ann['category_name'] in ['cell', 'merged_cell', 'table']
        ]
        
        # 카테고리 ID 재매핑
        category_id_map = {'cell': 1, 'merged_cell': 2, 'table': 3}
        for ann in filtered_annotations:
            ann['category_id'] = category_id_map[ann['category_name']]
        
        coco_format = {
            "images": [{"id": info['image_id'], "file_name": f"{info['image_id']:06d}.png", 
                        "width": info['image_width'], "height": info['image_height']} 
                       for info in dataset_info[subset]],
            "annotations": filtered_annotations,
            "categories": [
                {"id": 1, "name": "cell"},
                {"id": 2, "name": "merged_cell"},
                {"id": 3, "name": "table"}
            ]
        }
        with open(os.path.join(output_dir, f'{subset}_annotations.json'), 'w') as f:
            json.dump(coco_format, f)

if __name__ == "__main__":
    output_dir = 'yolox_table_dataset_simple'
    num_images = 50000
    imperfect_ratio = 0.3
    train_ratio = 0.8
    batch_size = 3000

    dataset_info = batch_dataset(
        output_dir=output_dir, 
        total_num_images=num_images, 
        batch_size=batch_size,
        imperfect_ratio=imperfect_ratio, 
        train_ratio=train_ratio
    )

    print("Dataset generation completed.")
    print(f"Total images generated: {sum(len(info) for info in dataset_info.values())}")
    print(f"Train images: {len(dataset_info['train'])}")
    print(f"Validation images: {len(dataset_info['val'])}")
    print(f"Dataset saved in: {output_dir}")
