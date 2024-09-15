import os

import random
from tqdm import tqdm
import numpy as np
import logging
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from concurrent.futures import ProcessPoolExecutor
from opti_calc import *
from rollback_draw import *
from rollback_constant import *
from rollback_utils import *
from rollback_files import *
import itertools

# 전역 카운터 생성
global_counter = itertools.count(1)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
def generate_random_resolution():
    width = random.randint(MIN_IMAGE_WIDTH, MAX_IMAGE_WIDTH)
    height = random.randint(MIN_IMAGE_HEIGHT, MAX_IMAGE_HEIGHT)
    
    width = width - (width % 8)
    height = height - (height % 8)
    
    return (width, height)


def create_table(image_width, image_height, has_gap=False):
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
                new_cell = [
                        cells[start_row * cols + start_col][0],
                        cells[start_row * cols + start_col][1],
                        cells[(start_row + merge_rows - 1) * cols + (start_col + merge_cols - 1)][2],
                        cells[(start_row + merge_rows - 1) * cols + (start_col + merge_cols - 1)][3],
                        start_row, start_col, True  # True는 병합된 셀을 의미
                ]
                for r in range(start_row, start_row + merge_rows):
                    for c in range(start_col, start_col + merge_cols):
                        merged_cells[r * cols + c] = None
                
                merged_cells[start_row * cols + start_col] = new_cell
                merged_areas.append(merge_area)
                break
            
            attempts += 1

    return [cell for cell in merged_cells if cell is not None]

def is_overlapping(area1, area2):
    return not (area1[2] <= area2[0] or area1[0] >= area2[2] or
                area1[3] <= area2[1] or area1[1] >= area2[3])


def generate_coco_annotations(cells, table_bbox, image_width, image_height, rows, cols, image_id):
    coco_annotations = []
    annotation_id = 1

    # Row categories
    for row in range(rows):
        row_cells = [cell for cell in cells if cell[4] == row]
        if row_cells:
            min_x = min(cell[0] for cell in row_cells)
            max_x = max(cell[2] for cell in row_cells)
            y = row_cells[0][1]
            height = row_cells[0][3] - row_cells[0][1]
            coco_annotations.append({
                "id": annotation_id,
                "image_id": image_id,
                "category_id": 3,  # Row category
                "bbox": [min_x, y, max_x - min_x, height],
                "area": (max_x - min_x) * height,
                "iscrowd": 0
            })
            annotation_id += 1

    # Column categories
    for col in range(cols):
        col_cells = [cell for cell in cells if cell[5] == col]
        if col_cells:
            min_y = min(cell[1] for cell in col_cells)
            max_y = max(cell[3] for cell in col_cells)
            x = col_cells[0][0]
            width = col_cells[0][2] - col_cells[0][0]
            coco_annotations.append({
                "id": annotation_id,
                "image_id": image_id,
                "category_id": 4,  # Column category
                "bbox": [x, min_y, width, max_y - min_y],
                "area": width * (max_y - min_y),
                "iscrowd": 0
            })
            annotation_id += 1

    # Cell annotations
    for cell in cells:
        is_merged = cell[6] if len(cell) > 6 else False  # 병합된 셀인지 여부
        category_id = 2 if is_merged else 1  # 2 for merged cell, 1 for normal cell
        width = cell[2] - cell[0]
        height = cell[3] - cell[1]
        coco_annotations.append({
            "id": annotation_id,
            "image_id": image_id,
            "category_id": category_id,
            "bbox": [cell[0], cell[1], width, height],
            "area": width * height,
            "iscrowd": 0,
            "is_merged": is_merged
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
        "iscrowd": 0
    })

    return coco_annotations

def create_coco_annotation(category_id, x, y, width, height, annotation_id, image_id, row=None, column=None, is_merged=False, merged_rows=None, merged_cols=None):
    annotation = {
        "id": annotation_id,
        "image_id": image_id,
        "category_id": category_id + 1,
        "bbox": [x, y, width, height],
        "area": width * height,
        "iscrowd": 0,
        "row": row,
        "column": column,
        "is_merged": is_merged
    }
    if is_merged:
        annotation["merged_rows"] = merged_rows
        annotation["merged_cols"] = merged_cols
    return annotation

def generate_image_and_labels(image_id, resolution, bg_mode, has_gap, is_imperfect=False):
    try:
        image_width, image_height = resolution  # 랜덤 해상도 사용

        cells, table_bbox, gap, rows, cols = create_table(image_width, image_height, has_gap)
        
        bg_color = random.choice(list(BACKGROUND_COLORS[bg_mode].values()))
        img = Image.new('L', (image_width, image_height), color=bg_color)
        draw = ImageDraw.Draw(img)
        
        line_widths, has_outer_lines = draw_table(draw, cells, table_bbox, bg_color, has_gap, is_imperfect)
        
        add_content_to_cells(draw, cells, random.choice(FONTS), bg_color)
        if is_imperfect:
            img = apply_imperfections(img, cells)
        
        coco_annotations = generate_coco_annotations(cells, table_bbox, image_width, image_height, rows, cols, image_id)

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
            'num_merged_cells': sum(1 for cell in cells if len(cell) > 6)
        }
        
        return img, coco_annotations, image_stats
    except Exception as e:
        logger.error(f"Error generating image {image_id}: {str(e)}")
        return None, None, None, None
def batch_dataset(output_dir, total_num_images, batch_size=1000, imperfect_ratio=0.3, train_ratio=0.8):
    # 출력 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)
    
    for subset in ['train', 'val']:
        os.makedirs(os.path.join(output_dir, subset, 'images'), exist_ok=True)
        
        
    last_ids = load_last_image_ids(output_dir)
    
    all_dataset_info = {subset: [] for subset in ['train', 'val']}
    all_stats = []
    
    for subset in ['train', 'val']:
        subset_total = int(total_num_images * (train_ratio if subset == 'train' else 1-train_ratio))
        global_counter = itertools.count(last_ids[subset] + 1)
        
        for start in range(0, subset_total, batch_size):
            end = min(start + batch_size, subset_total)
            batch_dataset_info, batch_stats = process_dataset(
                output_dir=output_dir, 
                num_images=end - start, 
                imperfect_ratio=imperfect_ratio,
                global_counter=global_counter,
                subset=subset
            )
            
            all_dataset_info[subset].extend(batch_dataset_info[subset])
            all_stats.extend(batch_stats)
            
            gc.collect()
        
        last_ids[subset] = next(global_counter) - 1
    
    save_last_image_id(output_dir, last_ids)
    save_dataset_info(output_dir, all_dataset_info, all_stats)
    return all_dataset_info

def process_dataset(output_dir, num_images, imperfect_ratio=0.3, global_counter=None, subset='train'):
    create_directory(output_dir)
    create_directory(os.path.join(output_dir, subset, 'images'))
    
    dataset_info = {subset: []}
    stats = []
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
            save_image_and_annotations(img, image_stats, output_dir, subset)
            dataset_info[subset].append(image_stats)
            stats.append(image_stats)
            all_coco_annotations.extend(coco_annotations)

    save_coco_annotations(output_dir, dataset_info, all_coco_annotations, subset)
    return dataset_info, stats


def determine_subset(train_ratio, val_ratio):
    rand = random.random()
    if rand < train_ratio:
        return 'train'
    elif rand < train_ratio + val_ratio:
        return 'val'
    else:
        return 'test'


if __name__ == "__main__":
    output_dir = 'table_dataset_real'
    num_images = 100
    imperfect_ratio = 0.1
    train_ratio = 0.8
    dataset_info = batch_dataset(output_dir, num_images, imperfect_ratio=imperfect_ratio, train_ratio=train_ratio)
    print("Dataset generation completed.")
