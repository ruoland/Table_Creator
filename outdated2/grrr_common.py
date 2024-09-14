import os
import random
import numpy as np
import logging
from PIL import Image, ImageDraw
from concurrent.futures import ProcessPoolExecutor
import yaml
import csv
import json
import itertools, gc
from tqdm import tqdm
from grrr_stats import calculate_summary_stats
from grrr_draw import draw_table, add_content_to_cells, apply_imperfections
from grrr_constants import *
from grrr_utils import create_directory

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
random.seed(RANDOM_SEED)
# 전역 카운터 생성
global_counter = itertools.count(1)
def create_table(image_width, image_height, has_gap=False):
    cols = random.randint(MIN_COLS, MAX_COLS)
    rows = random.randint(MIN_ROWS, MAX_ROWS)
    
    cell_width = min(max(image_width // cols, BASE_CELL_WIDTH), image_width // MIN_COLS)
    cell_height = min(max(image_height // rows, BASE_CELL_HEIGHT), image_height // MIN_ROWS)
    
    # 테이블 크기 조정
    cols = min(cols, image_width // cell_width)
    rows = min(rows, image_height // cell_height)
    
    gap = random.randint(1, 3) if has_gap else 0
    
    cells = [
        [col * cell_width + gap,
         row * cell_height + gap,
         (col + 1) * cell_width - gap,
         (row + 1) * cell_height - gap,
         row, col]
        for row in range(rows)
        for col in range(cols)
    ]
    
    cells = merge_cells(cells, rows, cols)
    
    return cells, [0, 0, cols * cell_width, rows * cell_height], gap, rows, cols
def merge_cells(cells, rows, cols):
    merged_cells = cells.copy()
    num_merges = int(len(cells) * MERGED_CELL_RATIO)
    merged_areas = []

    for _ in range(num_merges):
        attempts = 0
        while attempts < 20:
            start_row = random.randint(0, rows - 1)
            start_col = random.randint(0, cols - 1)
            merge_rows = random.randint(1, min(2, rows - start_row))
            merge_cols = random.randint(1, min(2, cols - start_col))
            
            if merge_rows == 1 and merge_cols == 1:
                continue

            merge_area = (start_row, start_col, start_row + merge_rows, start_col + merge_cols)
            
            if not any(is_overlapping(merge_area, area) for area in merged_areas):
                new_cell = [
                    cells[start_row * cols + start_col][0],
                    cells[start_row * cols + start_col][1],
                    cells[(start_row + merge_rows - 1) * cols + (start_col + merge_cols - 1)][2],
                    cells[(start_row + merge_rows - 1) * cols + (start_col + merge_cols - 1)][3],
                    start_row, start_col, merge_rows, merge_cols
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

def get_last_image_id(output_dir):
    last_id = 0
    for subset in ['train', 'val', 'test']:
        image_dir = os.path.join(output_dir, subset, 'images')
        if os.path.exists(image_dir):
            image_files = [f for f in os.listdir(image_dir) if f.endswith('.png')]
            if image_files:
                last_file = max(image_files)
                last_id = max(last_id, int(last_file[6:-4]))  # 'table_00123.png' -> 123
    return last_id

def save_last_image_id(output_dir, last_id):
    with open(os.path.join(output_dir, 'last_image_id.json'), 'w') as f:
        json.dump({'last_image_id': last_id}, f)

def load_last_image_id(output_dir):
    try:
        with open(os.path.join(output_dir, 'last_image_id.json'), 'r') as f:
            data = json.load(f)
            return data['last_image_id']
    except FileNotFoundError:
        return 0

def generate_header_info(col):
    if col == 0:  # 첫 번째 열
        header_type = random.choice(['day', 'time'])
        if header_type == 'day':
            return random.choice(DAYS), 'day'
        else:
            return random.choice(TIMES), 'time'
    else:  # 나머지 열
        return random.choice(GRADES), 'grade'

def generate_image_and_labels(image_id, resolution, bg_mode, has_gap, is_imperfect=False):
    image_width, image_height = resolution
    cells, table_bbox, gap, rows, cols = create_table(image_width, image_height, has_gap)
    
    if not cells:  # cells가 비어있는 경우 처리
        logger.warning(f"No cells generated for image {image_id}")
        return None, None, None, None

    bg_color = random.choice(list(BACKGROUND_COLORS[bg_mode].values()))
    img = Image.new('L', (image_width, image_height), color=bg_color)
    draw = ImageDraw.Draw(img)
    
    line_widths, has_outer_lines = draw_table(draw, cells, table_bbox, bg_color, has_gap, is_imperfect)
    
   # 헤더 정보 생성 및 추가
    header_cells = []
    for i in range(cols):
        header_text, header_type = generate_header_info(i)
        
        if len(cells) <= i:
            logger.warning(f"Not enough cells for header at column {i}")
            break
        header_cell = list(cells[i])[:6]  # 기존 셀 정보 복사 (병합 정보 제외)
        header_cell.extend([header_text, header_type])  # 헤더 정보 추가
        header_cells.append(tuple(header_cell))

    # 헤더 셀을 cells 리스트의 앞부분에 추가
    cells = header_cells + cells[cols:]

    add_content_to_cells(draw, cells, random.choice(FONTS), bg_color)
    if is_imperfect:
        img = apply_imperfections(img, cells)

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
        'num_merged_cells': sum(1 for cell in cells if len(cell) > 6),
        'num_header_cells': cols  # 헤더 셀 개수 추가
    }
    
    return img, cells, table_bbox, image_stats

def generate_large_dataset_in_batches(output_dir, total_num_images, batch_size=1000, imperfect_ratio=0.3, train_ratio=0.8, val_ratio=0):
    last_id = load_last_image_id(output_dir)
    if last_id == 0:
        last_id = get_last_image_id(output_dir)
    
    global_counter = itertools.count(last_id + 1)
    all_dataset_info = {subset: [] for subset in ['train', 'val', 'test']}
    all_stats = []
    
    for start in range(last_id, total_num_images, batch_size):
        end = min(start + batch_size, total_num_images)
        batch_dataset_info, batch_stats = generate_large_dataset(output_dir, end - start, imperfect_ratio, train_ratio, val_ratio, global_counter)
        
        for subset in all_dataset_info:
            all_dataset_info[subset].extend(batch_dataset_info[subset])
        all_stats.extend(batch_stats)
        
        gc.collect()
    
    save_last_image_id(output_dir, next(global_counter) - 1)
    save_dataset_info(output_dir, all_dataset_info, all_stats)
    return all_dataset_info
def generate_random_resolution():
    width = random.randint(MIN_IMAGE_WIDTH, MAX_IMAGE_WIDTH)
    height = random.randint(MIN_IMAGE_HEIGHT, MAX_IMAGE_HEIGHT)
    
    width = width - (width % 8)
    height = height - (height % 8)
    
    return (width, height)

def generate_large_dataset(output_dir, num_images, imperfect_ratio=0.3, train_ratio=0.8, val_ratio=0.1, global_counter=None):
    create_directory(output_dir)
    for subset in ['train', 'val', 'test']:
        create_directory(os.path.join(output_dir, subset, 'images'))
        create_directory(os.path.join(output_dir, subset, 'labels'))
    
    dataset_info = {subset: [] for subset in ['train', 'val', 'test']}
    stats = []

    with ProcessPoolExecutor() as executor:
        futures = [
            executor.submit(generate_image_and_labels, next(global_counter), 
                            generate_random_resolution(), 
                            random.choice(['light', 'dark']), 
                            random.choice([True, False]), 
                            random.random() < imperfect_ratio)
            for _ in range(num_images)
        ]
        
        for future in tqdm(futures, total=num_images, desc="Generating images"):
            result = future.result()
            if result[0] is None:
                continue
            img, cells, table_bbox, image_stats = result
            subset = determine_subset(train_ratio, val_ratio)
            image_stats['subset'] = subset
            save_image_and_labels(img, cells, image_stats, output_dir, subset)
            dataset_info[subset].append(image_stats)
            stats.append(image_stats)

    save_dataset_info(output_dir, dataset_info, stats)
    return dataset_info

def determine_subset(train_ratio, val_ratio):
    rand = random.random()
    if rand < TRAIN_RATIO:
        return 'train'
    elif rand < TRAIN_RATIO + VAL_RATIO:
        return 'val'
    else:
        return 'test'
    
def save_image_and_labels(img, cells, image_stats, output_dir, subset):
    img_filename = f"table_{image_stats['image_id']:06d}.png"
    img_path = os.path.join(output_dir, subset, 'images', img_filename)
    img.save(img_path)

    label_filename = f"table_{image_stats['image_id']:06d}.txt"
    label_path = os.path.join(output_dir, subset, 'labels', label_filename)
    with open(label_path, 'w') as f:
        for cell in cells:
            x_center = (cell[0] + cell[2]) / 2 / image_stats['image_width']
            y_center = (cell[1] + cell[3]) / 2 / image_stats['image_height']
            width = (cell[2] - cell[0]) / image_stats['image_width']
            height = (cell[3] - cell[1]) / image_stats['image_height']
            
            if len(cell) > 7:  # 헤더 셀
                class_id = 2
            elif len(cell) > 6:  # 병합된 셀
                class_id = 1
            else:  # 일반 셀
                class_id = 0
            
            f.write(f"{class_id} {x_center} {y_center} {width} {height}\n")

def save_dataset_info(output_dir, dataset_info, stats):
    total_images = sum(len(info) for info in dataset_info.values())
    print(f"총 생성된 이미지 수: {total_images}")

    for subset, info in dataset_info.items():
        print(f"{subset.capitalize()} 세트:")
        print(f"  총 이미지 수: {len(info)}")
        print(f"  밝은 이미지 수: {sum(1 for item in info if item['bg_mode'] == 'light')}")
        print(f"  어두운 이미지 수: {sum(1 for item in info if item['bg_mode'] == 'dark')}")
        print(f"  간격 있는 이미지 수: {sum(1 for item in info if item['has_gap'])}")
        print(f"  불완전 이미지 수: {sum(1 for item in info if item['is_imperfect'])}")
        print()

        yaml_content = {
        'train': os.path.join(output_dir, 'train', 'images'),
        'val': os.path.join(output_dir, 'val', 'images'),
        'test': os.path.join(output_dir, 'test', 'images'),
        'nc': 6,  # cell, merged_cell, row, column, table, header_cell
        'names': ['cell', 'merged_cell', 'row', 'column', 'table', 'header_cell']
    }
    
    with open(os.path.join(output_dir, 'dataset.yaml'), 'w') as f:
        yaml.dump(yaml_content, f)

    with open(os.path.join(output_dir, 'dataset_stats.csv'), 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=stats[0].keys())
        writer.writeheader()
        writer.writerows(stats)

    summary_stats = calculate_summary_stats(stats)
    with open(os.path.join(output_dir, 'summary_stats.json'), 'w', encoding='utf-8') as f:
        json.dump(summary_stats, f, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    output_dir = 'table_dataset_new_opti22222o'
    num_images = 10
    imperfect_ratio = 0.1
    train_ratio = 0.8
    val_ratio = 0.1
    dataset_info = generate_large_dataset_in_batches(output_dir, num_images, imperfect_ratio=imperfect_ratio, train_ratio=train_ratio, val_ratio=val_ratio)
    print("Dataset generation completed.")
