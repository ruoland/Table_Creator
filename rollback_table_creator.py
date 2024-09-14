import os
import random
from tqdm import tqdm
import numpy as np
import logging
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from concurrent.futures import ProcessPoolExecutor
import yaml
from opti_calc import *
from grrr_draw import *
from grrr_constants import *
from grrr_utils import *
import itertools

# 전역 카운터 생성
global_counter = itertools.count(1)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
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
         row, col]
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
    
def generate_yolo_labels(cells, table_bbox, image_width, image_height, rows, cols):
    yolo_labels = []
    merged_cells = [cell for cell in cells if len(cell) > 6]

    cell_width = (table_bbox[2] - table_bbox[0]) / cols
    cell_height = (table_bbox[3] - table_bbox[1]) / rows

    # Row labels
    for row in range(rows):
        row_cells = [cell for cell in cells if cell[4] == row and cell not in merged_cells]
        if row_cells:
            min_x = min(cell[0] for cell in row_cells)
            max_x = max(cell[2] for cell in row_cells)
            y = row_cells[0][1]
            height = cell_height
            yolo_labels.append(create_yolo_label(2, min_x, y, max_x - min_x, height, image_width, image_height))

    # Column labels
    for col in range(cols):
        col_cells = [cell for cell in cells if cell[5] == col and cell not in merged_cells]
        if col_cells:
            min_y = min(cell[1] for cell in col_cells)
            max_y = max(cell[3] for cell in col_cells)
            x = col_cells[0][0]
            width = cell_width
            yolo_labels.append(create_yolo_label(3, x, min_y, width, max_y - min_y, image_width, image_height))

    # Cell labels
    for cell in cells:
        class_id = 1 if len(cell) > 6 else 0  # merged cell or normal cell
        yolo_labels.append(create_yolo_label(class_id, cell[0], cell[1], cell[2] - cell[0], cell[3] - cell[1], image_width, image_height))

    # Table label
    yolo_labels.append(create_yolo_label(4, table_bbox[0], table_bbox[1], table_bbox[2] - table_bbox[0], table_bbox[3] - table_bbox[1], image_width, image_height))

    return yolo_labels
def create_yolo_label(class_id, x, y, width, height, image_width, image_height):
    x_center = (x + width / 2) / image_width
    y_center = (y + height / 2) / image_height
    width = width / image_width
    height = height / image_height
    return f"{class_id} {x_center:.4f} {y_center:.4f} {width:.4f} {height:.4f}"

def generate_image_and_labels(image_id, resolution, bg_mode, has_gap, is_imperfect=False):
    image_width, image_height = resolution
    cols = random.randint(MIN_COLS, MAX_COLS)
    rows = random.randint(MIN_ROWS, MAX_ROWS)

    cells, table_bbox, gap, rows, cols = create_table(image_width, image_height, has_gap)
    
    bg_color = random.choice(list(BACKGROUND_COLORS[bg_mode].values()))
    img = Image.new('L', (image_width, image_height), color=bg_color)
    draw = ImageDraw.Draw(img)
    
    line_widths, has_outer_lines = draw_table(draw, cells, table_bbox, bg_color, has_gap, is_imperfect)
    
    add_content_to_cells(draw, cells, random.choice(FONTS), bg_color)
    if is_imperfect:
        img = apply_imperfections(img, cells)
    
    yolo_labels = generate_yolo_labels(cells, table_bbox, image_width, image_height, rows, cols)

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
    
    return img, yolo_labels, image_stats
import gc
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
            img, yolo_labels, image_stats = future.result()
            subset = determine_subset(train_ratio, val_ratio)
            image_stats['subset'] = subset
            save_image_and_labels(img, yolo_labels, image_stats, output_dir, subset)
            dataset_info[subset].append(image_stats)
            stats.append(image_stats)

    save_dataset_info(output_dir, dataset_info, stats)
    return dataset_info
def determine_subset(train_ratio, val_ratio):
    rand = random.random()
    if rand < train_ratio:
        return 'train'
    elif rand < train_ratio + val_ratio:
        return 'val'
    else:
        return 'test'
def save_image_and_labels(img, yolo_labels, image_stats, output_dir, subset):
    img_filename = f"table_{image_stats['image_id']:06d}.png"
    img_path = os.path.join(output_dir, subset, 'images', img_filename)
    img.save(img_path)

    label_filename = f"table_{image_stats['image_id']:06d}.txt"
    label_path = os.path.join(output_dir, subset, 'labels', label_filename)
    with open(label_path, 'w') as f:
        f.write('\n'.join(yolo_labels))

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
        'nc': 5,
        'names': ['cell', 'merged_cell', 'row', 'column', 'table']
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
    num_images = 20000
    imperfect_ratio = 0.1
    train_ratio = 0.8
    val_ratio = 0.1
    dataset_info = generate_large_dataset_in_batches(output_dir, num_images, imperfect_ratio=imperfect_ratio, train_ratio=train_ratio, val_ratio=val_ratio)
    print("Dataset generation completed.")