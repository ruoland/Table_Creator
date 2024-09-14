import os
import random
from tqdm import tqdm
import numpy as np
import logging
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from concurrent.futures import ProcessPoolExecutor
import yaml
from opti_calc import *
from opti_draw import *
from opti_stats import *
from opti_constants import *
from opti_utils import *
import itertools

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_table(image_width, image_height, has_gap=False):
    cols = random.randint(MIN_COLS, MAX_COLS)
    rows = random.randint(MIN_ROWS, MAX_ROWS)
    
    col_ratios = random.choices(COLUMN_WIDTH_RATIOS, k=cols)
    row_ratios = random.choices(ROW_HEIGHT_RATIOS, k=rows)
    
    total_col_ratio = sum(col_ratios)
    total_row_ratio = sum(row_ratios)
    
    base_cell_width = image_width / total_col_ratio
    base_cell_height = image_height / total_row_ratio
    
    gap = random.randint(1, 3) if has_gap else 0
    
    cells = [
        [
            sum(col_ratios[:col]) * base_cell_width + (gap if has_gap else 0),
            sum(row_ratios[:row]) * base_cell_height + (gap if has_gap else 0),
            sum(col_ratios[:col+1]) * base_cell_width - (gap if has_gap else 0),
            sum(row_ratios[:row+1]) * base_cell_height - (gap if has_gap else 0),
            row, col
        ]
        for row in range(rows)
        for col in range(cols)
    ]
    
    return merge_cells(cells, rows, cols), [0, 0, image_width, image_height], gap, rows, cols

def merge_cells(cells, rows, cols):
    merged_cells = cells.copy()
    num_merges = int(len(cells) * CELL_MERGE_PROBABILITY)
    merged_areas = []

    for _ in range(num_merges):
        attempts = 0
        while attempts < 10:
            start_row = random.randint(0, rows - 2)
            start_col = random.randint(0, cols - 2)
            merge_rows = random.randint(2, min(MAX_MERGE_SIZE[0], rows - start_row))
            merge_cols = random.randint(2, min(MAX_MERGE_SIZE[1], cols - start_col))
            
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

    # Row labels
    for row in range(rows):
        y = (table_bbox[1] + row * (table_bbox[3] - table_bbox[1]) / rows) / image_height
        height = (table_bbox[3] - table_bbox[1]) / (rows * image_height)
        yolo_labels.append(create_yolo_label(2, table_bbox[0] / image_width, y, 
                                             (table_bbox[2] - table_bbox[0]) / image_width, height))

    # Column labels
    for col in range(cols):
        x = (table_bbox[0] + col * (table_bbox[2] - table_bbox[0]) / cols) / image_width
        width = (table_bbox[2] - table_bbox[0]) / (cols * image_width)
        yolo_labels.append(create_yolo_label(3, x, table_bbox[1] / image_height, 
                                             width, (table_bbox[3] - table_bbox[1]) / image_height))

    # Cell labels
    for cell in cells:
        class_id = 1 if len(cell) > 6 else 0  # merged cell or normal cell
        x_center = (cell[0] + cell[2]) / (2 * image_width)
        y_center = (cell[1] + cell[3]) / (2 * image_height)
        width = (cell[2] - cell[0]) / image_width
        height = (cell[3] - cell[1]) / image_height
        yolo_labels.append(create_yolo_label(class_id, x_center, y_center, width, height))

    # Table label
    yolo_labels.append(create_yolo_label(4, 
                                         (table_bbox[0] + table_bbox[2]) / (2 * image_width),
                                         (table_bbox[1] + table_bbox[3]) / (2 * image_height),
                                         (table_bbox[2] - table_bbox[0]) / image_width,
                                         (table_bbox[3] - table_bbox[1]) / image_height))

    return yolo_labels

def create_yolo_label(class_id, x_center, y_center, width, height):
    return f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
def generate_image_and_labels(args):
    image_id, resolution, bg_mode, has_gap, is_imperfect = args
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
    except Exception as e:
        logger.error(f"Error generating image {image_id}: {str(e)}")
        logger.exception("Exception details:")
        return None, None, None


import gc
def generate_large_dataset(output_dir, num_images, imperfect_ratio=0.3, train_ratio=0.8, val_ratio=0.1, global_counter=None):
    create_directory(output_dir)
    logger.info(num_images)
    for subset in ['train', 'val', 'test']:
        create_directory(os.path.join(output_dir, subset, 'images'))
        create_directory(os.path.join(output_dir, subset, 'labels'))
    
    dataset_info = {subset: [] for subset in ['train', 'val', 'test']}
    stats = []

    for _ in tqdm(range(num_images), desc="Generating images"):
        try:
            image_id = next(global_counter)
            
            result = generate_image_and_labels(
                image_id,
                generate_random_resolution(),
                random.choice(['light', 'dark']),
                random.choice([True, False]),
                random.random() < imperfect_ratio
            )
            
            if result[0] is not None:
                img, yolo_labels, image_stats = result
                subset = determine_subset(train_ratio, val_ratio)
                image_stats['subset'] = subset
                save_image_and_labels(img, yolo_labels, image_stats, output_dir, subset)
                dataset_info[subset].append(image_stats)
                stats.append(image_stats)
            else:
                logger.warning(f"Failed to generate image {image_id}, skipping...")
        except Exception as e:
            logger.error(f"Error processing image: {str(e)}")
            logger.exception("Exception details:")

    logger.info(f"Generated {len(stats)} images")
    return dataset_info, stats

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

def generate_large_dataset_in_batches(output_dir, total_num_images, batch_size=1000, imperfect_ratio=0.1, train_ratio=0.8, val_ratio=0.1):
    last_id = load_last_image_id(output_dir)
    if last_id >= total_num_images:
        logger.warning(f"Last ID ({last_id}) is greater than or equal to total_num_images ({total_num_images}). No new images will be generated.")
        return

    global_counter = itertools.count(last_id + 1)
    all_dataset_info = {subset: [] for subset in ['train', 'val', 'test']}
    all_stats = []

    with ProcessPoolExecutor() as executor:
        for start in range(last_id, total_num_images, batch_size):
            end = min(start + batch_size, total_num_images)
            batch_args = [
                (next(global_counter), generate_random_resolution(), 
                 random.choice(['light', 'dark']), random.choice([True, False]), 
                 random.random() < imperfect_ratio)
                for _ in range(end - start)
            ]
            
            results = list(tqdm(executor.map(generate_image_and_labels, batch_args), total=len(batch_args), desc="Generating images"))
            
            for img, yolo_labels, image_stats in results:
                if img is not None:
                    subset = determine_subset(train_ratio, val_ratio)
                    image_stats['subset'] = subset
                    save_image_and_labels(img, yolo_labels, image_stats, output_dir, subset)
                    all_dataset_info[subset].append(image_stats)
                    all_stats.append(image_stats)

    save_last_image_id(output_dir, next(global_counter) - 1)
    if all_stats:
        save_dataset_info(output_dir, all_dataset_info, all_stats)
        
        total_images = len(all_stats)
        merged_cell_ratio = sum(s['num_merged_cells'] for s in all_stats) / sum(s['num_cells'] for s in all_stats)
        imperfect_ratio = sum(1 for s in all_stats if s['is_imperfect']) / total_images
        
        logger.info(f"총 생성된 이미지 수: {total_images}")
        logger.info(f"병합된 셀 비율: {merged_cell_ratio:.2%}")
        logger.info(f"불완전한 이미지 비율: {imperfect_ratio:.2%}")
        logger.info(f"빈 셀 비율: {EMPTY_CELL_RATIO:.2%}")
        logger.info(f"도형 생성 비율: {SHAPE_GENERATION_RATIO:.2%}")
    else:
        logger.warning("No stats were generated. Check your data generation process.")
    
    return all_dataset_info

if __name__ == "__main__":
    output_dir = 'table_dataset_opti2'
    num_images = 13000
    
    dataset_info = generate_large_dataset_in_batches(output_dir, num_images, BATCH_SIZE)
    print("Dataset generation completed.")