from grrr_common import generate_image_and_labels, generate_random_resolution, determine_subset
from grrr_utils import create_directory, calculate_summary_stats
import os
from tqdm import tqdm
import random
import itertools
import yaml
import csv
import json
def generate_yolo_labels(cells, table_bbox, image_width, image_height, rows, cols):
    yolo_labels = []
    
    # Cell labels
    for cell in cells:
        class_id = 1 if len(cell) > 6 else 0  # merged cell or normal cell
        yolo_labels.append(create_yolo_label(class_id, cell[0], cell[1], cell[2] - cell[0], cell[3] - cell[1], image_width, image_height))

    # Row labels
    for row in range(rows):
        row_cells = [cell for cell in cells if cell[4] == row]
        if row_cells:
            min_x = min(cell[0] for cell in row_cells)
            max_x = max(cell[2] for cell in row_cells)
            y = row_cells[0][1]
            height = cell_height = row_cells[0][3] - row_cells[0][1]
            yolo_labels.append(create_yolo_label(2, min_x, y, max_x - min_x, height, image_width, image_height))

    # Column labels
    for col in range(cols):
        col_cells = [cell for cell in cells if cell[5] == col]
        if col_cells:
            min_y = min(cell[1] for cell in col_cells)
            max_y = max(cell[3] for cell in col_cells)
            x = col_cells[0][0]
            width = col_cells[0][2] - col_cells[0][0]
            yolo_labels.append(create_yolo_label(3, x, min_y, width, max_y - min_y, image_width, image_height))

    # Table label
    yolo_labels.append(create_yolo_label(4, table_bbox[0], table_bbox[1], table_bbox[2] - table_bbox[0], table_bbox[3] - table_bbox[1], image_width, image_height))

    return yolo_labels

def create_yolo_label(class_id, x, y, width, height, image_width, image_height):
    x_center = (x + width / 2) / image_width
    y_center = (y + height / 2) / image_height
    width = width / image_width
    height = height / image_height
    return f"{class_id} {x_center:.4f} {y_center:.4f} {width:.4f} {height:.4f}"

def generate_large_dataset_yolo(output_dir, num_images, imperfect_ratio=0.3, train_ratio=0.8, val_ratio=0.1):
    create_directory(output_dir)
    for subset in ['train', 'val', 'test']:
        create_directory(os.path.join(output_dir, subset, 'images'))
        create_directory(os.path.join(output_dir, subset, 'labels'))
    
    dataset_info = {subset: [] for subset in ['train', 'val', 'test']}
    stats = []
    
    global_counter = itertools.count(1)
    
    for _ in tqdm(range(num_images), desc="Generating images"):
        img, cells, table_bbox, image_stats = generate_image_and_labels(
            next(global_counter), 
            generate_random_resolution(), 
            random.choice(['light', 'dark']), 
            random.choice([True, False]), 
            random.random() < imperfect_ratio
        )
        
        subset = determine_subset(train_ratio, val_ratio)
        img_filename = f"table_{image_stats['image_id']:06d}.png"
        img_path = os.path.join(output_dir, subset, 'images', img_filename)
        img.save(img_path)
        
        yolo_labels = generate_yolo_labels(cells, table_bbox, image_stats['image_width'], image_stats['image_height'], image_stats['rows'], image_stats['cols'])
        label_filename = f"table_{image_stats['image_id']:06d}.txt"
        label_path = os.path.join(output_dir, subset, 'labels', label_filename)
        with open(label_path, 'w') as f:
            f.write('\n'.join(yolo_labels))
        
        image_stats['subset'] = subset
        dataset_info[subset].append(image_stats)
        stats.append(image_stats)
    
    save_dataset_info(output_dir, dataset_info, stats)
    return dataset_info

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
    output_dir = 'table_dataset_yolo'
    num_images = 20000
    imperfect_ratio = 0.1
    train_ratio = 0.8
    val_ratio = 0.1
    generate_large_dataset_yolo(output_dir, num_images, imperfect_ratio, train_ratio, val_ratio)
    print("YOLO dataset generation completed.")
