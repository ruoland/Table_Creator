import os
import random, sys
from tqdm import tqdm
from dataset_io import save_subset_results
import numpy as np
import logging
from PIL import Image, ImageDraw
from dataset_draw_preprocess import apply_realistic_effects
from dataset_draw import add_content_to_cells, add_title_to_image, validate_cell, draw_table, add_shapes, apply_imperfections
from dataset_constant import BACKGROUND_COLORS, FONTS, MIN_COLS, MAX_COLS, MIN_CELL_HEIGHT, MIN_CELL_WIDTH, MIN_ROWS, MAX_ROWS, MAX_CELL_HEIGHT, MAX_CELL_WIDTH
from dataset_utils import count_existing_images, generate_random_resolution, is_overlapping

import cProfile
from dataset_config import TableGenerationConfig, logging as logger
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
CELL_CATEGORY_ID = 0
TABLE_CATEGORY_ID = 1

def create_table(image_width, image_height, margins, title_height, config):
    logger.debug(f"create_table 시작: 이미지 크기 {image_width}x{image_height}")
    
    # 모든 입력값을 정수로 변환
    margin_left, margin_top, margin_right, margin_bottom = map(int, margins)
    title_height = int(title_height)
    
    # 갭 설정
    gap = random.randint(config.min_cell_gap, config.max_cell_gap) if config.enable_cell_gap else 0
    
    # 테이블 크기 계산 (갭 포함)
    table_width = max(config.min_table_width, image_width - margin_left - margin_right)
    table_height = max(config.min_table_height, image_height - margin_top - margin_bottom - title_height)

    # 행과 열 수 계산 (갭 고려), 최소 2개의 셀 보장
    cols = max(2, min(config.max_cols, (table_width + gap) // (config.min_cell_width + gap)))
    rows = max(2, min(config.max_rows, (table_height + gap) // (config.min_cell_height + gap)))
    
    # 만약 행이나 열이 1인 경우, 다른 쪽을 2로 설정
    if rows == 1:
        cols = max(2, cols)
    elif cols == 1:
        rows = max(2, rows)
    
    logger.debug(f"테이블 크기: {table_width}x{table_height}, 행 수: {rows}, 열 수: {cols}, 갭: {gap}")

    # 셀 크기 계산 (갭 제외)
    available_width = table_width - (cols - 1) * gap
    available_height = table_height - (rows - 1) * gap
    min_cell_width = max(1, available_width // cols)
    min_cell_height = max(1, available_height // rows)

    col_widths = [min_cell_width] * cols
    row_heights = [min_cell_height] * rows

    # 남은 공간 분배
    extra_width = available_width - sum(col_widths)
    extra_height = available_height - sum(row_heights)

    for i in range(extra_width):
        col_widths[i % cols] += 1
    for i in range(extra_height):
        row_heights[i % rows] += 1

    # 셀 생성
    cells = []
    y = margin_top + title_height

    for row in range(rows):
        x = margin_left
        for col in range(cols):
            is_header = (config.table_type in ['header_row', 'header_both'] and row == 0) or \
                        (config.table_type in ['header_column', 'header_both'] and col == 0)

            cell = [
                x,
                y,
                x + col_widths[col],
                y + row_heights[row],
                row, col, False, is_header,
                y + row_heights[row]  # 원래 높이
            ]
            cells.append(cell)
            logger.debug(f"셀 생성: {cell}")
            x += col_widths[col] + gap
        y += row_heights[row] + gap

    # 셀 병합 (옵션)
    if config.enable_cell_merging and len(cells) > 2:
        cells = merge_cells(cells, rows, cols, config)

    # 테이블의 전체 경계 상자 계산
    table_bbox = [
        margin_left,
        margin_top + title_height,
        margin_left + table_width,
        margin_top + title_height + table_height
    ]
    
    # 최종 검증: 셀이 2개 미만인 경우 처리
    if len(cells) < 2:
        logger.warning("셀 수가 2개 미만입니다. 테이블을 2x1로 재구성합니다.")
        cells = [
            [margin_left, margin_top + title_height, margin_left + table_width // 2, margin_top + title_height + table_height, 0, 0, False, False, margin_top + title_height + table_height],
            [margin_left + table_width // 2, margin_top + title_height, margin_left + table_width, margin_top + title_height + table_height, 0, 1, False, False, margin_top + title_height + table_height]
        ]
    
    logger.debug(f"create_table 종료: 생성된 셀 수 {len(cells)}")
    return cells, table_bbox

def merge_cells(cells, rows, cols, config):
    logger.debug(f"merge_cells 시작: 행 {rows}, 열 {cols}")
    if rows < 4 or cols < 3:
        logger.debug("테이블이 너무 작아 병합을 수행하지 않습니다.")
        return cells
    merged_cells = cells.copy()
    num_merges = random.randint(max(1, rows * cols // 10), max(3, rows * cols // 5))
    merged_areas = []

    for _ in range(num_merges):
        attempts = 0
        while attempts < 20:
            start_row = random.randint(0, rows - 2)
            start_col = random.randint(0, cols - 2)
            
            # 헤더 셀은 병합하지 않음
            if cells[start_row * cols + start_col][7]:  # is_header
                continue

            merge_rows = random.randint(2, min(4, rows - start_row))
            merge_cols = random.randint(2, min(4, cols - start_col))
            
            merge_area = (start_row, start_col, start_row + merge_rows, start_col + merge_cols)
            
            if not any(is_overlapping(merge_area, area) for area in merged_areas):
                new_cell_info = [
                    cells[start_row * cols + start_col][0],
                    cells[start_row * cols + start_col][1],
                    max(cells[(start_row + merge_rows - 1) * cols + (start_col + merge_cols - 1)][2],
                        cells[start_row * cols + start_col][0] + config.min_cell_width),
                    max(cells[(start_row + merge_rows - 1) * cols + (start_col + merge_cols - 1)][3],
                        cells[start_row * cols + start_col][1] + config.min_cell_height),
                    start_row, start_col, True,
                    False,  # is_header
                    start_row, start_col,
                    start_row + merge_rows - 1, start_col + merge_cols - 1
                ]
                # 병합된 셀의 높이와 너비 확인
                if (new_cell_info[2] - new_cell_info[0] < config.min_cell_width or 
                    new_cell_info[3] - new_cell_info[1] < config.min_cell_height):
                    continue  # 최소 크기를 만족하지 않으면 이 병합을 건너뜁니다
                for r in range(start_row, start_row + merge_rows):
                    for c in range(start_col, start_col + merge_cols):
                        merged_cells[r * cols + c] = None
                
                merged_cells[start_row * cols + start_col] = new_cell_info
                merged_areas.append(merge_area)
                break
            
            attempts += 1
    cells = [cell for cell in cells if validate_cell(cell)]
    logger.debug(f"merge_cells 종료: 병합된 셀 수 {len([cell for cell in merged_cells if cell is not None])}")
    return [cell for cell in merged_cells if cell is not None and validate_cell(cell)]
import cv2
def generate_coco_annotations(cells, table_bbox, image_id, image_width, image_height, transform_matrix):
    coco_annotations = []
    annotation_id = 1

    def transform_coordinates(coords):
        pts = np.array([[coords[0], coords[1]], [coords[2], coords[1]], 
                        [coords[2], coords[3]], [coords[0], coords[3]]], dtype=np.float32)
        transformed_pts = cv2.perspectiveTransform(pts.reshape(-1, 1, 2), transform_matrix).reshape(-1, 2)
        return [np.min(transformed_pts[:, 0]), np.min(transformed_pts[:, 1]),
                np.max(transformed_pts[:, 0]), np.max(transformed_pts[:, 1])]

    for cell_info in cells:
        original_coords = cell_info[:4]
        transformed_coords = transform_coordinates(original_coords)

        x1, y1, x2, y2 = transformed_coords

        # 이미지 경계로 셀 좌표 클리핑
        x1 = max(0, min(x1, image_width))
        y1 = max(0, min(y1, image_height))
        x2 = max(0, min(x2, image_width))
        y2 = max(0, min(y2, image_height))

        width = max(0, x2 - x1)
        height = max(0, y2 - y1)

        if width > 0 and height > 0:
            cell_annotation = {
                "id": annotation_id,
                "image_id": image_id,
                "category_id": CELL_CATEGORY_ID,
                "bbox": [x1, y1, width, height],
                "area": width * height,
                "iscrowd": 0,
                "category_name": "cell"
            }
            coco_annotations.append(cell_annotation)
            annotation_id += 1

    # 테이블 어노테이션 생성 (변환 적용)
    transformed_table_bbox = transform_coordinates(table_bbox)
    table_x1, table_y1, table_x2, table_y2 = transformed_table_bbox

    # 이미지 경계로 테이블 좌표 클리핑
    table_x1 = max(0, min(table_x1, image_width))
    table_y1 = max(0, min(table_y1, image_height))
    table_x2 = max(0, min(table_x2, image_width))
    table_y2 = max(0, min(table_y2, image_height))

    table_width = max(0, table_x2 - table_x1)
    table_height = max(0, table_y2 - table_y1)

    if table_width > 0 and table_height > 0:
        table_annotation = {
            "id": annotation_id,
            "image_id": image_id,
            "category_id": TABLE_CATEGORY_ID,
            "bbox": [table_x1, table_y1, table_width, table_height],
            "area": table_width * table_height,
            "iscrowd": 0,
            "category_name": "table"
        }
        coco_annotations.append(table_annotation)

    return coco_annotations

def generate_image_and_labels(image_id, resolution, margins, bg_mode, has_gap, is_imperfect=False, config=None):
    logger.debug(f"generate_image_and_labels 시작: 이미지 ID {image_id}")
    try:
        image_width, image_height = resolution
        
        bg_color = BACKGROUND_COLORS['light']['white'] if bg_mode == 'light' else BACKGROUND_COLORS['dark']['black']
        
        img = Image.new('RGB', (image_width, image_height), color=bg_color)
        
        if config.enable_title:
            title_height = add_title_to_image(img, image_width, image_height, margins[1], bg_color)
        else:
            title_height = 0
        
        cells, table_bbox = create_table(image_width, image_height, margins, title_height, config=config)
        
        if config.enable_background_shapes:
            max_height = max(title_height, table_bbox[1])
            add_shapes(img, 0, title_height, image_width, max_height, bg_color)
        
        draw = ImageDraw.Draw(img)
        draw_table(draw, cells, table_bbox, bg_color, has_gap, is_imperfect)
        
        if config.enable_text_generation or config.enable_shapes:
            add_content_to_cells(img, cells, random.choice(FONTS), bg_color)
        
        if is_imperfect:
            img = apply_imperfections(img, cells)
        
        img, cells, table_bbox, transform_matrix, new_width, new_height = apply_realistic_effects(img, cells, table_bbox, title_height, config)
    
        coco_annotations = generate_coco_annotations(cells, table_bbox, image_id, new_width, new_height, transform_matrix)
        
        return img, coco_annotations, new_width, new_height
    except Exception as e:
        logger.error(f"Error generating image {image_id}: {str(e)}")
        logger.error(f"info: Error occurred in generate_image_and_labels - {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return None, None, None, None

def apply_table_cropping(img, cells, table_bbox, max_crop_ratio):
    width, height = img.size
    crop_direction = random.choice(['left', 'right', 'top', 'bottom'])
    crop_amount = int(random.uniform(0, max_crop_ratio) * min(width, height))

    if crop_direction == 'left':
        new_img = img.crop((crop_amount, 0, width, height))
        cells = [[max(0, c[0] - crop_amount), c[1], max(0, c[2] - crop_amount), c[3]] + c[4:] for c in cells]
        table_bbox = [max(0, table_bbox[0] - crop_amount), table_bbox[1], table_bbox[2] - crop_amount, table_bbox[3]]
    elif crop_direction == 'right':
        new_img = img.crop((0, 0, width - crop_amount, height))
        cells = [[c[0], c[1], min(width - crop_amount, c[2]), c[3]] + c[4:] for c in cells]
        table_bbox = [table_bbox[0], table_bbox[1], min(width - crop_amount, table_bbox[2]), table_bbox[3]]
    elif crop_direction == 'top':
        new_img = img.crop((0, crop_amount, width, height))
        cells = [[c[0], max(0, c[1] - crop_amount), c[2], max(0, c[3] - crop_amount)] + c[4:] for c in cells]
        table_bbox = [table_bbox[0], max(0, table_bbox[1] - crop_amount), table_bbox[2], table_bbox[3] - crop_amount]
    else:  # bottom
        new_img = img.crop((0, 0, width, height - crop_amount))
        cells = [[c[0], c[1], c[2], min(height - crop_amount, c[3])] + c[4:] for c in cells]
        table_bbox = [table_bbox[0], table_bbox[1], table_bbox[2], min(height - crop_amount, table_bbox[3])]

    # 유효하지 않은 셀 제거
    cells = [cell for cell in cells if cell[2] > cell[0] and cell[3] > cell[1]]

    return new_img, cells, table_bbox

def batch_dataset(output_dir, total_num_images, imperfect_ratio=0.3, train_ratio=0.8, num_processes=None, config=None):
    logger.info(f"batch_dataset 시작: 총 이미지 수 {total_num_images}")
    
    if num_processes is None:
        num_processes = cpu_count()

    os.makedirs(output_dir, exist_ok=True)
    for subset in ['train', 'val']:
        os.makedirs(os.path.join(output_dir, subset, 'images'), exist_ok=True)

    tasks = []
    train_images = int(total_num_images * train_ratio)
    val_images = total_num_images - train_images

    # Train 작업 생성
    for i in range(0, train_images, train_images // num_processes):

        start_id = i
        end_id = min(i + train_images // num_processes, train_images)
        tasks.append((start_id, end_id, output_dir, 'train', imperfect_ratio, config))

    # Val 작업 생성
    for i in range(0, val_images, val_images // num_processes):

        start_id = train_images + i
        end_id = min(train_images + i + val_images // num_processes, total_num_images)
        tasks.append((start_id, end_id, output_dir, 'val', imperfect_ratio, config))

    with Pool(processes=num_processes) as pool:
        results = list(tqdm(pool.imap(process_images, tasks), total=len(tasks), desc="Processing images"))

    all_dataset_info = {'train': [], 'val': []}
    all_coco_annotations = {'train': [], 'val': []}
    
    for dataset_info, coco_annotations in results:
        subset = dataset_info[0]['subset'] if dataset_info else 'train'  # 기본값 설정
        all_dataset_info[subset].extend(dataset_info)
        all_coco_annotations[subset].extend(coco_annotations)

    for subset in ['train', 'val']:
        save_subset_results(output_dir, subset, all_dataset_info[subset], all_coco_annotations[subset])

    logger.info("Dataset generation completed.")
    for subset in ['train', 'val']:
        files = len([f for f in os.listdir(os.path.join(output_dir, subset, 'images')) if f.endswith('.png')])
        logger.info(f"{subset.capitalize()} files generated: {files}")

    return all_dataset_info, all_coco_annotations
def process_images(args):
    start_id, end_id, output_dir, subset, imperfect_ratio, config = args
    logger.info(f"process_images 시작: {start_id} ~ {end_id} for {subset}")

    dataset_info = []
    coco_annotations = []

    for image_id in range(start_id, end_id):
        config.randomize_settings()
        config.enable_table_cropping = False
        
        logger.debug(f"이미지 생성 시작: ID {image_id} for {subset}")
        resolution, margins = generate_random_resolution()
        img, annotations, actual_width, actual_height = generate_image_and_labels(
            image_id, resolution, margins,
            random.choice(['light', 'dark']),
            random.choice([True, False]),
            random.random() < imperfect_ratio,
            config
        )
        
        if img is not None:
            img_path = os.path.join(output_dir, subset, 'images', f"{image_id:06d}.png")
            img.save(img_path)
            
            # 이미지 크기와 어노테이션 비교
            for ann in annotations:
                bbox = ann['bbox']
                if bbox[0] < 0 or bbox[1] < 0 or bbox[0] + bbox[2] > actual_width or bbox[1] + bbox[3] > actual_height:
                    logger.warning(f"Annotation out of bounds for image {image_id}: {bbox}")
        
            image_info = {
                'image_id': image_id,
                'image_width': actual_width,
                'image_height': actual_height,
                'subset': subset
            }
            dataset_info.append(image_info)
            coco_annotations.extend(annotations)
            logger.debug(f"이미지 저장 완료: {img_path}")
        else:
            logger.warning(f"이미지 생성 실패: ID {image_id} for {subset}")

    logger.info(f"process_images 종료: {start_id} ~ {end_id} for {subset}")            
    return dataset_info, coco_annotations

def main():
    output_dir = r'C:\project\table2'
    num_images = 300
    imperfect_ratio = 0.3
    train_ratio = 0.8
    num_processes = cpu_count()

    config = TableGenerationConfig()

    
    # 수동 설정 적용
    
    all_dataset_info, all_coco_annotations = batch_dataset(
        output_dir=output_dir, 
        total_num_images=num_images,
        imperfect_ratio=imperfect_ratio, 
        train_ratio=train_ratio,
        num_processes=num_processes,
        config=config
    )

    logger.info("프로그램 실행 완료")

if __name__ == "__main__":
    main()
