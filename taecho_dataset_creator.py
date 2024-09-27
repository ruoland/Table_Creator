import cv2
import numpy as np
import random
import os
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import json
from PIL import Image, ImageDraw, ImageFont
import string
from dataset_constant import *

CELL_CATEGORY_ID = 0
TABLE_CATEGORY_ID = 1

def create_table(image_width, image_height, margins, title_height):
    margin_left, margin_top, margin_right, margin_bottom = margins
    table_width = image_width - margin_left - margin_right
    table_height = image_height - margin_top - margin_bottom - title_height

    cols = random.randint(MIN_COLS, min(MAX_COLS, table_width // MIN_CELL_WIDTH))
    rows = random.randint(MIN_ROWS, min(MAX_ROWS, table_height // MIN_CELL_HEIGHT))

    col_widths = [random.randint(MIN_CELL_WIDTH, MAX_CELL_WIDTH) for _ in range(cols)]
    row_heights = [random.randint(MIN_CELL_HEIGHT, MAX_CELL_HEIGHT) for _ in range(rows)]

    total_width = sum(col_widths)
    total_height = sum(row_heights)
    col_widths = [max(MIN_CELL_WIDTH, int(w * table_width / total_width)) for w in col_widths]
    row_heights = [max(MIN_CELL_HEIGHT, int(h * table_height / total_height)) for h in row_heights]


    cells = []
    y = margin_top + title_height
    for row in range(rows):
        x = margin_left
        for col in range(cols):
            cell = [x, y, x + col_widths[col], y + row_heights[row], row, col, False]
            cells.append(cell)
            x += col_widths[col]
        y += row_heights[row]

    # Merge some cells
    num_merges = int(len(cells) * MERGED_CELL_RATIO)
    for _ in range(num_merges):
        if len(cells) > 1:
            i = random.randint(0, len(cells) - 2)
            if random.choice([True, False]):  # Horizontal merge
                if cells[i][4] == cells[i+1][4]:  # Same row
                    cells[i][2] = cells[i+1][2]
                    cells[i][6] = True
                    cells.pop(i+1)
            else:  # Vertical merge
                if i + cols < len(cells):
                    cells[i][3] = cells[i+cols][3]
                    cells[i][6] = True
                    cells.pop(i+cols)

    table_bbox = [margin_left, margin_top + title_height,
                  margin_left + table_width,
                  margin_top + title_height + table_height]

    return cells, table_bbox, rows, cols

def draw_table(img, cells, table_bbox, bg_color, is_imperfect):
    line_color = (0, 0, 0) if bg_color == (255, 255, 255) else (255, 255, 255)
    
    # Draw outer border
    line_style = random.choice(LINE_STYLES)
    line_thickness = random.randint(1, 3)
    if line_style == 'solid':
        cv2.rectangle(img, (table_bbox[0], table_bbox[1]), (table_bbox[2], table_bbox[3]), line_color, line_thickness)
    elif line_style == 'dashed':
        draw_dashed_line(img, (table_bbox[0], table_bbox[1]), (table_bbox[2], table_bbox[1]), line_color, line_thickness)
        draw_dashed_line(img, (table_bbox[2], table_bbox[1]), (table_bbox[2], table_bbox[3]), line_color, line_thickness)
        draw_dashed_line(img, (table_bbox[2], table_bbox[3]), (table_bbox[0], table_bbox[3]), line_color, line_thickness)
        draw_dashed_line(img, (table_bbox[0], table_bbox[3]), (table_bbox[0], table_bbox[1]), line_color, line_thickness)
    elif line_style == 'dotted':
        draw_dotted_line(img, (table_bbox[0], table_bbox[1]), (table_bbox[2], table_bbox[1]), line_color, line_thickness)
        draw_dotted_line(img, (table_bbox[2], table_bbox[1]), (table_bbox[2], table_bbox[3]), line_color, line_thickness)
        draw_dotted_line(img, (table_bbox[2], table_bbox[3]), (table_bbox[0], table_bbox[3]), line_color, line_thickness)
        draw_dotted_line(img, (table_bbox[0], table_bbox[3]), (table_bbox[0], table_bbox[1]), line_color, line_thickness)

    for cell in cells:
        line_style = random.choice(LINE_STYLES)
        line_thickness = random.randint(1, 2)
        if line_style == 'solid':
            cv2.rectangle(img, (cell[0], cell[1]), (cell[2], cell[3]), line_color, line_thickness)
        elif line_style == 'dashed':
            draw_dashed_line(img, (cell[0], cell[1]), (cell[2], cell[1]), line_color, line_thickness)
            draw_dashed_line(img, (cell[2], cell[1]), (cell[2], cell[3]), line_color, line_thickness)
            draw_dashed_line(img, (cell[2], cell[3]), (cell[0], cell[3]), line_color, line_thickness)
            draw_dashed_line(img, (cell[0], cell[3]), (cell[0], cell[1]), line_color, line_thickness)
        elif line_style == 'dotted':
            draw_dotted_line(img, (cell[0], cell[1]), (cell[2], cell[1]), line_color, line_thickness)
            draw_dotted_line(img, (cell[2], cell[1]), (cell[2], cell[3]), line_color, line_thickness)
            draw_dotted_line(img, (cell[2], cell[3]), (cell[0], cell[3]), line_color, line_thickness)
            draw_dotted_line(img, (cell[0], cell[3]), (cell[0], cell[1]), line_color, line_thickness)
    if is_imperfect:
        noise = np.random.normal(0, 3, img.shape).astype(np.uint8)
        img = cv2.add(img, noise)
        img = np.clip(img, 0, 255).astype(np.uint8)
    

    return True
def draw_dashed_line(img, start, end, color, thickness):
    x1, y1 = start
    x2, y2 = end
    dash_length = 10
    gap_length = 5
    if x1 == x2:  # vertical line
        for y in range(y1, y2, dash_length + gap_length):
            cv2.line(img, (x1, y), (x1, min(y + dash_length, y2)), color, thickness)
    else:  # horizontal line
        for x in range(x1, x2, dash_length + gap_length):
            cv2.line(img, (x, y1), (min(x + dash_length, x2), y1), color, thickness)

def draw_dotted_line(img, start, end, color, thickness):
    x1, y1 = start
    x2, y2 = end
    gap_length = 5
    if x1 == x2:  # vertical line
        for y in range(y1, y2, gap_length):
            cv2.circle(img, (x1, y), thickness // 2, color, -1)
    else:  # horizontal line
        for x in range(x1, x2, gap_length):
            cv2.circle(img, (x, y1), thickness // 2, color, -1)
def add_text_to_cell(img, cell, text_color):
    cell_width, cell_height = cell[2] - cell[0], cell[3] - cell[1]
    padding = 2
    
    if cell_width < MIN_CELL_SIZE_FOR_CONTENT or cell_height < MIN_CELL_SIZE_FOR_CONTENT:
        return

    max_font_size = max(10, min(int(cell_height * 0.7), int(cell_width * 0.2)))
    min_font_size = min(10, max_font_size)
    font_size = random.randint(min_font_size, max_font_size)
    
    text = random.choice(COMMON_WORDS + DEPARTMENTS + SUBJECTS + PROFESSORS + BUILDINGS + ROOM_TYPES + TIMES + CLASS_TYPES + ACADEMIC_TERMS + EVENTS)
    
    pil_img = Image.fromarray(img.astype(np.uint8))
    draw = ImageDraw.Draw(pil_img)
    
    font = get_font(font_size)
    text_bbox = draw.textbbox((0, 0), text, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]
    
    # If text is too large for the cell, reduce font size
    while text_width > cell_width - 2*padding or text_height > cell_height - 2*padding:
        font_size = max(min_font_size, font_size - 1)
        if font_size == min_font_size:
            # If we've reached the minimum font size and text still doesn't fit, truncate the text
            text = text[:len(text)//2] + '...'
        font = get_font(font_size)
        text_bbox = draw.textbbox((0, 0), text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
    
    # Randomly position text within cell
    position = random.choice(TEXT_POSITIONS)
    if position == 'center':
        text_x = cell[0] + (cell_width - text_width) // 2
        text_y = cell[1] + (cell_height - text_height) // 2
    elif position == 'top_left':
        text_x = cell[0] + padding
        text_y = cell[1] + padding
    elif position == 'top_right':
        text_x = cell[2] - text_width - padding
        text_y = cell[1] + padding
    elif position == 'bottom_left':
        text_x = cell[0] + padding
        text_y = cell[3] - text_height - padding
    elif position == 'bottom_right':
        text_x = cell[2] - text_width - padding
        text_y = cell[3] - text_height - padding
    
    draw.text((text_x, text_y), text, font=font, fill=text_color)
    
    img[:] = np.array(pil_img)
from multiprocessing import Pool
def generate_dataset(output_dir, total_num_images, batch_size=100, imperfect_ratio=0.3, train_ratio=0.8, max_workers=None):
    os.makedirs(output_dir, exist_ok=True)
    all_coco_annotations = {'train': [], 'val': []}

    for subset in ['train', 'val']:
        subset_dir = os.path.join(output_dir, subset, 'images')
        os.makedirs(subset_dir, exist_ok=True)
        
        subset_total = int(total_num_images * (train_ratio if subset == 'train' else 1-train_ratio))
        batch_infos = []
        start_id = 1
        
        for batch_start in range(0, subset_total, batch_size):
            batch_end = min(batch_start + batch_size, subset_total)
            batch_infos.append({
                'output_dir': output_dir,
                'num_images': batch_end - batch_start,
                'imperfect_ratio': imperfect_ratio,
                'start_id': start_id,
                'subset': subset
            })
            start_id += batch_end - batch_start

        with Pool(processes=max_workers) as pool:
            results = list(tqdm(pool.imap(process_batch, batch_infos), total=len(batch_infos), desc=f"Processing {subset} dataset"))
        
        for batch_annotations in results:
            all_coco_annotations[subset].extend(batch_annotations)

    # COCO 형식으로 주석 저장
    for subset in ['train', 'val']:
        coco_output = {
            "images": [],
            "annotations": [],
            "categories": [{"id": 1, "name": "table_cell"}, {"id": 2, "name": "table"}]
        }
        
        annotation_id = 1
        for img_id, annotations in enumerate(all_coco_annotations[subset], start=1):
            coco_output["images"].append({
                "id": img_id,
                "file_name": f"{img_id:06d}.png",
                "height": annotations[0]["image_height"],
                "width": annotations[0]["image_width"]
            })
            for ann in annotations:
                ann["id"] = annotation_id
                ann["image_id"] = img_id
                coco_output["annotations"].append(ann)
                annotation_id += 1

        with open(os.path.join(output_dir, f'{subset}_annotations.json'), 'w') as f:
            json.dump(coco_output, f)

    print("Dataset generation completed.")
    print(f"Train files generated: {len(all_coco_annotations['train'])}")
    print(f"Val files generated: {len(all_coco_annotations['val'])}")


def generate_coco_annotations(cells, table_bbox, image_id):
    coco_annotations = []
    annotation_id = 1

    for cell in cells:
        x1, y1, x2, y2 = cell[:4]
        width = x2 - x1
        height = y2 - y1

        cell_annotation = {
            "id": annotation_id,
            "image_id": image_id,
            "category_id": CELL_CATEGORY_ID,
            "bbox": [x1, y1, width, height],
            "area": width * height,
            "iscrowd": 0,
        }
        coco_annotations.append(cell_annotation)
        annotation_id += 1
    
    table_width = table_bbox[2] - table_bbox[0]
    table_height = table_bbox[3] - table_bbox[1]
    table_annotation = {
        "id": annotation_id,
        "image_id": image_id,
        "category_id": TABLE_CATEGORY_ID,
        "bbox": [table_bbox[0], table_bbox[1], table_width, table_height],
        "area": table_width * table_height,
        "iscrowd": 0,
    }
    coco_annotations.append(table_annotation)

    return coco_annotations
def generate_image_and_labels(image_id, width, height, bg_mode, is_imperfect):
    print(f"Starting generation of image {image_id}")
    
    # 타임아웃 설정 (60초)
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(60)
    
    try:
        print(f"Setting background for image {image_id}")
        bg_color = BACKGROUND_COLORS['light']['white'] if bg_mode == 'light' else BACKGROUND_COLORS['dark']['black']
        img = np.full((height, width, 3), bg_color, dtype=np.uint8)
        
        print(f"Creating table for image {image_id}")
        margins = (random.randint(MIN_MARGIN, MAX_MARGIN), random.randint(MIN_MARGIN, MAX_MARGIN),
                   random.randint(MIN_MARGIN, MAX_MARGIN), random.randint(MIN_MARGIN, MAX_MARGIN))
        title_height = random.randint(MIN_TITLE_SIZE, MAX_TITLE_SIZE)
        cells, table_bbox, rows, cols = create_table(width, height, margins, title_height)
        
        print(f"Drawing table for image {image_id}")
        draw_table(img, cells, table_bbox, bg_color, is_imperfect)
        
        print(f"Adding text to cells for image {image_id}")
        text_color = (0, 0, 0) if bg_mode == 'light' else (255, 255, 255)
        for cell in cells:
            if random.random() > EMPTY_CELL_RATIO:
                add_text_to_cell(img, cell, text_color)
        
        print(f"Adding title for image {image_id}")
        title = generate_random_title().format(random.choice(COMMON_WORDS + DEPARTMENTS + SUBJECTS))
        pil_img = Image.fromarray(img)
        draw = ImageDraw.Draw(pil_img)
        title_font = get_font(title_height)
        draw.text((margins[0], margins[1]), title, font=title_font, fill=text_color)
        img = np.array(pil_img)
        
        print(f"Generating annotations for image {image_id}")
        coco_annotations = generate_coco_annotations(cells, table_bbox, image_id)
        
        print(f"Completed generation of image {image_id}")
        return img, coco_annotations
    
    except TimeoutException:
        print(f"Timeout occurred while generating image {image_id}")
        return None, None
    except Exception as e:
        print(f"Error occurred while generating image {image_id}: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return None, None
    finally:
        signal.alarm(0)

current_dir = os.path.dirname(os.path.abspath(__file__))

def get_font(font_size):
    font_path = os.path.join(current_dir, random.choice(FONTS))
    try:
        return ImageFont.truetype(font_path, font_size)
    except OSError:
        print(f"Warning: Could not load font from {font_path}. Using default font.")
        return ImageFont.load_default()

def generate_random_resolution():
    width = random.randint(800, 1200)
    height = random.randint(600, 1000)
    margins = (random.randint(10, 50), random.randint(10, 50), random.randint(10, 50), random.randint(10, 50))
    return (width, height), margins

def process_batch(batch_info):
    batch_coco_annotations = []
    images = []
    output_dir = batch_info['output_dir']
    
    for i in range(batch_info['num_images']):
        image_id = batch_info['start_id'] + i
        width, height = random.randint(MIN_IMAGE_WIDTH, MAX_IMAGE_WIDTH), random.randint(MIN_IMAGE_HEIGHT, MAX_IMAGE_HEIGHT)
        bg_mode = random.choice(['light', 'dark'])
        is_imperfect = random.random() < batch_info['imperfect_ratio']
        
        try:
            img, annotations = generate_image_and_labels(image_id, width, height, bg_mode, is_imperfect)
            
            if img is not None and annotations is not None:
                img_filename = f"{image_id:06d}.png"
                img_path = os.path.join(output_dir, batch_info['subset'], 'images', img_filename)
                cv2.imwrite(img_path, img)
                images.append((img_filename, img))
                batch_coco_annotations.extend(annotations)
                print(f"Generated image {image_id}")
            else:
                print(f"Failed to generate image {image_id}")
        except Exception as e:
            print(f"Error processing image {image_id}: {str(e)}")

    return batch_coco_annotations
import random
import time
import signal

# 랜덤 시드 고정
random.seed(42)

class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutException("Function call timed out")


def batch_dataset(output_dir, total_num_images, batch_size=1000, imperfect_ratio=0.3, train_ratio=0.8, max_workers=None):
    os.makedirs(output_dir, exist_ok=True)
    for subset in ['train', 'val']:
        os.makedirs(os.path.join(output_dir, subset, 'images'), exist_ok=True)

    if max_workers is None:
        max_workers = multiprocessing.cpu_count()

    all_coco_annotations = {subset: [] for subset in ['train', 'val']}

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for subset in ['train', 'val']:
            subset_total = int(total_num_images * (train_ratio if subset == 'train' else 1-train_ratio))
            futures = []
            for batch_start in range(0, subset_total, batch_size):
                batch_end = min(batch_start + batch_size, subset_total)
                batch_info = {
                    'output_dir': output_dir,
                    'num_images': batch_end - batch_start,
                    'imperfect_ratio': imperfect_ratio,
                    'start_id': batch_start + 1,
                    'subset': subset
                }
                future = executor.submit(process_batch, batch_info, output_dir)
                futures.append(future)

            for future in tqdm(futures, desc=f"Processing {subset} dataset"):
                try:
                    batch_coco_annotations = future.result()
                    all_coco_annotations[subset].extend(batch_coco_annotations)
                except Exception as e:
                    print(f"Error processing batch: {str(e)}")
                    import traceback
                    print(traceback.format_exc())
    for subset in ['train', 'val']:
        with open(os.path.join(output_dir, f"{subset}_coco_annotations.json"), 'w') as f:
            json.dump({
                "images": [{"id": ann["image_id"], "file_name": f"{ann['image_id']:06d}.png"} for ann in all_coco_annotations[subset] if ann["category_id"] == TABLE_CATEGORY_ID],
                "annotations": all_coco_annotations[subset],
                "categories": [
                    {"id": CELL_CATEGORY_ID, "name": "cell"},
                    {"id": TABLE_CATEGORY_ID, "name": "table"}
                ]
            }, f)

    print("Dataset generation completed.")
    for subset in ['train', 'val']:
        files = len([f for f in os.listdir(os.path.join(output_dir, subset, 'images')) if f.endswith('.png')])
        print(f"{subset.capitalize()} files generated: {files}")

if __name__ == "__main__":
    output_dir = "table_dataset"
    num_images = 1000
    batch_dataset(output_dir, num_images)
