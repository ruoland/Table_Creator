import os, math, random
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import json
from tqdm import tqdm
from typing import List, Tuple

from new_constants import *
from new_cell_content import random_text, wrap_text
from new_draw import get_line_color

def create_table_image(width: int = None, height: int = None) -> Tuple[Image.Image, ImageDraw.Draw, List[List[int]]]:
    """테이블 이미지를 생성합니다."""
    if width is None:
        width = random.randint(MIN_IMAGE_WIDTH, MAX_IMAGE_WIDTH)
    if height is None:
        height = random.randint(MIN_IMAGE_HEIGHT, MAX_IMAGE_HEIGHT)
    
    rows = random.randint(MIN_ROWS, MAX_ROWS)
    cols = random.randint(MIN_COLS, MAX_COLS)
    
    bg_mode = random.choice(['light', 'dark'])
    bg_color = random.choice(list(BACKGROUND_COLORS[bg_mode].values()))
    
    img = Image.new('RGB', (width, height), color=bg_color)
    draw = ImageDraw.Draw(img)
    
    cell_width = width // cols
    cell_height = height // rows
    
    cells = []
    for row in range(rows):
        for col in range(cols):
            x1 = col * cell_width
            y1 = row * cell_height
            x2 = x1 + cell_width
            y2 = y1 + cell_height
            cells.append([x1, y1, x2, y2])
    
    cells = merge_cells(cells, rows, cols)
    draw_table(draw, cells, bg_color)
    
    return img, draw, cells

def merge_cells(cells: List[List[int]], rows: int, cols: int) -> List[List[int]]:
    """셀을 병합합니다."""
    num_merges = int(len(cells) * MERGED_CELL_RATIO)
    merged_cells = cells.copy()
    
    for _ in range(num_merges):
        if len(merged_cells) <= 1:
            break
        
        i = random.randint(0, len(merged_cells) - 2)
        cell1 = merged_cells[i]
        cell2 = merged_cells[i + 1]
        
        if cell1[2] == cell2[0] and cell1[1] == cell2[1] and cell1[3] == cell2[3]:  # 가로로 인접한 셀
            merged_cell = [cell1[0], cell1[1], cell2[2], cell1[3]]
            merged_cells[i] = merged_cell
            merged_cells.pop(i + 1)
    
    return merged_cells

def draw_table(draw: ImageDraw.Draw, cells: List[List[int]], bg_color: Tuple[int, int, int]):
    """테이블을 그립니다."""
    for cell in cells:
        line_color = get_line_color(bg_color)
        line_width = random.randint(MIN_LINE_WIDTH, MAX_LINE_WIDTH)
        line_style = random.choice(LINE_STYLES)
        
        if line_style == 'solid':
            draw.rectangle(cell, outline=line_color, width=line_width)
        elif line_style == 'dashed':
            draw_dashed_rectangle(draw, cell, line_color, line_width)
        elif line_style == 'dotted':
            draw_dotted_rectangle(draw, cell, line_color, line_width)

def draw_dashed_rectangle(draw: ImageDraw.Draw, cell: List[int], color: Tuple[int, int, int], width: int):
    """점선 사각형을 그립니다."""
    x0, y0, x1, y1 = cell
    dash_length = 5
    
    for x in range(x0, x1, dash_length * 2):
        draw.line((x, y0, min(x + dash_length, x1), y0), fill=color, width=width)
        draw.line((x, y1, min(x + dash_length, x1), y1), fill=color, width=width)
    
    for y in range(y0, y1, dash_length * 2):
        draw.line((x0, y, x0, min(y + dash_length, y1)), fill=color, width=width)
        draw.line((x1, y, x1, min(y + dash_length, y1)), fill=color, width=width)

def draw_dotted_rectangle(draw: ImageDraw.Draw, cell: List[int], color: Tuple[int, int, int], width: int):
    """점선 사각형을 그립니다."""
    x0, y0, x1, y1 = cell
    dot_spacing = 5
    
    for x in range(x0, x1, dot_spacing):
        draw.point((x, y0), fill=color)
        draw.point((x, y1), fill=color)
    
    for y in range(y0, y1, dot_spacing):
        draw.point((x0, y), fill=color)
        draw.point((x1, y), fill=color)

def add_content_to_cells(draw: ImageDraw.Draw, cells: List[List[int]], font_path: str):
    """셀에 내용을 추가합니다."""
    font = ImageFont.truetype(font_path, size=12)
    for cell in cells:
        if random.random() < EMPTY_CELL_RATIO:
            continue
        
        x1, y1, x2, y2 = cell
        cell_width = x2 - x1
        cell_height = y2 - y1
        
        if cell_width < MIN_CELL_SIZE_FOR_TEXT or cell_height < MIN_CELL_SIZE_FOR_TEXT:
            continue
        
        content_type = random.choice(['text', 'shape'])
        
        if content_type == 'text':
            text = random_text()
            wrapped_text = wrap_text(text, font, cell_width - TEXT_PADDING)
            
            text_bbox = draw.multiline_textbbox((0, 0), wrapped_text, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
            
            text_x = x1 + (cell_width - text_width) // 2
            text_y = y1 + (cell_height - text_height) // 2
            
            draw.multiline_text((text_x, text_y), wrapped_text, font=font, fill=(0, 0, 0), align='center')
        else:
            add_shape_to_cell(draw, cell)

def add_shape_to_cell(draw: ImageDraw.Draw, cell: List[int]):
    """셀에 도형을 추가합니다."""
    x1, y1, x2, y2 = cell
    shape_type = random.choice(SHAPE_TYPES)
    color = tuple(random.randint(0, 255) for _ in range(3))
    
    if shape_type == 'rectangle':
        draw.rectangle([x1+5, y1+5, x2-5, y2-5], outline=color, width=2)
    elif shape_type == 'ellipse':
        draw.ellipse([x1+5, y1+5, x2-5, y2-5], outline=color, width=2)
    elif shape_type == 'triangle':
        draw.polygon([(x1+5, y2-5), ((x1+x2)//2, y1+5), (x2-5, y2-5)], outline=color, width=2)

def add_imperfections(img: Image.Image, draw: ImageDraw.Draw):
    """이미지에 불완전성을 추가합니다."""
    if random.random() < BLUR_PROBABILITY:
        img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(MIN_BLUR, MAX_BLUR)))
    
    if random.random() < NOISE_PROBABILITY:
        img = add_noise(img)
    
    if random.random() < ROTATION_PROBABILITY:
        angle = random.uniform(-MAX_ROTATION_ANGLE, MAX_ROTATION_ANGLE)
        img = img.rotate(angle, resample=Image.BICUBIC, expand=True)
    
    if random.random() < CONTRAST_ADJUSTMENT_PROBABILITY:
        from PIL import ImageEnhance
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(random.uniform(0.8, 1.2))
    
    return img

def add_noise(img: Image.Image) -> Image.Image:
    """이미지에 노이즈를 추가합니다."""
    draw = ImageDraw.Draw(img)
    width, height = img.size
    for _ in range(random.randint(MIN_NOISE, MAX_NOISE)):
        x = random.randint(0, width-1)
        y = random.randint(0, height-1)
        color = tuple(random.randint(0, 255) for _ in range(3))
        draw.point((x, y), fill=color)
    return img

def create_dataset(num_images: int, output_dir: str):
    """데이터셋을 생성합니다."""
    os.makedirs(output_dir, exist_ok=True)
    train_dir = os.path.join(output_dir, 'train')
    val_dir = os.path.join(output_dir, 'val')
    test_dir = os.path.join(output_dir, 'test')
    
    for dir in [train_dir, val_dir, test_dir]:
        os.makedirs(dir, exist_ok=True)
    
    annotations = {
        'train': [],
        'val': [],
        'test': []
    }
    
    for i in tqdm(range(num_images)):
        img, draw, cells = create_table_image()
        add_content_to_cells(draw, cells, random.choice(FONTS))
        
        if random.random() < IMPERFECT_RATIO:
            img = add_imperfections(img, draw)
        
        image_filename = f"table_{i:05d}.png"
        
        # 데이터 분할
        if i < num_images * TRAIN_RATIO:
            subset = 'train'
        elif i < num_images * (TRAIN_RATIO + VAL_RATIO):
            subset = 'val'
        else:
            subset = 'test'
        
        image_path = os.path.join(output_dir, subset, image_filename)
        img.save(image_path, format=IMAGE_FORMAT, quality=IMAGE_QUALITY)
        
        annotation = {
            'image': image_filename,
            'cells': cells
        }
        annotations[subset].append(annotation)
    
    # 주석 파일 저장
    for subset in ['train', 'val', 'test']:
        with open(os.path.join(output_dir, f'{subset}_annotations.json'), 'w') as f:
            json.dump(annotations[subset], f)

if __name__ == "__main__":
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    
    num_images = 1000  # 생성할 이미지 수
    output_dir = 'table_dataset'
    
    create_dataset(num_images, output_dir)
    print(f"Dataset created with {num_images} images in {output_dir}")
    print(f"Train images: {int(num_images * TRAIN_RATIO)}")
    print(f"Validation images: {int(num_images * VAL_RATIO)}")
    print(f"Test images: {int(num_images * TEST_RATIO)}")
