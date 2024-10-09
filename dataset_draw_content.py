
from PIL import ImageDraw, ImageFilter, Image, ImageFont
from dataset_utils import *
from dataset_constant import *
from dataset_config import TableGenerationConfig, MIN_CELL_SIZE_FOR_CONTENT
import cv2
import random
from logging_config import  get_memory_handler, table_logger

from dataset_draw_text import add_text_to_cell, get_text_color
from typing import Tuple, List, Optional

def apply_imperfections(img: Image.Image, cells: List[Dict[str, Any]]) -> Image.Image:
    if not config.enable_table_imperfections:
        return img

    width, height = img.size
    draw = ImageDraw.Draw(img)
    if random.random() < config.blur_probability:
        min_radius, max_radius = config.blur_radius_range
        img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(min_radius, max_radius)))
    
    if config.enable_random_lines:
        add_random_lines(draw, width, height)
    
    if config.enable_cell_inside_imperfections:
        add_random_cell_imperfections(draw, cells)
    return img

#랜덤한 선을 추가하는 부분
def add_random_cell_imperfections(draw: ImageDraw.Draw, cells: List[Dict[str, Any]]) -> None:
    for cell in cells:
        if random.random() < config.cell_imperfection_probability:
            x1, y1, x2, y2 = cell['x1'], cell['y1'], cell['x2'], cell['y2']
            if x2 > x1 and y2 > y1:
                start_x = random.randint(x1, max(x1, x2-1))
                start_y = random.randint(y1, max(y1, y2-1))
                end_x = random.randint(start_x, x2)
                end_y = random.randint(start_y, y2)
                draw.line([(start_x, start_y), (end_x, end_y)], fill=(0, 0, 0), width=random.randint(1, 3))
                
def add_random_lines(draw: ImageDraw.Draw, width: int, height: int) -> None:
    if random.random() < config.random_line_probability:
        for _ in range(random.randint(1, 3)):
            x1, y1 = random.randint(0, width-1), random.randint(0, height-1)
            x2, y2 = random.randint(0, width-1), random.randint(0, height-1)
            
            gap_start = random.uniform(0.1, 1.0)
            gap_end = min(gap_start + random.uniform(0.1, 0.6), 1.0)
            
            draw.line([(x1, y1), 
                       (int(x1 + (x2-x1)*gap_start), int(y1 + (y2-y1)*gap_start))], 
                      fill=(255, 255, 255), width=1)
            draw.line([(int(x1 + (x2-x1)*gap_end), int(y1 + (y2-y1)*gap_end)), 
                       (x2, y2)], 
                      fill=(255, 255, 255), width=1)

def add_shapes(img: Image.Image, x: int, y: int, width: int, height: int, 
               bg_color: Tuple[int, int, int], num_shapes: Optional[int] = None, 
               size_range: Optional[Tuple[int, int]] = None) -> int:
    draw = ImageDraw.Draw(img)
    shape_color = get_line_color(bg_color, config)
    num_shapes = num_shapes if num_shapes is not None else random.randint(config.min_shapes, config.max_shapes)
    
    # size_range 조정
    min_size = min(config.min_shape_size, width // 2, height // 2)
    max_size = min(config.max_shape_size, width, height)
    size_range = (min_size, max(min_size, max_size))
    
    max_y = y
    for _ in range(num_shapes):
        shape_type = random.choice(config.shape_types)
        size = random.randint(*size_range)
        
        # 도형이 셀 내부에 위치하도록 보장
        shape_x = random.randint(x, max(x, x + width - size))
        shape_y = random.randint(y, max(y, y + height - size))
        
        draw_shape(draw, shape_type, size, shape_x, shape_y, shape_color)
        max_y = max(max_y, shape_y + size)

    return max_y - y + random.randint(10, 30)

def draw_shape(draw: ImageDraw.Draw, shape_type: str, size: int, x: int, y: int, color: Tuple[int, int, int]) -> None:
    try:
        if shape_type == 'rectangle':
            draw.rectangle([x, y, x + size, y + size], outline=color, width=config.shape_line_width)
        elif shape_type == 'circle':
            draw.ellipse([x, y, x + size, y + size], outline=color, width=config.shape_line_width)
        elif shape_type == 'triangle':
            draw.polygon([(x, y + size), (x + size, y + size), (x + size // 2, y)], outline=color, width=config.shape_line_width)
        elif shape_type == 'line':
            draw.line([(x, y), (x + size, y + size)], fill=color, width=config.shape_line_width)
        elif shape_type == 'arc':
            draw.arc([x, y, x + size, y + size], 0, 270, fill=color, width=config.shape_line_width)
        elif shape_type == 'polygon':
            pts = [(random.randint(x, x + size), random.randint(y, y + size)) for _ in range(5)]
            draw.polygon(pts, outline=color, width=config.shape_line_width)
    except Exception as e:
        table_logger.error(f"Error drawing shape: {e}")

        
def add_protrusion(draw: ImageDraw.Draw, x1: int, y1: int, x2: int, y2: int, color: Tuple[int, int, int]) -> None:
    mid_x, mid_y = (x1 + x2) // 2, (y1 + y2) // 2
    protrusion_thickness = random.randint(config.min_protrusion_thickness, config.max_protrusion_thickness)
    protrusion_length = random.randint(config.min_protrusion_length, config.max_protrusion_length)
    
    protrusion_color = validate_color(color)
    
    if x1 == x2:
        draw.line([(mid_x, mid_y), (mid_x + protrusion_length, mid_y)], 
                  fill=protrusion_color, width=protrusion_thickness)
    else:
        draw.line([(mid_x, mid_y), (mid_x, mid_y + protrusion_length)], 
                  fill=protrusion_color, width=protrusion_thickness)

def add_dots(draw: ImageDraw.Draw, x1: int, y1: int, x2: int, y2: int, color: Tuple[int, int, int]) -> None:
    num_dots = random.randint(1, 3)
    for _ in range(num_dots):
        dot_x = random.randint(min(x1, x2), max(x1, x2))
        dot_y = random.randint(min(y1, y2), max(y1, y2))
        dot_size = random.randint(config.min_dot_size, config.max_dot_size)
        draw.ellipse([(dot_x-dot_size, dot_y-dot_size), 
                      (dot_x+dot_size, dot_y+dot_size)], 
                     fill=color)
def add_content_to_cells(img: Image.Image, cells: List[dict], font_path: str, 
                         bg_color: Tuple[int, int, int], empty_cell_ratio: float = config.empty_cell_ratio) -> None:
    draw = ImageDraw.Draw(img)
    width, height = img.size
    for cell in cells:
        if random.random() < empty_cell_ratio:
            continue
        
        cell_width, cell_height = int(cell['x2'] - cell['x1']), int(cell['y2'] - cell['y1'])
        if cell_width < MIN_CELL_SIZE_FOR_CONTENT or cell_height < MIN_CELL_SIZE_FOR_CONTENT:
            continue

        content_type = random.choice(config.cell_content_types)
        position = random.choice(config.text_positions)
        # 좌표 유효성 검사 및 조정
        x1 = max(0, min(cell['x1'], width - 1))
        y1 = max(0, min(cell['y1'], height - 1))
        x2 = max(0, min(cell['x2'], width - 1))
        y2 = max(0, min(cell['y2'], height - 1))
        
        if x2 <= x1 or y2 <= y1:
            continue  # 유효하지 않은 셀 건너뛰기
        cell_bg_color = img.getpixel((cell['x1'] + 1, cell['y1'] + 1))
        content_color = get_text_color(cell_bg_color)
        
        if config.enable_text_generation and content_type in ['text', 'mixed']:
            # 'is_header' 키가 없는 경우를 처리
            is_header = cell.get('is_header', False)
            
            # 여러 줄의 텍스트 생성
            multi_line_text = generate_multi_line_text(min_lines=4, max_lines=6)
            
            # 셀 크기에 맞게 텍스트 줄바꿈
            font = ImageFont.truetype(font_path, size=random.randint(10, 20))
            wrapped_text = wrap_text(multi_line_text, cell_width - 10, font)  # 10은 여백
            
            add_text_to_cell(draw, {**cell, 'is_header': is_header}, font_path, content_color, position, text=wrapped_text)
        
        if config.enable_shapes and content_type in ['shapes', 'mixed']:
            add_shapes(img, cell['x1'], cell['y1'], cell_width, cell_height, cell_bg_color, 
                       num_shapes=random.randint(1, 2), size_range=(5, min(cell_width, cell_height) // 3))

def generate_multi_line_text(min_lines=4, max_lines=6):
    lines = []
    num_lines = random.randint(min_lines, max_lines)
    for _ in range(num_lines):
        lines.append(random.choice(COMMON_WORDS))
    return '\n'.join(lines)

def wrap_text(text, max_width, font):
    words = text.split()
    lines = []
    current_line = []
    for word in words:
        test_line = ' '.join(current_line + [word])
        bbox = font.getbbox(test_line)
        width, height = bbox[2] - bbox[0], bbox[3] - bbox[1]
        if width <= max_width:
            current_line.append(word)
        else:
            lines.append(' '.join(current_line))
            current_line = [word]
    lines.append(' '.join(current_line))
    return '\n'.join(lines)


def add_title_to_image(img: Image.Image, image_width: int, image_height: int, 
                       margin_top: int, bg_color: Tuple[int, int, int]) -> int:
    if not config.enable_title:
        return 0

    bg_color = validate_color(bg_color)
    title = generate_random_title()
    
    font_size = max(config.min_title_size, min(random.randint(int(image_width * 0.02), int(image_width * 0.1)), config.max_title_size))
    font = ImageFont.truetype(random.choice(config.fonts), font_size)
    
    draw = ImageDraw.Draw(img)
    left, top, right, bottom = draw.textbbox((0, 0), title, font=font)
    text_width, text_height = right - left, bottom - top
    
    if text_width > image_width * 0.9:
        words = title.split()
        half = len(words) // 2
        title = ' '.join(words[:half]) + '\n' + ' '.join(words[half:])
        left, top, right, bottom = draw.textbbox((0, 0), title, font=font)
        text_width, text_height = right - left, bottom - top
    
    x = (image_width - text_width) // 2
    y = margin_top + random.randint(5, 20)
    text_color = get_line_color(bg_color, config)
    
    draw.text((x, y), title, font=font, fill=text_color)

    return y + text_height