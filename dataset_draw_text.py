from PIL import Image, ImageColor, ImageFont, ImageDraw
from dataset_utils import *
from dataset_constant import *
from logging_config import table_logger
from dataset_config import config, MIN_CELL_SIZE_FOR_TEXT, MIN_FONT_SIZE, PADDING
import numpy as np
from typing import Dict, Any
import random

# 상수 정의
font_cache = {}
text_box_cache = {}

def get_font(font_path, size):
    key = (font_path, size)
    if key not in font_cache:
        font_cache[key] = ImageFont.truetype(font_path, size)
    return font_cache[key]

def calculate_text_position(x: int, y: int, cell_width: int, cell_height: int, 
                            text_width: int, text_height: int, position: str) -> Tuple[int, int]:
    if position == 'center':
        return (x + (cell_width - text_width) // 2, y + (cell_height - text_height) // 2)
    elif position == 'top_left':
        return (x, y)
    elif position == 'top_right':
        return (x + cell_width - text_width, y)
    elif position == 'bottom_left':
        return (x, y + cell_height - text_height)
    elif position == 'bottom_right':
        return (x + cell_width - text_width, y + cell_height - text_height)
    elif position == 'top_center':
        return (x + (cell_width - text_width) // 2, y)
    elif position == 'bottom_center':
        return (x + (cell_width - text_width) // 2, y + cell_height - text_height)
    elif position == 'left_center':
        return (x, y + (cell_height - text_height) // 2)
    elif position == 'right_center':
        return (x + cell_width - text_width, y + (cell_height - text_height) // 2)
    else:
        return (x + (cell_width - text_width) // 2, y + (cell_height - text_height) // 2)

def adjust_font_size(draw, text, font, max_width, max_height):
    min_size = MIN_FONT_SIZE
    max_size = font.size
    while min_size <= max_size:
        mid_size = (min_size + max_size) // 2
        font = get_font(font.path, mid_size)
        bbox = calculate_text_box(draw, text, font)
        if bbox[2] - bbox[0] <= max_width and bbox[3] - bbox[1] <= max_height:
            min_size = mid_size + 1
        else:
            max_size = mid_size - 1
    return get_font(font.path, max_size)

def calculate_text_box(draw, text, font):
    key = (text, font.path, font.size)
    if key not in text_box_cache:
        text_box_cache[key] = draw.multiline_textbbox((0, 0), text, font=font)
    return text_box_cache[key]

def generate_cell_text(is_header, cell_width, font_size):
    if is_header:
        return random.choice([
            str(random.randint(1, 100)),
            random.choice(SUBJECTS),
            random.choice(DEPARTMENTS),
            random.choice(CLASS_TYPES),
            random.choice(TIMES),
            random.choice(BUILDINGS)
        ])
    else:
        num_lines = random.randint(2, 6)
        max_text_length = max(1, min(20, cell_width // (font_size // 3)))
        return '\n'.join([
            random_class_info() if random.random() < 0.3 else
            random_text(min_length=1, max_length=max_text_length)
            for _ in range(num_lines)
        ])

def add_text_to_cell(draw: ImageDraw.Draw, cell: Dict[str, Any], font_path: str, 
                     color: Tuple[int, int, int], position: str, text: str = None) -> None:
    cell_width = cell['x2'] - cell['x1']
    cell_height = cell['y2'] - cell['y1']
    
    if cell_width < MIN_CELL_SIZE_FOR_TEXT or cell_height < MIN_CELL_SIZE_FOR_TEXT:
        return ""

    is_header = cell['is_header']
    max_font_size = min(int(cell_height * (0.7 if is_header else 0.5)), 
                        int(cell_width * (0.4 if is_header else 0.25)))
    
    text = text or generate_cell_text(is_header, cell_width, max_font_size)
    
    try:
        font = adjust_font_size(draw, text, get_font(font_path, max_font_size), 
                                cell_width - 2*PADDING, cell_height - 2*PADDING)
        
        if font.size <= MIN_FONT_SIZE:
            return ""
        
        wrapped_text = wrap_text(text, cell_width - 2*PADDING, font)
        text_bbox = calculate_text_box(draw, wrapped_text, font)
        text_width, text_height = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]
        
        start_x, start_y = calculate_text_position(
            cell['x1']+PADDING, cell['y1']+PADDING, 
            cell_width-2*PADDING, cell_height-2*PADDING, 
            text_width, text_height, position
        )
        
        draw.multiline_text((start_x, start_y), wrapped_text, font=font, fill=color, align="left")
        
        return wrapped_text
    except Exception as e:
        table_logger.error(f"Error adding text to cell: {e}")
        table_logger.error(f"Cell details: {cell}")
        table_logger.error(f"Font path: {font_path}, Text color: {color}, Position: {position}")
        return ""
COMMON_WORDS = tuple(COMMON_WORDS)  # 리스트를 튜플로 변환
def generate_multi_line_text(min_lines=4, max_lines=6):
    return '\n'.join(random.choices(COMMON_WORDS, k=random.randint(min_lines, max_lines)))


def wrap_text(text: str, max_width: int, font: ImageFont.FreeTypeFont) -> str:
    lines = []
    space_width = font.getbbox(' ')[2]
    for paragraph in text.split('\n'):
        words = paragraph.split()
        current_line = []
        current_width = 0
        for word in words:
            word_width = font.getbbox(word)[2]
            if current_width + word_width <= max_width:
                current_line.append(word)
                current_width += word_width + space_width
            else:
                lines.append(' '.join(current_line))
                current_line = [word]
                current_width = word_width + space_width
        lines.append(' '.join(current_line))
    return '\n'.join(lines)


def get_text_color(bg_color: Tuple[int, int, int]) -> Tuple[int, int, int]:
    brightness = (bg_color[0] * 299 + bg_color[1] * 587 + bg_color[2] * 114) / 1000
    if brightness > 128:
        return (max(0, bg_color[0] - 130), max(0, bg_color[1] - 130), max(0, bg_color[2] - 130))
    else:
        return (min(255, bg_color[0] + 130), min(255, bg_color[1] + 130), min(255, bg_color[2] + 130))
