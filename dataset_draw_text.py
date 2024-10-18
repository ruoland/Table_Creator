from PIL import Image, ImageColor, ImageFont, ImageDraw
from dataset_utils import *
from dataset_constant import *
from logging_config import  table_logger
from dataset_config import config, MIN_CELL_SIZE_FOR_TEXT, MIN_FONT_SIZE, PADDING
import numpy as np
from typing import Dict, Any
# 상수 정의

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
   
def adjust_font_size(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont, 
                     max_width: int, max_height: int) -> ImageFont.FreeTypeFont:
    while font.size > MIN_FONT_SIZE:
        bbox = draw.multiline_textbbox((0, 0), text, font=font)
        if bbox[2] - bbox[0] <= max_width and bbox[3] - bbox[1] <= max_height:
            break
        font = ImageFont.truetype(font.path, font.size - 1)
    return font
def add_text_to_cell(draw: ImageDraw.Draw, cell: Dict[str, Any], font_path: str, 
                     color: Tuple[int, int, int], position: str, text: str = None) -> None:
    
    cell_width = cell['x2'] - cell['x1']
    cell_height = cell['y2'] - cell['y1']
    is_header = cell['is_header']
    if cell_width < MIN_CELL_SIZE_FOR_TEXT or cell_height < MIN_CELL_SIZE_FOR_TEXT:
        return ""

    max_font_size = min(int(cell_height * 0.7), int(cell_width * 0.4)) if is_header else min(int(cell_height * 0.5), int(cell_width * 0.25))
    font_size = max_font_size
    
    if is_header:
        header_options = [
            str(random.randint(1, 100)),
            random.choice(SUBJECTS),
            random.choice(DEPARTMENTS),
            random.choice(CLASS_TYPES),
            random.choice(TIMES),
            random.choice(BUILDINGS)
        ]
        text = random.choice(header_options)
    else:
        # 여러 줄의 텍스트 생성
        num_lines = random.randint(2, 6)  # 4-6줄의 텍스트 생성
        text_lines = []
        for _ in range(num_lines):
            max_text_length = max(1, min(20, cell_width // (font_size // 3)))
            if random.random() < 0.3:  # 30% 확률로 수업 정보 생성
                text_lines.append(random_class_info())
            else:
                text_lines.append(random_text(min_length=1, max_length=max_text_length))
        text = '\n'.join(text_lines)
    
    try:
        font = adjust_font_size(draw, text, ImageFont.truetype(font_path, font_size), cell_width - 2*PADDING, cell_height - 2*PADDING)
        
        if font.size <= MIN_FONT_SIZE:
            return ""
        
        # 텍스트를 셀 크기에 맞게 줄바꿈
        wrapped_text = wrap_text(text, cell_width - 2*PADDING, font)
        
        text_bbox = draw.multiline_textbbox((0, 0), wrapped_text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        
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

def wrap_text(text: str, max_width: int, font: ImageFont.FreeTypeFont) -> str:
    lines = []
    for paragraph in text.split('\n'):
        words = paragraph.split()
        current_line = []
        for word in words:
            test_line = ' '.join(current_line + [word])
            bbox = font.getbbox(test_line)
            if bbox[2] - bbox[0] <= max_width:
                current_line.append(word)
            else:
                lines.append(' '.join(current_line))
                current_line = [word]
        lines.append(' '.join(current_line))
    return '\n'.join(lines)

def generate_multi_line_text(min_lines=2, max_lines=6):
    lines = []
    num_lines = random.randint(min_lines, max_lines)
    for _ in range(num_lines):
        lines.append(random.choice(COMMON_WORDS))
    return '\n'.join(lines)

def get_text_color(bg_color: Tuple[int, int, int]) -> Tuple[int, int, int]:
    brightness = (bg_color[0] * 299 + bg_color[1] * 587 + bg_color[2] * 114) / 1000
    if brightness > 128:
        return (max(0, bg_color[0] - 130), max(0, bg_color[1] - 130), max(0, bg_color[2] - 130))
    else:
        return (min(255, bg_color[0] + 130), min(255, bg_color[1] + 130), min(255, bg_color[2] + 130))