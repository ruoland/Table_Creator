from PIL import Image, ImageColor, ImageFont, ImageDraw
from dataset_utils import *
from dataset_constant import *
from logging_config import  get_memory_handler, table_logger
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
def add_text_to_cell(draw: ImageDraw.ImageDraw, cell: Dict[str, Any], font_path: str, 
                     text_color: Tuple[int, int, int], position: str) -> str:
    cell_width = cell['x2'] - cell['x1']
    cell_height = cell['y2'] - cell['y1']
    is_header = cell['is_header']
    if cell_width < MIN_CELL_SIZE_FOR_TEXT or cell_height < MIN_CELL_SIZE_FOR_TEXT:
        return ""

    max_font_size = min(int(cell_height * 0.6), int(cell_width * 0.3)) if is_header else min(int(cell_height * 0.4), int(cell_width * 0.2))
    font_size = max_font_size
    
    if is_header:
        text = str(random.randint(1, 100))  # 헤더 셀의 경우 1부터 100 사이의 숫자
    else:
        max_text_length = max(1, min(5, cell_width // (font_size // 2)))
        text = random_text(min_length=1, max_length=max_text_length)
    
    try:
        font = adjust_font_size(draw, text, ImageFont.truetype(font_path, font_size), cell_width - 2*PADDING, cell_height - 2*PADDING)
        
        if font.size <= MIN_FONT_SIZE:
            return ""
        
        # 텍스트 위치 랜덤 선택
        
        text_bbox = draw.multiline_textbbox((0, 0), text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        
        start_x, start_y = calculate_text_position(
            cell['x1']+PADDING, cell['y1']+PADDING, 
            cell_width-2*PADDING, cell_height-2*PADDING, 
            text_width, text_height, position
        )
        
        draw.text((start_x, start_y), text, font=font, fill=text_color)
        
        return text
    except Exception as e:
        table_logger.error(f"Error adding text to cell: {e}")
        return ""



def get_text_color(bg_color: Tuple[int, int, int]) -> Tuple[int, int, int]:
    brightness = (bg_color[0] * 299 + bg_color[1] * 587 + bg_color[2] * 114) / 1000
    if brightness > 128:
        return (max(0, bg_color[0] - 100), max(0, bg_color[1] - 100), max(0, bg_color[2] - 100))
    else:
        return (min(255, bg_color[0] + 100), min(255, bg_color[1] + 100), min(255, bg_color[2] + 100))
