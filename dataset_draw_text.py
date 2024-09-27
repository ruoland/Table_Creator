from PIL import Image, ImageColor, ImageFont, ImageDraw
from dataset_utils import *
from dataset_constant import *
import logging
from dataset_config import config
import numpy as np
def wrap_text(text, font, max_width):
    words = text.split()
    lines = []
    current_line = []
    for word in words:
        test_line = ' '.join(current_line + [word])
        bbox = font.getbbox(test_line)
        if bbox[2] - bbox[0] <= max_width:
            current_line.append(word)
        else:
            if current_line:  # 현재 줄에 단어가 있으면 추가
                lines.append(' '.join(current_line))
                current_line = [word]
            else:  # 현재 줄이 비어있으면 단어를 강제로 자름
                lines.append(word[:int(max_width/font.getbbox('W')[2])])
                current_line = []
    if current_line:
        lines.append(' '.join(current_line))
    return '\n'.join(lines)


def calculate_text_position(x, y, cell_width, cell_height, text_width, text_height, position):
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
    
def adjust_font_size(draw, text, font, max_width, max_height):
    while font.size > 8:
        bbox = draw.multiline_textbbox((0, 0), text, font=font)
        if bbox[2] - bbox[0] <= max_width and bbox[3] - bbox[1] <= max_height:
            break
        font = ImageFont.truetype(font.path, font.size - 1)
    return font
def add_text_to_cell(draw, cell, font_path, text_color, position):
    cell_width, cell_height = cell[2] - cell[0], cell[3] - cell[1]
    padding = 2
    
    if cell_width < 20 or cell_height < 20:
        return ""

    max_font_size = min(int(cell_height * 0.4), int(cell_width * 0.2))
    font_size = max_font_size
    max_text_length = max(1, min(5, cell_width // (font_size // 2)))
    text = random_text(min_length=1, max_length=max_text_length)
    
    # 텍스트가 셀에 맞을 때까지 폰트 크기 조정
    while font_size > 8:  # 최소 폰트 크기 설정
        font = ImageFont.truetype(font_path, font_size)
        lines = wrap_text(text, font, cell_width - 2*padding)
        text_bbox = draw.multiline_textbbox((0, 0), lines, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        
        if text_width <= cell_width - 2*padding and text_height <= cell_height - 2*padding:
            break
        font_size -= 1
    
    if font_size <= 8:  # 텍스트가 너무 길어서 맞지 않는 경우
        return ""  # 텍스트를 그리지 않음
    
    start_x, start_y = calculate_text_position(
        cell[0]+padding, cell[1]+padding, 
        cell_width-2*padding, cell_height-2*padding, 
        text_width, text_height, position
    )
    
    # 텍스트가 셀 경계를 넘지 않도록 보정
    start_x = max(cell[0] + padding, min(start_x, cell[2] - text_width - padding))
    start_y = max(cell[1] + padding, min(start_y, cell[3] - text_height - padding))
    
    draw.multiline_text((start_x, start_y), lines, font=font, fill=text_color, align='center')
    
    return text



def add_text_to_image(img, text, position, font_path, font_size, text_color):
    draw = ImageDraw.Draw(img)
    
    # 폰트 로드
    font = ImageFont.truetype(font_path, font_size)
    
    # PIL 이미지에 텍스트 추가
    draw.text(position, text, font=font, fill=text_color)
    
    return img

