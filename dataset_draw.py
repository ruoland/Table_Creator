from PIL import Image, ImageDraw, ImageFilter, ImageEnhance, ImageChops

import cv2

import numpy as np
import random
from dataset_utils import *
from dataset_constant import *
import logging
from dataset_config import config
from dataset_draw_text import add_text_to_cell
from PIL import ImageFont, ImageDraw, ImageFilter

# 로깅 설정

def draw_cell_line(draw, start, end, color, thickness, is_imperfect):
    x1, y1 = start
    x2, y2 = end
    
    try:
        draw.line([start, end], fill=color, width=thickness)
        
        if is_imperfect:
            if random.random() < 0.2:
                mid_x, mid_y = (x1 + x2) // 2, (y1 + y2) // 2
                protrusion_thickness = random.randint(1, 5)
                protrusion_length = random.randint(1, 3)
                
                protrusion_color = validate_color(color)
                
                if x1 == x2:
                    draw.line([(mid_x, mid_y), (mid_x + protrusion_length, mid_y)], 
                              fill=protrusion_color, width=protrusion_thickness)
                else:
                    draw.line([(mid_x, mid_y), (mid_x, mid_y + protrusion_length)], 
                              fill=protrusion_color, width=protrusion_thickness)
            
            if random.random() < 0.2:
                num_dots = random.randint(1, 3)
                for _ in range(num_dots):
                    dot_x = random.randint(min(x1, x2), max(x1, x2))
                    dot_y = random.randint(min(y1, y2), max(y1, y2))
                    dot_size = random.randint(1, 2)
                    draw.ellipse([(dot_x-dot_size, dot_y-dot_size), 
                                  (dot_x+dot_size, dot_y+dot_size)], 
                                 fill=color)
    except Exception as e:
        logger.error(f"Error drawing cell line: {e}")

def draw_cell_lines(draw, cell, line_color, line_thickness, is_merged, has_gap):
    draw.line([(cell[0], cell[1]), (cell[2], cell[1])], fill=line_color, width=line_thickness)
    draw.line([(cell[0], cell[3]), (cell[2], cell[3])], fill=line_color, width=line_thickness)
    
    if is_merged or has_gap:
        draw.line([(cell[0], cell[1]), (cell[0], cell[3])], fill=line_color, width=line_thickness)
        draw.line([(cell[2], cell[1]), (cell[2], cell[3])], fill=line_color, width=line_thickness)
    else:
        sides_to_draw = random.sample([(cell[0], cell[1], cell[0], cell[3]), 
                                       (cell[2], cell[1], cell[2], cell[3])], 
                                      random.randint(1, 2))
        for side in sides_to_draw:
            draw.line([(side[0], side[1]), (side[2], side[3])], fill=line_color, width=line_thickness)

def apply_imperfections(img, cells):
    if not config.enable_imperfections:
        return img

    width, height = img.size
    draw = ImageDraw.Draw(img)

    if random.random() < 0.3:
        img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.3, 0.7)))
    
    if random.random() < 0.3:
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
    
    for cell in cells:
        if random.random() < 0.1:
            x1, y1, x2, y2 = map(int, cell[:4])
            if x2 > x1 and y2 > y1:
                start_x = random.randint(x1, max(x1, x2-1))
                start_y = random.randint(y1, max(y1, y2-1))
                end_x = random.randint(start_x, x2)
                end_y = random.randint(start_y, y2)
                draw.line([(start_x, start_y), (end_x, end_y)], fill=(0, 0, 0), width=random.randint(1, 3))
    
    return img


def draw_table(draw, cells, table_bbox, bg_color, has_gap, is_imperfect):
    line_color = get_line_color(bg_color)
    header_color = get_header_color(bg_color)
    
    for cell in cells:
        if not validate_cell(cell):
            logger.warning(f"Invalid cell: {cell}")
            continue
        
        # 테이블 경계 내에 있도록 셀 좌표 조정
        x1, y1, x2, y2 = max(cell[0], table_bbox[0]), max(cell[1], table_bbox[1]), \
                         min(cell[2], table_bbox[2]), min(cell[3], table_bbox[3])
        
        if x2 <= x1 or y2 <= y1:
            logger.warning(f"Cell outside table boundaries: {cell}")
            continue
        
        is_header = cell[7] if len(cell) > 7 else False
        cell_color = header_color if is_header else line_color
        is_overflow = len(cell) > 8 and cell[3] > cell[8]
        
        try:
            draw_cell(draw, [x1, y1, x2, y2] + cell[4:], cell_color, is_header, has_gap, is_imperfect, is_overflow, table_bbox, bg_color)
        except Exception as e:
            logger.error(f"Error drawing cell {cell}: {str(e)}")
    
    if config.enable_outer_border:
        draw_outer_border(draw, table_bbox, line_color)
def draw_cell(draw, cell, color, is_header, has_gap, is_imperfect, is_overflow, table_bbox, bg_color):
    x1, y1, x2, y2 = cell[:4]
    original_y2 = cell[8] if len(cell) > 8 else y2
    
    x1, x2 = min(x1, x2), max(x1, x2)
    y1, y2 = min(y1, y2), max(y1, y2)
    
    x1, y1 = max(x1, table_bbox[0]), max(y1, table_bbox[1])
    x2, y2 = min(x2, table_bbox[2]), min(y2, table_bbox[3])
    
    if x2 <= x1 or y2 <= y1:
        logger.warning(f"Cell outside table boundaries: {cell}")
        return
    
    draw.rectangle([x1, y1, x2, y2], fill=bg_color)
    
    if is_overflow:
        draw.rectangle([x1, y1, x2, y2], outline=color)
        overflow_line_y = min(original_y2, table_bbox[3])
        if overflow_line_y > y1:
            draw.line([x1, overflow_line_y, x2, overflow_line_y], fill=color, width=1)
    else:
        if config.enable_cell_border or has_gap:
            draw.rectangle([x1, y1, x2, y2], outline=color)
        else:
            draw.line([x1, y1, x2, y1], fill=color)
            draw.line([x1, y2, x2, y2], fill=color)
    
    draw.line([x1, y1, x1, y2], fill=color)
    draw.line([x2, y1, x2, y2], fill=color)

    if is_imperfect:
        apply_cell_imperfections(draw, x1, y1, x2, y2, color)


def apply_cell_imperfections(draw, x1, y1, x2, y2, color):
    if random.random() < 0.2:
        mid_x, mid_y = (x1 + x2) // 2, (y1 + y2) // 2
        protrusion_thickness = random.randint(1, 3)
        protrusion_length = random.randint(1, 2)
        
        if random.choice([True, False]):
            draw.line([(mid_x, y1), (mid_x, y1 - protrusion_length)], fill=color, width=protrusion_thickness)
        else:
            draw.line([(x1, mid_y), (x1 - protrusion_length, mid_y)], fill=color, width=protrusion_thickness)
def get_merged_cells(cells):
    merged_cells = set()
    for cell in cells:
        if len(cell) > 6 and cell[6]:
            for r in range(cell[7], cell[9] + 1):
                for c in range(cell[8], cell[10] + 1):
                    merged_cells.add((r, c))
    return merged_cells

def draw_outer_border(draw, table_bbox, line_color):
    if random.random() > 0.2:
        outer_line_thickness = random.randint(1, 3)
        draw.rectangle(table_bbox, outline=line_color, width=outer_line_thickness)

def get_overflow_color(base_color):
    # 기본 색상을 기반으로 오버플로우 색상 생성 (약간 더 밝게)
    r, g, b = base_color
    return (min(r + 30, 255), min(g + 30, 255), min(b + 30, 255))
def draw_overflow(draw, cell, original_x2, original_y2, color):
    x1, y1, x2, y2 = map(int, cell[:4])
    dash_length = 5
    space_length = 5
    
    if x2 > original_x2:
        for x in range(int(original_x2), x2, dash_length + space_length):
            draw.line([(x, y1), (min(x + dash_length, x2), y1)], fill=color, width=1)
            draw.line([(x, y2), (min(x + dash_length, x2), y2)], fill=color, width=1)
    
    if y2 > original_y2:
        for y in range(int(original_y2), y2, dash_length + space_length):
            draw.line([(x1, y), (x1, min(y + dash_length, y2))], fill=color, width=1)
            draw.line([(x2, y), (x2, min(y + dash_length, y2))], fill=color, width=1)

def get_header_color(bg_color):
    return (0, 0, 0) if sum(bg_color) > 382 else (255, 255, 255)

def apply_imperfections(img, cells):
    if not config.enable_imperfect_lines:
        return img

    width, height = img.size
    draw = ImageDraw.Draw(img)

    if random.random() < 0.3:
        img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.3, 0.7)))
    
    if random.random() < 0.3:
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
    
    for cell in cells:
        if random.random() < 0.1:
            x1, y1, x2, y2 = map(int, cell[:4])
            if x2 > x1 and y2 > y1:
                start_x = random.randint(x1, max(x1, x2-1))
                start_y = random.randint(y1, max(y1, y2-1))
                end_x = random.randint(start_x, x2)
                end_y = random.randint(start_y, y2)
                draw.line([(start_x, start_y), (end_x, end_y)], fill=(0, 0, 0), width=random.randint(1, 3))
    
    return img
def add_shapes(img, x, y, width, height, bg_color, num_shapes=None, size_range=None):
    draw = ImageDraw.Draw(img)  # ImageDraw 객체 생성
    shape_color = get_line_color(bg_color)
    num_shapes = num_shapes if num_shapes is not None else random.randint(2, 8)
    size_range = size_range or (MIN_SHAPE_SIZE, MAX_SHAPE_SIZE)
    
    max_y = y
    for _ in range(num_shapes):
        shape_type = random.choice(SHAPE_TYPES)
        size = random.randint(*size_range)
        shape_x = random.randint(x, max(x, x + width - size))
        shape_y = random.randint(y, max(y, y + height - size))
        
        draw_shape(draw, shape_type, size, shape_x, shape_y, shape_color)
        max_y = max(max_y, shape_y + size)

    return max_y - y + random.randint(10, 30)

def draw_shape(draw, shape_type, size, x, y, color):
    try:
        if shape_type == 'rectangle':
            draw.rectangle([x, y, x + size, y + size], outline=color, width=2)
        elif shape_type == 'circle':
            draw.ellipse([x, y, x + size, y + size], outline=color, width=2)
        elif shape_type == 'triangle':
            draw.polygon([(x, y + size), (x + size, y + size), (x + size // 2, y)], outline=color, width=2)
        elif shape_type == 'line':
            draw.line([(x, y), (x + size, y + size)], fill=color, width=2)
        elif shape_type == 'arc':
            draw.arc([x, y, x + size, y + size], 0, 270, fill=color, width=2)
        elif shape_type == 'polygon':
            pts = [(random.randint(x, x + size), random.randint(y, y + size)) for _ in range(5)]
            draw.polygon(pts, outline=color, width=2)
    except Exception as e:
        logger.error(f"Error drawing shape: {e}")

def add_title_and_shapes(draw, image_width, image_height, margin_top, bg_color):
    title_height = add_title_to_image(draw, image_width, margin_top, bg_color)
    shapes_height = add_shapes(draw, image_width, image_height, title_height, bg_color)
    return title_height + shapes_height
def add_content_to_cells(img, cells, font_path, bg_color, empty_cell_ratio=EMPTY_CELL_RATIO):
    for cell in cells:
        if random.random() < empty_cell_ratio:
            continue
        
        cell_width, cell_height = int(cell[2] - cell[0]), int(cell[3] - cell[1])
        if cell_width < MIN_CELL_SIZE_FOR_CONTENT or cell_height < MIN_CELL_SIZE_FOR_CONTENT:
            continue

        content_type = random.choice(CELL_CONTENT_TYPES)
        position = random.choice(TEXT_POSITIONS)
        content_color = get_line_color(bg_color)
        
        if config.enable_text_generation and content_type in ['text', 'mixed']:
            add_text_to_cell(ImageDraw.Draw(img), cell, font_path, content_color, position)
        
        if config.enable_shapes and content_type in ['shapes', 'mixed']:
            add_shapes(img, cell[0], cell[1], cell_width, cell_height, bg_color, num_shapes=random.randint(1, 2), size_range=(5, min(cell_width, cell_height) // 3))            
def add_title_to_image(img, image_width, image_height, margin_top, bg_color):
    if not config.enable_title:
        return 0

    bg_color = validate_color(bg_color)
    title = generate_random_title()
    
    font_size = max(MIN_TITLE_SIZE, min(random.randint(int(image_width * 0.02), int(image_width * 0.1)), MAX_TITLE_SIZE))
    font = ImageFont.truetype(random.choice(FONTS), font_size)
    
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
    text_color = get_line_color(bg_color)
    
    draw.text((x, y), title, font=font, fill=text_color)

    return y + text_height


