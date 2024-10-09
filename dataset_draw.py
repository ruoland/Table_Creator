from PIL import ImageDraw, Image
import cv2
import random
from typing import Tuple, List, Optional
from dataset_utils import *
from dataset_constant import *
from dataset_config import TableGenerationConfig
from dataset_draw_content import add_content_to_cells, add_shapes, add_title_to_image, apply_imperfections
from table_generation import create_table
from dataset_draw_cell import draw_cell, draw_outer_border, redraw_cell_with_overflow, draw_divider_lines, draw_rounded_rectangle
from dataset_draw_preprocess import apply_realistic_effects
from table_generation import generate_coco_annotations
from logging_config import  get_memory_handler, table_logger
import traceback
import traceback


def log_trace():
    stack = traceback.extract_stack()
    table_logger.debug("Current call stack:")
    for filename, lineno, name, line in stack[:-1]:  # 마지막 항목(현재 함수)은 제외
        table_logger.debug(f"  File {filename}, line {lineno}, in {name}")
        if line:
            table_logger.debug(f"    {line.strip()}")


def generate_image_and_labels(image_id, resolution, margins, bg_mode, has_gap, is_imperfect=False, config:TableGenerationConfig=None):
    log_trace()
    table_logger.debug(f"generate_image_and_labels 시작: 이미지 ID {image_id}")
    
    try:
        image_width, image_height = resolution
        
        bg_color = config.background_colors['light']['white'] if bg_mode == 'light' else config.background_colors['dark']['black']
        
        img = Image.new('RGB', (image_width, image_height), color=bg_color)
        
        if config.enable_title:
            title_height = add_title_to_image(img, image_width, image_height, margins[1], bg_color)
        else:
            title_height = 0
        
        cells, table_bbox, is_table_rounded, table_corner_radius = create_table(image_width, image_height, margins, title_height, config=config)    

        if config.enable_shapes: #도형
            max_height = max(title_height, table_bbox[1])
            add_shapes(img, 0, title_height, image_width, max_height, bg_color)
        validate_cell_structure(cells, "테이블 생성 후")
        
        draw = ImageDraw.Draw(img)
        draw_table(draw, cells, table_bbox, bg_color, has_gap, is_imperfect, config, is_table_rounded, table_corner_radius)
        if config.enable_text_generation or config.enable_shapes:
            add_content_to_cells(img, cells, random.choice(config.fonts), bg_color)

        if is_imperfect:
            img = apply_imperfections(img, cells)
            validate_cell_structure(cells, "불완전성 적용 후")
        img, cells, table_bbox, transform_matrix, new_width, new_height = apply_realistic_effects(img, cells, table_bbox, title_height, config)
        validate_cell_structure(cells, "현실적 효과 적용 후")
        if config.enable_table_cropping and random.random() < config.table_crop_probability:
            img, cells, table_bbox = apply_table_cropping(img, cells, table_bbox, config.max_crop_ratio)
            
            new_width, new_height = img.size

        coco_annotations = generate_coco_annotations(cells, table_bbox, image_id, config)
        
        return img, coco_annotations, new_width, new_height
    except Exception as e:
        table_logger.error(f"Error generating image {image_id}: {str(e)}")
        table_logger.error(f"info: Error occurred in generate_image_and_labels - {str(e)}")
        import traceback
        table_logger.error(traceback.format_exc())
        return None, None, None, None
def validate_cell_structure(cells, stage):
    log_trace()
    row_ids = sorted(set(cell['row'] for cell in cells))
    col_ids = sorted(set(cell['col'] for cell in cells))
    table_logger.info(f"{stage} - 현재 셀 행 ID: {row_ids}")
    table_logger.info(f"{stage} - 현재 셀 열 ID: {col_ids}")
    table_logger.info(f"{stage} - 총 셀 수: {len(cells)}")
# 각 주요 단계 후에 호출


def apply_table_cropping(img, cells, table_bbox, max_crop_ratio):
    log_trace()
    width, height = img.size
    crop_direction = random.choice(['left', 'right', 'top', 'bottom'])
    crop_amount = int(random.uniform(0, max_crop_ratio) * min(width, height))

    if crop_direction == 'left':
        new_img = img.crop((crop_amount, 0, width, height))
        cells = [{**c, 'x1': max(0, c['x1'] - crop_amount), 'x2': max(0, c['x2'] - crop_amount)} for c in cells]
        table_bbox = [max(0, table_bbox[0] - crop_amount), table_bbox[1], table_bbox[2] - crop_amount, table_bbox[3]]
    elif crop_direction == 'right':
        new_img = img.crop((0, 0, width - crop_amount, height))
        cells = [{**c, 'x2': min(width - crop_amount, c['x2'])} for c in cells]
        table_bbox = [table_bbox[0], table_bbox[1], min(width - crop_amount, table_bbox[2]), table_bbox[3]]
    elif crop_direction == 'top':
        new_img = img.crop((0, crop_amount, width, height))
        cells = [{**c, 'y1': max(0, c['y1'] - crop_amount), 'y2': max(0, c['y2'] - crop_amount)} for c in cells]
        table_bbox = [table_bbox[0], max(0, table_bbox[1] - crop_amount), table_bbox[2], table_bbox[3] - crop_amount]
    else:  # bottom
        new_img = img.crop((0, 0, width, height - crop_amount))
        cells = [{**c, 'y2': min(height - crop_amount, c['y2'])} for c in cells]
        table_bbox = [table_bbox[0], table_bbox[1], table_bbox[2], min(height - crop_amount, table_bbox[3])]

    # 유효하지 않은 셀 제거
    cells = [cell for cell in cells if cell['x2'] > cell['x1'] and cell['y2'] > cell['y1']]

    return new_img, cells, table_bbox
def draw_rounded_rectangle_with_selective_sides(draw, bbox, radius, color, width, draw_left, draw_right):
    x0, y0, x1, y1 = bbox
    # 상단 선
    draw.line([(x0 + radius, y0), (x1 - radius, y0)], fill=color, width=width)
    # 하단 선
    draw.line([(x0 + radius, y1), (x1 - radius, y1)], fill=color, width=width)
    
    if draw_left:
        # 좌측 선
        draw.line([(x0, y0 + radius), (x0, y1 - radius)], fill=color, width=width)
        # 좌상단 모서리
        draw.arc([x0, y0, x0 + radius * 2, y0 + radius * 2], 180, 270, fill=color, width=width)
        # 좌하단 모서리
        draw.arc([x0, y1 - radius * 2, x0 + radius * 2, y1], 90, 180, fill=color, width=width)
    
    if draw_right:
        # 우측 선
        draw.line([(x1, y0 + radius), (x1, y1 - radius)], fill=color, width=width)
        # 우상단 모서리
        draw.arc([x1 - radius * 2, y0, x1, y0 + radius * 2], 270, 0, fill=color, width=width)
        # 우하단 모서리
        draw.arc([x1 - radius * 2, y1 - radius * 2, x1, y1], 0, 90, fill=color, width=width)

def draw_rectangle_with_selective_sides(draw, bbox, color, width, draw_line):
    x0, y0, x1, y1 = bbox
    # 상단 선
    draw.line([(x0, y0), (x1, y0)], fill=color, width=width)
    # 하단 선
    draw.line([(x0, y1), (x1, y1)], fill=color, width=width)
    
    if draw_line:
        # 좌측 선
        draw.line([(x0, y0), (x0, y1)], fill=color, width=width)
    
    if draw_line:
        # 우측 선
        draw.line([(x1, y0), (x1, y1)], fill=color, width=width)
def draw_table(draw: ImageDraw.Draw, cells: List[dict], table_bbox: List[int], 
               bg_color: Tuple[int, int, int], has_gap: bool, is_imperfect: bool, 
               config: TableGenerationConfig, is_table_rounded: bool, table_corner_radius: int) -> None:
    log_trace()
    line_color = get_line_color(bg_color, config)
    line_thickness = config.get_random_line_thickness()
    
    if is_table_rounded:
        draw_rounded_rectangle(draw, table_bbox, table_corner_radius, table_bbox, outline=line_color, width=line_thickness)
    else:
        draw.rectangle(table_bbox, outline=line_color, width=line_thickness)
    # 표의 좌우 선을 그릴 확률 설정
    can_draw_outer_line = random.random() < config.table_side_line_probability


    if config.enable_rounded_corners and random.random() < config.rounded_corner_probability:
        corner_radius = random.randint(config.min_corner_radius, config.max_corner_radius)
        # 좌우 선을 선택적으로 그리기
        draw_rounded_rectangle_with_selective_sides(draw, table_bbox, corner_radius, line_color, line_thickness, can_draw_outer_line, can_draw_outer_line)
    else:
        # 좌우 선을 선택적으로 그리기
        draw_rectangle_with_selective_sides(draw, table_bbox, line_color, line_thickness, can_draw_outer_line)


    for cell in cells:
        if not strict_validate_cell(cell):
            table_logger.warning(f"Invalid cell: {cell}")
            continue
        
        is_header = cell['is_header']
        is_group_header = cell.get('is_group_header', False)
        
        
        cell_color = line_color  # 테두리 색상
        
        if is_group_header:
            cell_bg_color = config.get_group_header_color(bg_color)
        elif config.enable_colored_cells and not is_header:
            cell_bg_color = config.get_random_pastel_color(bg_color) or bg_color
        elif config.enable_gray_cells:
            cell_bg_color = config.get_random_gray_color() or bg_color
        else:
            cell_bg_color = bg_color

        try:

            draw_cell(draw, cell, cell_color, is_header or is_group_header, is_imperfect, table_bbox, cell_bg_color, config)
        except Exception as e:
            table_logger.error(f"Error drawing cell {cell}: {str(e)}")
            table_logger.error(f"Traceback:\n{traceback.format_exc()}")

    # 오버플로우된 셀 다시 그리기
    for cell in cells:
        if cell.get('overflow'):
            try:
                redraw_cell_with_overflow(draw, cell, line_color, table_bbox, cell_bg_color, config)
            except Exception as e:
                table_logger.error(f"Error redrawing cell with overflow {cell}: {str(e)}")
    
    if config.enable_outer_border:
        draw_outer_border(draw, table_bbox, line_color)

    # 구분선 그리기
    if config.enable_divider_lines:
        draw_divider_lines(draw, cells, table_bbox, line_color, config)
