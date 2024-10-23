from PIL import ImageDraw, Image
import random
from typing import Tuple, List, Optional
from table_generation import Table
from dataset_utils import *
from dataset_constant import *
from dataset_config import TableGenerationConfig
from dataset_draw_content import add_content_to_cells, add_shapes, add_title_to_image, apply_imperfections
from dataset_draw_cell import draw_cell, draw_outer_border, redraw_cell_with_overflow, draw_divider_lines, draw_rounded_rectangle
from dataset_draw_preprocess import apply_realistic_effects
from table_generation import generate_coco_annotations
from logging_config import  table_logger
import traceback
import traceback
DEBUG = False
def log_trace():
    if DEBUG:
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
        
        table, cells, table_bbox, is_table_rounded, table_corner_radius = Table(config).create_table(image_width, image_height, margins, title_height, config=config)    

        if config.enable_shapes: #도형
            max_height = max(title_height, table_bbox[1])
            add_shapes(img, 0, title_height, image_width, max_height, bg_color)
        validate_cell_structure(table.cells, "테이블 생성 후")
        
        draw = ImageDraw.Draw(img)
        draw_table(draw, cells, table_bbox, bg_color, has_gap, is_imperfect, config, is_table_rounded, table_corner_radius)
        if config.enable_text_generation or config.enable_shapes:
            add_content_to_cells(img, cells, random.choice(config.fonts), bg_color)

        if is_imperfect:
            img = apply_imperfections(img, cells)
            validate_cell_structure(cells, "불완전성 적용 후")
        img, cells, table_bbox, new_width, new_height = apply_realistic_effects(img, cells, table_bbox, config)
        validate_cell_structure(cells, "현실적 효과 적용 후")

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


def draw_rectangle_with_selective_sides(draw, bbox, color, width, radius= 0, draw_left=True, draw_right=True):
    x0, y0, x1, y1 = bbox
    # 상단 선
    draw.line([(x0, y0), (x1, y0)], fill=color, width=width)
    # 하단 선
    draw.line([(x0, y1), (x1, y1)], fill=color, width=width)
    
    if draw_left:
        if radius > 0:
            draw.line([(x0, y0 + radius), (x0, y1 - radius)], fill=color, width=width)
            # 좌상단 모서리
            draw.arc([x0, y0, x0 + radius * 2, y0 + radius * 2], 180, 270, fill=color, width=width)
            # 좌하단 모서리
            draw.arc([x0, y1 - radius * 2, x0 + radius * 2, y1], 90, 180, fill=color, width=width)
        
        else:# 좌측 선
            draw.line([(x0, y0), (x0, y1)], fill=color, width=width)
            
    
    if draw_right:
        # 우측 선
        if radius > 0:
            draw.line([(x1, y0 + radius), (x1, y1 - radius)], fill=color, width=width)
            # 우상단 모서리
            draw.arc([x1 - radius * 2, y0, x1, y0 + radius * 2], 270, 0, fill=color, width=width)
            # 우하단 모서리
            draw.arc([x1 - radius * 2, y1 - radius * 2, x1, y1], 0, 90, fill=color, width=width)
        else:
            draw.line([(x1, y0), (x1, y1)], fill=color, width=width)
def draw_table(draw: ImageDraw.Draw, cells: List[dict], table_bbox: List[int], 
               bg_color: Tuple[int, int, int], has_gap: bool, is_imperfect: bool, 
               config: TableGenerationConfig, is_table_rounded: bool, table_corner_radius: int) -> None:
    log_trace()
    line_color = get_line_color(bg_color, config)
    line_thickness = config.get_random_line_thickness()
    
    # 표의 좌우 선을 그릴 확률 설정
    can_draw_outer_line = random.random() < config.table_side_line_probability
    corner_radius = 0
    
    if config.enable_rounded_corners and random.random() < config.rounded_corner_probability:
        corner_radius = random.randint(config.min_corner_radius, config.max_corner_radius)
    
    if is_table_rounded:
        draw_rounded_rectangle(draw, table_bbox, table_corner_radius, table_bbox, outline=line_color, width=line_thickness)
    else:
        draw.rectangle(table_bbox, outline=line_color, width=line_thickness)

    draw_rectangle_with_selective_sides(draw, table_bbox, line_color, line_thickness, corner_radius, can_draw_outer_line, can_draw_outer_line)

    # 일반 셀 그리기
    for cell in cells:
        if not cell.get('overflow'):
            if not strict_validate_cell(cell):
                table_logger.warning(f"Invalid cell: {cell}")
                continue

            is_gray = cell.get('is_gray', False)
            cell_bg_color = config.get_random_gray_color(bg_color) or bg_color if config.enable_gray_cells and is_gray else bg_color

            try:
                draw_cell(draw, cell, line_color, is_imperfect, table_bbox, cell_bg_color, config)
            except Exception as e:
                table_logger.error(f"Error drawing cell {cell}: {str(e)}")
                table_logger.error(f"Traceback:\n{traceback.format_exc()}")

    # 셀 이동 효과 적용
    apply_cell_shift_effect(draw, cells, bg_color, line_color, config)

    # 오버플로우된 셀 그리기
    for cell in cells:
        if cell.get('overflow'):
            try:
                redraw_cell_with_overflow(draw, cell, line_color, table_bbox, bg_color, config)
            except Exception as e:
                table_logger.error(f"Error redrawing cell with overflow {cell}: {str(e)}")
    
    if config.enable_outer_border:
        draw_outer_border(draw, table_bbox, line_color)

    # 구분선 그리기
    if config.enable_divider_lines:
        draw_divider_lines(draw, cells, table_bbox, line_color, config)
def apply_cell_shift_effect(draw: ImageDraw.Draw, cells: List[dict], bg_color: Tuple[int, int, int], 
                            line_color: Tuple[int, int, int], config: TableGenerationConfig):
    # 셀을 행과 열 기준으로 정렬
    sorted_cells = sorted(cells, key=lambda c: (c['row'], c['col']))
    
    for i, cell in enumerate(sorted_cells):
        if (not cell.get('overflow') and 
            not cell.get('is_header') and 
            random.random() < config.cell_shift_down_probability):
            
            x1, y1, x2, y2 = cell['x1'], cell['y1'], cell['x2'], cell['y2']
            offset = random.randint(1, 3)
            is_gray = cell.get('is_gray', False)
            cell_bg_color = config.get_random_gray_color(bg_color) or bg_color if config.enable_gray_cells and is_gray else bg_color
            
            # 위쪽 셀 확인
            above_cell = next((c for c in sorted_cells[:i] if c['row'] == cell['row'] - 1 and c['col'] == cell['col']), None)
            
            # 아래쪽 셀 확인
            below_cell = next((c for c in sorted_cells[i+1:] if c['row'] == cell['row'] + 1 and c['col'] == cell['col']), None)
            
            # 위쪽 셀이 있고, 이미 아래로 이동된 경우 고려
            if above_cell and above_cell.get('shifted_down'):
                y1 = above_cell['shifted_y2']
            
            # 위쪽 선을 배경색으로 덮기 (위쪽 셀의 아래 테두리 유지)
            draw.line([(x1, y1+1), (x2, y1+1)], fill=bg_color, width=offset-1)
            
            # 셀 배경 그리기 (약간 아래로 이동)
            draw.rectangle([x1, y1 + offset, x2, y2], fill=cell_bg_color)
            
            # 아래쪽 선 그리기 (아래 셀이 없거나 이동되지 않은 경우에만)
            if not below_cell or not below_cell.get('shifted_down'):
                draw.line([(x1, y2), (x2, y2)], fill=line_color, width=config.get_random_line_thickness())
            
            # 좌우 선 다시 그리기
            line_thickness = config.get_random_line_thickness()
            draw.line([(x1, y1), (x1, y2)], fill=line_color, width=line_thickness)
            draw.line([(x2, y1), (x2, y2)], fill=line_color, width=line_thickness)
            
            # 이동된 셀의 새로운 y 좌표 저장
            cell['shifted_y1'] = y1 + offset
            cell['shifted_y2'] = y2
            cell['shifted_down'] = True

    # 모든 셀의 테두리 다시 그리기
    for i, cell in enumerate(sorted_cells):
        x1, y1, x2, y2 = cell['x1'], cell['y1'], cell['x2'], cell['y2']
        if cell.get('shifted_down'):
            y1 = cell['shifted_y1']
            y2 = cell['shifted_y2']
        
        # 위쪽 테두리 (위 셀이 이동되지 않았거나 현재 셀이 첫 번째 행인 경우에만)
        above_cell = next((c for c in sorted_cells[:i] if c['row'] == cell['row'] - 1 and c['col'] == cell['col']), None)
        if not above_cell or not above_cell.get('shifted_down'):
            draw.line([(x1, y1), (x2, y1)], fill=line_color, width=config.get_random_line_thickness())
        
        # 아래쪽 테두리 (아래 셀이 없거나 이동되지 않은 경우에만)
        below_cell = next((c for c in sorted_cells[i+1:] if c['row'] == cell['row'] + 1 and c['col'] == cell['col']), None)
        if not below_cell or not below_cell.get('shifted_down'):
            draw.line([(x1, y2), (x2, y2)], fill=line_color, width=config.get_random_line_thickness())
        
        # 좌우 테두리
        draw.line([(x1, y1), (x1, y2)], fill=line_color, width=config.get_random_line_thickness())
        draw.line([(x2, y1), (x2, y2)], fill=line_color, width=config.get_random_line_thickness())
