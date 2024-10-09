
from PIL import ImageDraw, ImageFilter, Image
from dataset_utils import *
from dataset_constant import *
from dataset_config import config, TableGenerationConfig
import cv2
import random
from dataset_draw_content import add_protrusion, add_dots
from logging_config import  get_memory_handler, table_logger

from typing import Tuple, List, Optional
from PIL import ImageDraw, ImageFilter, Image, ImageFont
import random
import numpy as np
from typing import Tuple, List, Optional
def draw_cell(draw: ImageDraw.Draw, cell: dict, line_color: Tuple[int, int, int], 
              is_header: bool, is_imperfect: bool, 
              table_bbox: List[int], bg_color: Tuple[int, int, int], config: TableGenerationConfig,
              is_no_side_border_row: bool = False) -> Optional[Tuple[int, int, int]]:
    
    table_logger.debug(f"Drawing cell: {cell}")

    # cell_type 확인 및 로깅
    cell_type = cell.get('cell_type', 'normal_cell')
    table_logger.debug(f"Cell type: {cell_type}")


    is_gray = cell.get('is_gray', False)

    x1, y1, x2, y2 = cell['x1'], cell['y1'], cell['x2'], cell['y2']

    # 좌표 유효성 검사 및 조정
    x1, x2 = min(x1, x2), max(x1, x2)
    y1, y2 = min(y1, y2), max(y1, y2)
    
    # 오버플로우 처리
    if 'overflow_y1' in cell:
        y1 = min(y1, cell['overflow_y1'])
    if 'overflow_y2' in cell:
        y2 = max(y2, cell['overflow_y2'])


    # 최소 크기 보장
    if x2 <= x1:
        x2 = x1 + config.min_cell_width
    if y2 <= y1:
        y2 = y1 + config.min_cell_height

    table_logger.debug(f"Adjusted coordinates: x1={x1}, y1={y1}, x2={x2}, y2={y2}")

    # 셀 배경색 결정
    if config.enable_colored_cells:
        cell_bg_color = config.get_random_pastel_color(bg_color) or bg_color
    elif config.enable_gray_cells and is_gray:
        cell_bg_color = config.get_random_gray_color() or bg_color
    else:
        cell_bg_color = bg_color

    # 셀 그리기
    draw.rectangle([x1, y1, x2, y2], fill=cell_bg_color)

    can_draw_border = True

    # 컬러 셀이 활성화되어 있거나 현재 셀이 회색인 경우
    if config.enable_colored_cells or is_gray:
        # 무작위 값이 테두리 없는 셀의 확률보다 작은 경우
        if random.random() < config.no_border_cell_probability:
            can_draw_border = False

    # 현재 셀이 회색인 경우
    if is_gray:
        # 무작위 값이 회색 셀의 테두리 없는 확률보다 작은 경우
        if random.random() < config.gray_cell_no_border_probability:
            can_draw_border = False
            

    if can_draw_border:
        line_thickness = config.get_random_line_thickness()
        is_rounded = config.enable_rounded_corners and random.random() < config.rounded_corner_probability
        is_no_side_border = random.random() < config.no_side_borders_cells_probability

        if is_rounded:
            corner_radius = random.randint(config.min_corner_radius, config.max_corner_radius)
            draw_rounded_rectangle(draw, [x1, y1, x2, y2], corner_radius, table_bbox,
                                   fill=cell_bg_color, outline=line_color, width=line_thickness)
        elif is_imperfect and is_no_side_border:
            draw_imperfect_cell_border(draw, x1, y1, x2, y2, line_color, line_thickness, config, is_no_side_border)
        else:
            line_thickness = 1
            draw.rectangle([x1, y1, x2, y2], outline=line_color, width=line_thickness)

    # 오버플로우 처리
    if cell.get('overflow') and config.enable_overflow:
        overflow = cell['overflow']
        direction = overflow['direction']
        overflow_y1 = max(table_bbox[1], cell.get('overflow_y1', y1))
        overflow_y2 = min(table_bbox[3], cell.get('overflow_y2', y2))
        
        # 오버플로우 영역 그리기
        if direction in ['up', 'both'] and overflow_y1 < y1:
            draw.rectangle([x1, overflow_y1, x2, y1], fill=cell_bg_color)
        if direction in ['down', 'both'] and overflow_y2 > y2:
            draw.rectangle([x1, y2, x2, overflow_y2], fill=cell_bg_color)

        # 오버플로우 테두리 그리기
        if can_draw_border:
            if is_rounded:
                draw_rounded_overflow(draw, cell, line_color, line_thickness, corner_radius, table_bbox)
            else:
                draw_regular_overflow(draw, cell, line_color, line_thickness)
    table_logger.debug(f"Finished drawing cell: {cell}")
    return cell_bg_color
def draw_imperfect_cell_border(draw: ImageDraw.Draw, x1: int, y1: int, x2: int, y2: int, 
                               line_color: Tuple[int, int, int], line_thickness: int, 
                               config: TableGenerationConfig, no_side_borders: bool = False):
    sides = [
        ('top', [(x1, y1), (x2, y1)]),
        ('bottom', [(x1, y2), (x2, y2)]),
        ('left', [(x1, y1), (x1, y2)]),
        ('right', [(x2, y1), (x2, y2)])
    ]

    for side, (start, end) in sides:
        if no_side_borders and side in ['left', 'right']:
            continue  # 양옆 선을 그리지 않음
        if random.random() < config.imperfect_line_probability[side]:
            draw_imperfect_line(draw, start, end, line_color, line_thickness, config)
def draw_imperfect_line(draw: ImageDraw.Draw, start: Tuple[int, int], end: Tuple[int, int], 
                        line_color: Tuple[int, int, int], line_thickness: int, config: TableGenerationConfig):
    x1, y1 = start
    x2, y2 = end
    
    # 1. 선의 끊김 효과 (기존 코드)
    if random.random() < config.line_break_probability:
        mid = ((x1 + x2) // 2, (y1 + y2) // 2)
        draw.line([start, mid], fill=line_color, width=line_thickness)
        if random.random() < 0.5:  # 50% 확률로 나머지 반쪽을 그림
            draw.line([mid, end], fill=line_color, width=line_thickness)
        return  # 선이 끊긴 경우 다른 효과를 적용하지 않음
    
    # 2. 선의 불규칙한 두께
    if random.random() < config.irregular_thickness_probability:
        num_segments = random.randint(5, 10)
        for i in range(num_segments):
            segment_start = (
                int(x1 + (x2 - x1) * i / num_segments),
                int(y1 + (y2 - y1) * i / num_segments)
            )
            segment_end = (
                int(x1 + (x2 - x1) * (i + 1) / num_segments),
                int(y1 + (y2 - y1) * (i + 1) / num_segments)
            )
            segment_thickness = max(1, line_thickness + random.randint(-2, 2))
            draw.line([segment_start, segment_end], fill=line_color, width=segment_thickness)
    else:
        draw.line([start, end], fill=line_color, width=line_thickness)
    
    # 3. 선의 미세한 굴곡
    if random.random() < config.line_curve_probability:
        num_points = 10
        points = []
        for i in range(num_points):
            t = i / (num_points - 1)
            x = int(x1 + t * (x2 - x1) + random.randint(-3, 3))
            y = int(y1 + t * (y2 - y1) + random.randint(-3, 3))
            points.append((x, y))
        draw.line(points, fill=line_color, width=line_thickness)
    
        # 4. 색상 변화
    if random.random() < config.color_variation_probability:
        varied_color = apply_color_variation(line_color, line_color, config)
        draw.line([start, end], fill=varied_color, width=line_thickness)
    else:
        draw.line([start, end], fill=line_color, width=line_thickness)
            
    if random.random() < config.end_imperfection_probability:
        for point in [start, end]:
            size = random.randint(1, 3)  # 크기 변화
            offset_x = random.randint(-1, 1)  # x축 오프셋
            offset_y = random.randint(-1, 1)  # y축 오프셋
            draw.ellipse([point[0]-size+offset_x, point[1]-size+offset_y, 
                        point[0]+size+offset_x, point[1]+size+offset_y], 
                        fill=line_color, outline=line_color)
    # 6. 모서리 불완전성 (기존 코드)
    if random.random() < config.corner_imperfection_probability:
        corner_size = random.randint(1, 3)
        if (x1, y1) == start:  # 왼쪽 위 또는 왼쪽 아래 모서리
            draw.rectangle([x1-corner_size, y1-corner_size, x1+corner_size, y1+corner_size], fill=line_color)
        else:  # 오른쪽 위 또는 오른쪽 아래 모서리
            draw.rectangle([x2-corner_size, y2-corner_size, x2+corner_size, y2+corner_size], fill=line_color)
    

    # 8. 선의 흐림 효과 (기존 코드)
    if random.random() < config.line_blur_probability:
        line_image = Image.new('RGBA', (max(x1, x2) - min(x1, x2) + 10, max(y1, y2) - min(y1, y2) + 10), (0, 0, 0, 0))
        line_draw = ImageDraw.Draw(line_image)
        line_draw.line([(5, 5), (line_image.width-5, line_image.height-5)], fill=line_color, width=line_thickness)
        blurred_line = line_image.filter(ImageFilter.GaussianBlur(radius=1))
        draw.bitmap((min(x1, x2)-5, min(y1, y2)-5), blurred_line)

    # 9. 선의 투명도 변화
    if random.random() < config.transparency_variation_probability:
        transparent_line = Image.new('RGBA', (max(x1, x2) - min(x1, x2) + 10, max(y1, y2) - min(y1, y2) + 10), (0, 0, 0, 0))
        transparent_draw = ImageDraw.Draw(transparent_line)
        r, g, b = line_color
        transparent_draw.line([(5, 5), (transparent_line.width-5, transparent_line.height-5)], 
                              fill=(r, g, b, random.randint(128, 255)), width=line_thickness)
        draw.bitmap((min(x1, x2)-5, min(y1, y2)-5), transparent_line, fill=None)
def apply_color_variation(line_color: Tuple[int, int, int], bg_color: Tuple[int, int, int], config: TableGenerationConfig) -> Tuple[int, int, int]:
    """선 색상에 변화를 주되, 배경색과의 대비를 유지합니다."""
    r, g, b = line_color
    
    # 각 채널별로 독립적으로 변화를 줍니다
    varied_color = (
        max(0, min(255, r + random.randint(-20, 20))),
        max(0, min(255, g + random.randint(-20, 20))),
        max(0, min(255, b + random.randint(-20, 20)))
    )
    
    # 배경색과의 대비를 확인합니다
    if is_sufficient_contrast(bg_color, varied_color):
        return varied_color
    else:
        # 대비가 충분하지 않으면 원래 색상으로 돌아갑니다
        return line_color


def draw_rounded_overflow(draw, cell, line_color, border_width, corner_radius, table_bbox):
    x1, y1, x2, y2 = cell['x1'], cell['y1'], cell['x2'], cell['y2']
    overflow_y1 = cell.get('overflow_y1', y1)
    overflow_y2 = cell.get('overflow_y2', y2)
    direction = cell['overflow']['direction']

    if direction in ['up', 'both']:
        draw_rounded_rectangle(draw, [x1, overflow_y1, x2, y1], corner_radius, table_bbox, outline=line_color, width=border_width)
    if direction in ['down', 'both']:
        draw_rounded_rectangle(draw, [x1, y2, x2, overflow_y2], corner_radius, table_bbox, outline=line_color, width=border_width)

def draw_regular_overflow(draw, cell, line_color, border_width):
    x1, y1, x2, y2 = cell['x1'], cell['y1'], cell['x2'], cell['y2']
    overflow_y1 = cell.get('overflow_y1', y1)
    overflow_y2 = cell.get('overflow_y2', y2)
    direction = cell['overflow']['direction']

    if direction in ['up', 'both']:
        draw.line([(x1, overflow_y1), (x2, overflow_y1)], fill=line_color, width=border_width)
    if direction in ['down', 'both']:
        draw.line([(x1, overflow_y2), (x2, overflow_y2)], fill=line_color, width=border_width)
    draw.line([(x1, overflow_y1), (x1, overflow_y2)], fill=line_color, width=border_width)
    draw.line([(x2, overflow_y1), (x2, overflow_y2)], fill=line_color, width=border_width)
def redraw_cell_with_overflow(draw: ImageDraw.Draw, cell: dict, line_color: Tuple[int, int, int], 
                              table_bbox: List[int], bg_color: Tuple[int, int, int], config: TableGenerationConfig) -> None:
    # cell_type 확인 및 업데이트
    cell_type = cell.get('cell_type', 'normal_cell')
    if cell_type == 'merged_cell':
        cell['cell_type'] = 'merged_overflow_cell'
    elif cell_type == 'normal_cell':
        cell['cell_type'] = 'overflow_cell'

    table_logger.debug(f"Redrawing cell with overflow. Updated cell type: {cell['cell_type']}")

    x1, y1, x2, y2 = cell['x1'], cell['y1'], cell['x2'], cell['y2']
    
    is_gray = cell.get('is_gray', True)
    
    if is_gray:
        cell_bg_color = config.get_random_gray_color() or bg_color
    else:
        cell_bg_color = bg_color

    direction = cell['overflow']['direction']
    overflow_y1 = cell.get('overflow_y1', y1)
    overflow_y2 = cell.get('overflow_y2', y2)
    
    # 둥근 모서리 설정
    is_rounded = config.enable_rounded_corners and random.random() < config.rounded_corner_probability
    corner_radius = random.randint(config.min_corner_radius, config.max_corner_radius) if is_rounded else 0
    
    # 셀 배경 다시 그리기 (오버플로우 영역 포함)
    if is_rounded:
        draw_rounded_rectangle(draw, [x1, overflow_y1, x2, overflow_y2], corner_radius, table_bbox,
                               fill=cell_bg_color, outline=None)
    else:
        draw.rectangle([x1, overflow_y1, x2, overflow_y2], fill=cell_bg_color)

    # 셀 테두리 다시 그리기
    border_width = config.min_line_thickness

    if is_rounded:
        # 둥근 모서리로 테두리 그리기
        draw_rounded_rectangle(draw, [x1, overflow_y1, x2, overflow_y2], corner_radius, table_bbox,
                               outline=line_color, width=border_width)
    else:
        # 일반 테두리 그리기
        # 왼쪽과 오른쪽 테두리
        draw.line([(x1, overflow_y1), (x1, overflow_y2)], fill=line_color, width=border_width)
        draw.line([(x2, overflow_y1), (x2, overflow_y2)], fill=line_color, width=border_width)

        # 위쪽 테두리
        if direction in ['up', 'both']:
            draw.line([(x1, overflow_y1), (x2, overflow_y1)], fill=line_color, width=border_width)
        else:
            draw.line([(x1, y1), (x2, y1)], fill=line_color, width=border_width)

        # 아래쪽 테두리
        if direction in ['down', 'both']:
            draw.line([(x1, overflow_y2), (x2, overflow_y2)], fill=line_color, width=border_width)
        else:
            draw.line([(x1, y2), (x2, y2)], fill=line_color, width=border_width)

    table_logger.debug(f"Finished redrawing cell with overflow: {cell}")


def draw_outer_border(draw: ImageDraw.Draw, table_bbox: List[int], line_color: Tuple[int, int, int]) -> None:
    if random.random() > config.outer_border_probability:
        outer_line_thickness = random.randint(config.min_outer_line_thickness, config.max_outer_line_thickness)
        draw.rectangle(table_bbox, outline=line_color, width=outer_line_thickness)

def draw_divider_lines(draw: ImageDraw.Draw, cells: List[dict], table_bbox: List[int], 
                       line_color: Tuple[int, int, int], config: TableGenerationConfig):
    x1, y1, x2, y2 = table_bbox
    rows = sorted(set(cell['row'] for cell in cells))
    cols = sorted(set(cell['col'] for cell in cells))
    
    # 수평 구분선
    for i, row in enumerate(rows):
        if i == 0 or i == len(rows) - 1:  # 첫 행과 마지막 행은 제외
            continue
        
        # 현재 행의 셀들 확인
        row_cells = [cell for cell in cells if cell['row'] == row]
        
        # 오버플로우가 있는 셀이 있는지 확인
        has_overflow = any(cell.get('overflow') for cell in row_cells)
        
        if not has_overflow and random.random() < config.horizontal_divider_probability:
            y = sum(cell['y2'] for cell in row_cells) // len(row_cells)
            thickness = random.randint(*config.divider_line_thickness_range)
            draw_imperfect_line(draw, (x1, y), (x2, y), line_color, thickness, config)
    
    # 수직 구분선
    for i, col in enumerate(cols):
        if i == 0 or i == len(cols) - 1:  # 첫 열과 마지막 열은 제외
            continue
        # 현재 행의 셀들 확인
        col_cells = [cell for cell in cells if cell['col'] == col]
        
        # 오버플로우가 있는 셀이 있는지 확인
        has_overflow = any(cell.get('overflow') for cell in col_cells)
        
        if not has_overflow and random.random() < config.vertical_divider_probability:
            x = sum(cell['x2'] for cell in cells if cell['col'] == col) // len([cell for cell in cells if cell['col'] == col])
            thickness = random.randint(*config.divider_line_thickness_range)
            draw_imperfect_line(draw, (x, y1), (x, y2), line_color, thickness, config)
def draw_rounded_rectangle(draw, xy, corner_radius, table_bbox, fill=None, outline=None, width=1):
    x1, y1, x2, y2 = xy
    table_x1, table_y1, table_x2, table_y2 = table_bbox
    
    # 좌표 유효성 검사 및 조정
    x1, x2 = min(x1, x2), max(x1, x2)
    y1, y2 = min(y1, y2), max(y1, y2)
    
    # 최소 크기 보장
    corner_radius = min(corner_radius, (x2 - x1) // 2, (y2 - y1) // 2)
    
    # 내부 채우기
    if fill:
        draw.rectangle([x1, y1, x2, y2], fill=fill)
        
        # 모서리 그리기 (오버플로우 고려)
        if y1 >= table_y1:  # 상단 모서리
            draw.pieslice([x1, y1, x1 + corner_radius * 2, y1 + corner_radius * 2], 180, 270, fill=fill)
            draw.pieslice([x2 - corner_radius * 2, y1, x2, y1 + corner_radius * 2], 270, 360, fill=fill)
        if y2 <= table_y2:  # 하단 모서리
            draw.pieslice([x1, y2 - corner_radius * 2, x1 + corner_radius * 2, y2], 90, 180, fill=fill)
            draw.pieslice([x2 - corner_radius * 2, y2 - corner_radius * 2, x2, y2], 0, 90, fill=fill)
    
    # 외곽선 그리기
    if outline:
        # 상단 선
        if y1 >= table_y1:
            draw.arc([x1, y1, x1 + corner_radius * 2, y1 + corner_radius * 2], 180, 270, fill=outline, width=width)
            draw.line([x1 + corner_radius, y1, x2 - corner_radius, y1], fill=outline, width=width)
            draw.arc([x2 - corner_radius * 2, y1, x2, y1 + corner_radius * 2], 270, 360, fill=outline, width=width)
        
        # 하단 선
        if y2 <= table_y2:
            draw.arc([x1, y2 - corner_radius * 2, x1 + corner_radius * 2, y2], 90, 180, fill=outline, width=width)
            draw.line([x1 + corner_radius, y2, x2 - corner_radius, y2], fill=outline, width=width)
            draw.arc([x2 - corner_radius * 2, y2 - corner_radius * 2, x2, y2], 0, 90, fill=outline, width=width)
        
        # 좌우 선
        draw.line([x1, y1 + corner_radius, x1, y2 - corner_radius], fill=outline, width=width)
        draw.line([x2, y1 + corner_radius, x2, y2 - corner_radius], fill=outline, width=width)
