
from PIL import ImageDraw, ImageFilter, Image
from dataset_utils import *
from dataset_constant import *
from dataset_config import config, TableGenerationConfig
import random
from logging_config import  get_memory_handler, table_logger

from typing import Tuple, List, Optional
from PIL import ImageDraw, ImageFilter, Image, ImageFont
import random
import numpy as np
from typing import Tuple, List, Optional
# 셀 그리기
def draw_cell(draw: ImageDraw.Draw, cell: dict, line_color: Tuple[int, int, int], 
              is_imperfect: bool, 
              table_bbox: List[int], bg_color: Tuple[int, int, int], config: TableGenerationConfig):
    
    table_logger.debug(f"Drawing cell: {cell}")

    cell_type = cell.get('cell_type', 'normal_cell')
    table_logger.debug(f"Cell type: {cell_type}")

    is_gray = cell.get('is_gray', False)
    if config.enable_gray_cells and is_gray:
        cell_bg_color = config.get_random_gray_color(bg_color) or bg_color
    else:
        cell_bg_color = bg_color

    # 셀 배경색 정보 저장
    cell['bg_color'] = cell_bg_color
    is_overflow = cell.get('is_overflow', False)

    x1, y1, x2, y2 = cell['x1'], cell['y1'], cell['x2'], cell['y2']

    # 좌표 유효성 검사 및 조정
    x1, x2 = min(x1, x2), max(x1, x2)
    y1, y2 = min(y1, y2), max(y1, y2)
    
    # 오버플로우 처리는 이제 redraw_cell_with_overflow 함수에서 처리됩니다.

    # 최소 크기 보장
    if x2 <= x1:
        x2 = x1 + config.min_cell_width
    if y2 <= y1:
        y2 = y1 + config.min_cell_height

    table_logger.debug(f"Adjusted coordinates: x1={x1}, y1={y1}, x2={x2}, y2={y2}")

    # 셀 배경색 결정
    if config.enable_gray_cells and is_gray:
        cell_bg_color = config.get_random_gray_color(bg_color) or bg_color
    else:
        cell_bg_color = bg_color

    # 셀 그리기
    can_draw_border = (is_gray and (random.random() < config.no_border_gray_cell_probability)) or (not is_gray and random.random() < config.cell_no_border_probability)

    # 셀 배경 그리기
    draw.rectangle([x1, y1, x2, y2], fill=cell_bg_color)

    if can_draw_border or is_overflow:
        line_thickness = config.get_random_line_thickness()
        is_rounded = config.enable_rounded_corners and random.random() < config.rounded_corner_probability
        is_no_side_border = random.random() < config.no_side_borders_cells_probability

        # 테두리 그리기
        if is_rounded:
            corner_radius = random.randint(config.min_corner_radius, config.max_corner_radius)
            draw_rounded_rectangle(draw, [x1, y1, x2, y2], corner_radius, table_bbox,
                                   fill=None, outline=line_color, width=line_thickness)
        elif is_imperfect and is_no_side_border:
            draw_imperfect_cell_border(draw, x1, y1, x2, y2, line_color, line_thickness, config, is_no_side_border)
        else:
            if not is_overflow or cell['overflow']['direction'] == 'up':
                draw.line([(x1, y1), (x2, y1)], fill=line_color, width=line_thickness)
            if not is_overflow or cell['overflow']['direction'] == 'down':
                draw.line([(x1, y2), (x2, y2)], fill=line_color, width=line_thickness)
            if not is_overflow:
                draw.line([(x1, y1), (x1, y2)], fill=line_color, width=line_thickness)
                draw.line([(x2, y1), (x2, y2)], fill=line_color, width=line_thickness)

        # 셀을 아래로 이동한 효과 만들기 (오버플로우가 아닌 경우에만)
        if not cell.get('overflow') and not is_overflow and not cell.get('is_header') and random.random() < config.cell_shift_down_probability:
            offset = random.randint(1, 2)
            
            # 위쪽 선을 배경색으로 덮기
            draw.line([(x1, y1), (x2, y1)], fill=bg_color, width=offset)
            
            # 셀 배경 그리기 (약간 아래로 이동)
            draw.rectangle([x1, y1 + offset, x2, y2], fill=cell_bg_color)
            
            # 아래쪽 선을 셀 색으로 그리기
            draw.line([(x1, y2), (x2, y2)], fill=cell_bg_color, width=offset)

            # 좌우 선 다시 그리기
            draw.line([(x1, y1 + offset), (x1, y2)], fill=line_color, width=line_thickness)
            draw.line([(x2, y1 + offset), (x2, y2)], fill=line_color, width=line_thickness)
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
        if config.enable_imperfect_lines and random.random() < config.imperfect_line_probability[side]:
            draw_imperfect_line(draw, start, end, line_color, line_thickness, config)
            
            
def draw_imperfect_line(draw: ImageDraw.Draw, start: Tuple[int, int], end: Tuple[int, int], 
                        line_color: Tuple[int, int, int], line_thickness: int, config: TableGenerationConfig):
    x1, y1 = start
    x2, y2 = end
    r, g, b = line_color
    # 1. 선의 끊김 효과 (기존 코드)
    if random.random() < config.line_break_probability:
        mid = ((x1 + x2) // 2, (y1 + y2) // 2)
        draw.line([start, mid], fill=line_color, width=line_thickness)
        if random.random() < 0.5:  # 50% 확률로 나머지 반쪽을 그림
            draw.line([mid, end], fill=line_color, width=line_thickness)
    
    # 2. 선의 불규칙한 두께 (기존 코드)
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
    
    # 3. 선의 미세한 굴곡 (기존 코드)
    if random.random() < config.line_curve_probability:
        num_points = 10
        points = []
        for i in range(num_points):
            t = i / (num_points - 1)
            x = int(x1 + t * (x2 - x1) + random.randint(-3, 3))
            y = int(y1 + t * (y2 - y1) + random.randint(-3, 3))
            points.append((x, y))
        draw.line(points, fill=line_color, width=line_thickness)
    
    # 4. 선 끝점의 불완전성 (기존 코드)
    if random.random() < config.end_imperfection_probability:
        for point in [start, end]:
            size = random.randint(1, 3)  # 크기 변화
            offset_x = random.randint(-1, 1)  # x축 오프셋
            offset_y = random.randint(-1, 1)  # y축 오프셋
            draw.ellipse([point[0]-size+offset_x, point[1]-size+offset_y, 
                        point[0]+size+offset_x, point[1]+size+offset_y], 
                        fill=line_color, outline=line_color)
    
    # 5. 모서리 불완전성 (기존 코드)
    if random.random() < config.corner_imperfection_probability:
        corner_size = random.randint(1, 3)
        if (x1, y1) == start:  # 왼쪽 위 또는 왼쪽 아래 모서리
            draw.rectangle([x1-corner_size, y1-corner_size, x1+corner_size, y1+corner_size], fill=line_color)
        else:  # 오른쪽 위 또는 오른쪽 아래 모서리
            draw.rectangle([x2-corner_size, y2-corner_size, x2+corner_size, y2+corner_size], fill=line_color)

    # 6. 선의 흐림 효과 (기존 코드)
    if random.random() < config.line_blur_probability:
        line_image = Image.new('RGBA', (max(x1, x2) - min(x1, x2) + 10, max(y1, y2) - min(y1, y2) + 10), (0, 0, 0, 0))
        line_draw = ImageDraw.Draw(line_image)
        line_draw.line([(5, 5), (line_image.width-5, line_image.height-5)], fill=line_color, width=line_thickness)
        blurred_line = line_image.filter(ImageFilter.GaussianBlur(radius=1))
        draw.bitmap((min(x1, x2)-5, min(y1, y2)-5), blurred_line)

    # 7. 선의 투명도 변화 (기존 코드)
    if random.random() < config.transparency_variation_probability:
        transparent_line = Image.new('RGBA', (max(x1, x2) - min(x1, x2) + 10, max(y1, y2) - min(y1, y2) + 10), (0, 0, 0, 0))
        transparent_draw = ImageDraw.Draw(transparent_line)
        r, g, b = line_color
        transparent_draw.line([(5, 5), (transparent_line.width-5, transparent_line.height-5)], 
                              fill=(r, g, b, random.randint(128, 255)), width=line_thickness)
        draw.bitmap((min(x1, x2)-5, min(y1, y2)-5), transparent_line, fill=None)

    # 8. 점선 효과 (새로운 코드)
    if random.random() < config.dotted_line_probability:
        dash_length = random.randint(2, 5)
        gap_length = random.randint(2, 5)
        dx, dy = x2 - x1, y2 - y1
        distance = (dx**2 + dy**2)**0.5
        num_dashes = int(distance / (dash_length + gap_length))
        for i in range(num_dashes):
            start_t = i * (dash_length + gap_length) / distance
            end_t = start_t + dash_length / distance
            start_x, start_y = int(x1 + dx * start_t), int(y1 + dy * start_t)
            end_x, end_y = int(x1 + dx * end_t), int(y1 + dy * end_t)
            draw.line([(start_x, start_y), (end_x, end_y)], fill=line_color, width=line_thickness)

    # 9. 선의 색상 변화 (새로운 코드)
    if random.random() < config.color_variation_probability:
        num_segments = random.randint(3, 7)
        for i in range(num_segments):
            segment_start = (
                int(x1 + (x2 - x1) * i / num_segments),
                int(y1 + (y2 - y1) * i / num_segments)
            )
            segment_end = (
                int(x1 + (x2 - x1) * (i + 1) / num_segments),
                int(y1 + (y2 - y1) * (i + 1) / num_segments)
            )
            r, g, b = line_color
            variation = random.randint(-20, 20)
            segment_color = (max(0, min(255, r + variation)),
                             max(0, min(255, g + variation)),
                             max(0, min(255, b + variation)))
            draw.line([segment_start, segment_end], fill=segment_color, width=line_thickness)

    # 10. 선의 텍스처 효과 (수정된 코드)
    if random.random() < config.texture_effect_probability:
        texture_line = Image.new('RGBA', (max(x1, x2) - min(x1, x2) + 10, max(y1, y2) - min(y1, y2) + 10), (0, 0, 0, 0))
        texture_draw = ImageDraw.Draw(texture_line)
        texture_draw.line([(5, 5), (texture_line.width-5, texture_line.height-5)], fill=line_color, width=line_thickness)
        for _ in range(100):
            x = random.randint(0, texture_line.width-1)
            y = random.randint(0, texture_line.height-1)
            if texture_line.getpixel((x, y))[3] > 0:  # 알파 채널이 0보다 큰 경우 (선이 있는 부분)
                texture_line.putpixel((x, y), (random.randint(max(0, r-30), min(255, r+30)),
                                               random.randint(max(0, g-30), min(255, g+30)),
                                               random.randint(max(0, b-30), min(255, b+30)),
                                               255))
        draw.bitmap((min(x1, x2)-5, min(y1, y2)-5), texture_line, fill=None)

    
def redraw_cell_with_overflow(draw: ImageDraw.Draw, cell: dict, line_color: Tuple[int, int, int], 
                              table_bbox: List[int], bg_color: Tuple[int, int, int], config: TableGenerationConfig) -> None:
    x1, y1, x2, y2 = cell['x1'], cell['y1'], cell['x2'], cell['y2']
    overflow = cell['overflow']
    direction = overflow['direction']
    overflow_y1 = cell.get('overflow_y1', y1)
    overflow_y2 = cell.get('overflow_y2', y2)
    
    cell_bg_color = cell.get('bg_color', bg_color)
    line_thickness = config.get_random_line_thickness()

    # 오버플로우 영역 전체를 셀 배경색으로 완전히 덮어씌우기
    draw.rectangle([x1, overflow_y1, x2, overflow_y2], fill=cell_bg_color)

    # 오버플로우 영역 테두리 그리기
    draw.line([(x1, overflow_y1), (x2, overflow_y1)], fill=line_color, width=line_thickness)
    draw.line([(x1, overflow_y2), (x2, overflow_y2)], fill=line_color, width=line_thickness)
    draw.line([(x1, overflow_y1), (x1, overflow_y2)], fill=line_color, width=line_thickness)
    draw.line([(x2, overflow_y1), (x2, overflow_y2)], fill=line_color, width=line_thickness)

    # 원래 셀 영역의 테두리 다시 그리기 (오버플로우 방향에 따라)
    if direction in ['down', 'both']:
        draw.line([(x1, y1), (x2, y1)], fill=line_color, width=line_thickness)
    if direction in ['up', 'both']:
        draw.line([(x1, y2), (x2, y2)], fill=line_color, width=line_thickness)
    draw.line([(x1, y1), (x1, y2)], fill=line_color, width=line_thickness)
    draw.line([(x2, y1), (x2, y2)], fill=line_color, width=line_thickness)

    # 병합된 셀의 경우 내부 선 처리
    if cell.get('is_merged', False):
        merge_rows = cell.get('merge_rows', 1)
        merge_cols = cell.get('merge_cols', 1)
        cell_width = (x2 - x1) // merge_cols
        cell_height = (y2 - y1) // merge_rows

        for row in range(1, merge_rows):
            y = y1 + row * cell_height
            if y > overflow_y1 and y < overflow_y2:
                draw.line([(x1, y), (x2, y)], fill=cell_bg_color, width=line_thickness)

        for col in range(1, merge_cols):
            x = x1 + col * cell_width
            draw.line([(x, overflow_y1), (x, overflow_y2)], fill=cell_bg_color, width=line_thickness)


def draw_outer_border(draw: ImageDraw.Draw, table_bbox: List[int], line_color: Tuple[int, int, int]) -> None:
    if random.random() > config.outer_border_probability:
        outer_line_thickness = random.randint(config.min_outer_line_thickness, config.max_outer_line_thickness)
        draw.rectangle(table_bbox, outline=line_color, width=outer_line_thickness)

def draw_divider_lines(draw: ImageDraw.Draw, cells: List[dict], 
                       line_color: Tuple[int, int, int], config: TableGenerationConfig):
    rows = sorted(set(cell['row'] for cell in cells))
    cols = sorted(set(cell['col'] for cell in cells))
    
    def has_special_cells(cell_list):
        return any('is_merged' in cell and cell['is_merged'] for cell in cell_list) or \
               any('overflow' in cell and cell['overflow'] for cell in cell_list)

    # 수평 구분선
    for i in range(1, len(rows)):
        current_row = rows[i]
        previous_row = rows[i-1]
        
        # 현재 행과 이전 행의 셀들 확인
        current_row_cells = [cell for cell in cells if cell['row'] == current_row]
        previous_row_cells = [cell for cell in cells if cell['row'] == previous_row]
        
        if not has_special_cells(current_row_cells + previous_row_cells) and random.random() < config.horizontal_divider_probability:
            y = (max(cell['y1'] for cell in current_row_cells) + 
                 min(cell['y2'] for cell in previous_row_cells)) // 2
            
            # 실제 열이 존재하는 구간에서만 선 그리기
            for col in cols:
                col_cells = [cell for cell in current_row_cells + previous_row_cells if cell['col'] == col]
                if col_cells and not has_special_cells(col_cells):
                    x_start = min(cell['x1'] for cell in col_cells)
                    x_end = max(cell['x2'] for cell in col_cells)
                    thickness = random.randint(*config.divider_line_thickness_range)
                    draw_imperfect_line(draw, (x_start, y), (x_end, y), line_color, thickness, config)
    
    # 수직 구분선
    for i in range(1, len(cols)):
        current_col = cols[i]
        previous_col = cols[i-1]
        
        # 현재 열과 이전 열의 셀들 확인
        current_col_cells = [cell for cell in cells if cell['col'] == current_col]
        previous_col_cells = [cell for cell in cells if cell['col'] == previous_col]
        
        if not has_special_cells(current_col_cells + previous_col_cells) and random.random() < config.vertical_divider_probability:
            x = (max(cell['x1'] for cell in current_col_cells) + 
                 min(cell['x2'] for cell in previous_col_cells)) // 2
            
            # 실제 행이 존재하는 구간에서만 선 그리기
            for row in rows:
                row_cells = [cell for cell in current_col_cells + previous_col_cells if cell['row'] == row]
                if row_cells and not has_special_cells(row_cells):
                    y_start = min(cell['y1'] for cell in row_cells)
                    y_end = max(cell['y2'] for cell in row_cells)
                    thickness = random.randint(*config.divider_line_thickness_range)
                    draw_imperfect_line(draw, (x, y_start), (x, y_end), line_color, thickness, config)

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
        # 중앙 직사각형
        draw.rectangle([x1 + corner_radius, y1, x2 - corner_radius, y2], fill=fill)
        # 좌우 직사각형
        draw.rectangle([x1, y1 + corner_radius, x2, y2 - corner_radius], fill=fill)
        
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
