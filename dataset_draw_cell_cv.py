import cv2
import numpy as np
import random

def draw_cell(img, cell: dict, line_color: Tuple[int, int, int], 
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

    # OpenCV color format (BGR)
    cv_bg_color = (cell_bg_color[2], cell_bg_color[1], cell_bg_color[0])
    cv_line_color = (line_color[2], line_color[1], line_color[0])

    # 셀 그리기
    can_draw_border = not (is_gray and random.random() < config.no_border_gray_cell_probability)

    # 셀 배경 그리기
    cv2.rectangle(img, (x1, y1), (x2, y2), cv_bg_color, -1)

    if can_draw_border:
        line_thickness = config.get_random_line_thickness()
        is_rounded = config.enable_rounded_corners and random.random() < config.rounded_corner_probability
        is_no_side_border = random.random() < config.no_side_borders_cells_probability

        # 테두리 그리기
        if is_rounded:
            corner_radius = random.randint(config.min_corner_radius, config.max_corner_radius)
            draw_rounded_rectangle_cv(img, (x1, y1, x2, y2), corner_radius, cv_line_color, line_thickness)
        elif is_imperfect and is_no_side_border:
            draw_imperfect_cell_border_cv(img, x1, y1, x2, y2, cv_line_color, line_thickness, config, is_no_side_border)
        else:
            if not is_overflow or cell['overflow']['direction'] == 'up':
                cv2.line(img, (x1, y1), (x2, y1), cv_line_color, line_thickness)
            if not is_overflow or cell['overflow']['direction'] == 'down':
                cv2.line(img, (x1, y2), (x2, y2), cv_line_color, line_thickness)
            if not is_overflow:
                cv2.line(img, (x1, y1), (x1, y2), cv_line_color, line_thickness)
                cv2.line(img, (x2, y1), (x2, y2), cv_line_color, line_thickness)

        # 셀을 아래로 이동한 효과 만들기 (오버플로우가 아닌 경우에만)
        if not cell.get('overflow') and not is_overflow and not cell.get('is_header') and random.random() < config.cell_shift_down_probability:
            offset = random.randint(1, 2)
            
            # 위쪽 선을 배경색으로 덮기
            cv2.line(img, (x1, y1), (x2, y1), (bg_color[2], bg_color[1], bg_color[0]), offset)
            
            # 셀 배경 그리기 (약간 아래로 이동)
            cv2.rectangle(img, (x1, y1 + offset), (x2, y2), cv_bg_color, -1)
            
            # 아래쪽 선을 셀 색으로 그리기
            cv2.line(img, (x1, y2), (x2, y2), cv_bg_color, offset)

            # 좌우 선 다시 그리기
            cv2.line(img, (x1, y1 + offset), (x1, y2), cv_line_color, line_thickness)
            cv2.line(img, (x2, y1 + offset), (x2, y2), cv_line_color, line_thickness)

    table_logger.debug(f"Finished drawing cell: {cell}")
    return cell_bg_color

# 추가로 필요한 함수들
def draw_rounded_rectangle_cv(img, rect, corner_radius, color, thickness):
    x1, y1, x2, y2 = rect
    
    # Draw main rectangle
    cv2.rectangle(img, (x1+corner_radius, y1), (x2-corner_radius, y2), color, thickness)
    cv2.rectangle(img, (x1, y1+corner_radius), (x2, y2-corner_radius), color, thickness)
    
    # Draw corner circles
    cv2.circle(img, (x1+corner_radius, y1+corner_radius), corner_radius, color, thickness)
    cv2.circle(img, (x2-corner_radius, y1+corner_radius), corner_radius, color, thickness)
    cv2.circle(img, (x1+corner_radius, y2-corner_radius), corner_radius, color, thickness)
    cv2.circle(img, (x2-corner_radius, y2-corner_radius), corner_radius, color, thickness)

def draw_imperfect_cell_border_cv(img, x1, y1, x2, y2, color, thickness, config, no_side_borders):
    sides = [
        ('top', [(x1, y1), (x2, y1)]),
        ('bottom', [(x1, y2), (x2, y2)]),
        ('left', [(x1, y1), (x1, y2)]),
        ('right', [(x2, y1), (x2, y2)])
    ]

    for side, (start, end) in sides:
        if no_side_borders and side in ['left', 'right']:
            continue
        if config.enable_imperfect_lines and random.random() < config.imperfect_line_probability[side]:
            draw_imperfect_line_cv(img, start, end, color, thickness, config)
