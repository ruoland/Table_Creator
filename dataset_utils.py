import random
import os
from typing import List, Tuple, Union, Dict, Any
from dataset_constant import *
from dataset_config import config, TableGenerationConfig

from dataset_constant import *
from logging_config import  get_memory_handler, table_logger

from typing import Tuple
import random


# config 객체를 통해 상수 관리
COLOR_BRIGHTNESS_THRESHOLD = config.color_brightness_threshold
DARK_GRAY_RANGE = config.dark_gray_range
LIGHT_GRAY_RANGE = config.light_gray_range
MEDIUM_GRAY_RANGE = config.medium_gray_range
LIGHT_MEDIUM_GRAY_RANGE = config.light_medium_gray_range
FADED_COLOR_PROBABILITY = config.faded_color_probability
CLASS_INFO_PROBABILITY = config.class_info_probability
COMMON_WORD_PROBABILITY = config.common_word_probability

def get_line_color(bg_color: Tuple[int, int, int], config: TableGenerationConfig) -> Tuple[int, int, int]:
    """그레이스케일 이미지를 위한 다양한 명도의 선 색상을 반환합니다."""
    try:
        bg_brightness = sum(bg_color) / 3
    except (TypeError, ValueError):
        config.table_logger.warning(f"잘못된 배경색 형식: {bg_color}. 기본값을 사용합니다.")
        return (128, 128, 128)

    # 다양한 명도의 회색 옵션 (0: 검정, 255: 흰색)
    gray_options = [
        ('very_dark', 20),
        ('dark', 50),
        ('medium_dark', 80),
        ('medium', 128),
        ('medium_light', 170),
        ('light', 200),
        ('very_light', 230)
    ]

    # 배경 밝기에 따라 가중치 조정
    weights = []
    for _, gray_value in gray_options:
        contrast = abs(bg_brightness - gray_value)
        weight = min(contrast, 100)  # 최대 가중치를 100으로 제한
        weights.append(weight)

    # 가중치에 따라 회색 선택
    selected_gray = random.choices(gray_options, weights=weights, k=1)[0]
    gray_value = selected_gray[1]

    # 약간의 변동성 추가 (±10)
    gray_value = max(0, min(255, gray_value + random.randint(-10, 10)))

    return (gray_value, gray_value, gray_value)

def is_sufficient_contrast(bg_color: Tuple[int, int, int], line_color: Tuple[int, int, int]) -> bool:
    """배경색과 선 색상 간의 대비가 충분한지 확인합니다."""
    bg_brightness = sum(bg_color) / 3
    line_brightness = sum(line_color) / 3
    
    return abs(bg_brightness - line_brightness) > 50  # 임계값은 조정 가능



def validate_all_cells(cells, table_bbox, stage_name):
    for cell in cells:
        if cell['x1'] >= cell['x2'] or cell['y1'] >= cell['y2']:
            table_logger.warning(f"Invalid cell coordinates detected after {stage_name}: {cell}")
        if cell['x1'] < table_bbox[0] or cell['x2'] > table_bbox[2] or cell['y1'] < table_bbox[1] or cell['y2'] > table_bbox[3]:
            table_logger.warning(f"Cell coordinates out of table bounds after {stage_name}: {cell}")
    return cells
def random_text(min_length: int = 1, max_length: int = 10) -> str:
    """랜덤한 텍스트를 생성합니다."""
    if random.random() < CLASS_INFO_PROBABILITY:
        return random_class_info()
    elif random.random() < COMMON_WORD_PROBABILITY:
        return random.choice(SUBJECTS + DEPARTMENTS + CLASS_TYPES)
    else:
        return ''.join(random.choice('ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789') for _ in range(random.randint(min_length, max_length)))
def random_class_info() -> str:
    """랜덤한 수업 정보를 생성합니다."""
    formats = [
        "{subject}\n{professor}\n{room}",
        "{subject}\n{time}\n{room}",
        "{subject}\n{professor}\n{time}",
        "{class_type}\n{subject}\n{room}",
        "{department}\n{subject}\n{professor}"
    ]
    format_string = random.choice(formats)
    return format_string.format(
        subject=random.choice(SUBJECTS),
        professor=random.choice(PROFESSORS) + "교수",
        room=f"{random.choice(BUILDINGS)} {random.randint(100, 500)}",
        time=random.choice(TIMES),
        class_type=random.choice(CLASS_TYPES),
        department=random.choice(DEPARTMENTS)
    )

def is_overlapping(area1: Tuple[int, int, int, int], area2: Tuple[int, int, int, int]) -> bool:
    """두 영역이 겹치는지 확인합니다."""
    return not (area1[2] <= area2[0] or area1[0] >= area2[2] or
                area1[3] <= area2[1] or area1[1] >= area2[3])

def strict_validate_cell(cell: Dict[str, Any]) -> bool:
    if not isinstance(cell, dict):
        return False
    required_keys = ['x1', 'y1', 'x2', 'y2']
    if not all(key in cell for key in required_keys):
        return False
    if not all(isinstance(cell[key], (int, float)) for key in required_keys):
        return False
    if cell['x2'] <= cell['x1'] or cell['y2'] <= cell['y1']:
        return False
    return True

def validate_color(color: Union[int, Tuple[int, int, int]]) -> Tuple[int, int, int]:
    """색상의 유효성을 검사하고 올바른 형식으로 반환합니다."""
    if isinstance(color, int):
        return (color, color, color)
    elif isinstance(color, tuple) and len(color) == 3:
        return color
    else:
        table_logger.warning(f"Invalid color format: {color}. Using default black.")
        return (0, 0, 0)
    

def adjust_cell_positions(cells, config, table_bbox):
    if not config.enable_overflow:
        return cells
    
    table_logger.warning(f"셀 위치 조정 시작: 총 {len(cells)}개 셀")
    adjusted_count = 0
    
    for cell in cells:
        original_x1, original_x2 = cell['x1'], cell['x2']
        original_y1, original_y2 = cell['y1'], cell['y2']
        
        # 기본 x1 및 x2 좌표 검증 추가 
        cell['x1'] = max(cell['x1'], table_bbox[0])
        cell['x2'] = min(cell['x2'], table_bbox[2])

        # x1이 x2보다 크거나 같은 경우 조정 
        if cell['x1'] >= cell['x2']:
            middle_x = (cell['x1'] + cell['x2']) / 2 
            cell['x1'] = max(middle_x - 1, table_bbox[0]) 
            cell['x2'] = min(middle_x + 1, table_bbox[2]) 
        
        if cell['x1'] != original_x1 or cell['x2'] != original_x2:
            table_logger.warning(f"X 좌표 조정: row={cell['row']}, col={cell['col']}, x1: {original_x1} -> {cell['x1']}, x2: {original_x2} -> {cell['x2']}")

        if cell.get('overflow') and cell.get('overflow_applied', False):
            direction = cell['overflow']['direction']
            height_up = cell['overflow']['height_up']
            height_down = cell['overflow']['height_down']
            
            # 병합된 셀 여부 확인
            is_merged = cell.get('is_merged', False)
            merge_rows = cell.get('merge_rows', 1)
            merge_cols = cell.get('merge_cols', 1)
            
            if direction in ['up', 'both']:
                cell['overflow_y1'] = max(cell['y1'] - height_up, table_bbox[1])
            if direction in ['down', 'both']:
                cell['overflow_y2'] = min(cell['y2'] + height_down, table_bbox[3])
            
            # 실제로 오버플로우가 적용되었는지 다시 확인
            if cell['overflow_y1'] == original_y1 and cell['overflow_y2'] == original_y2:
                table_logger.warning(f"오버플로우 취소: row={cell['row']}, col={cell['col']}, 변화 없음")
                cell.pop('overflow', None)
                cell.pop('overflow_y1', None)
                cell.pop('overflow_y2', None)
                cell.pop('overflow_applied', None)
                if cell['is_merged']:
                    cell['cell_type'] = 'merged_cell'
                else:
                    cell['cell_type'] = 'cell'
            else:
                table_logger.warning(f"오버플로우 적용: row={cell['row']}, col={cell['col']}, direction={direction}, y1: {original_y1} -> {cell['overflow_y1']}, y2: {original_y2} -> {cell['overflow_y2']}")
                adjusted_count += 1 

    for cell in cells:
        if 'overflow_y1' in cell:
            original_y1 = cell['overflow_y1']
            cell['overflow_y1'] = max(min(cell['overflow_y1'], cell['y2']-1), table_bbox[1])
            if cell['overflow_y1'] != original_y1:
                table_logger.warning(f"오버플로우 y1 조정: row={cell['row']}, col={cell['col']}, y1: {original_y1} -> {cell['overflow_y1']}")
        
        if 'overflow_y2' in cell:
            original_y2 = cell['overflow_y2']
            cell['overflow_y2'] = min(max(cell['overflow_y2'], cell['y1']+1), table_bbox[3])
            if cell['overflow_y2'] != original_y2:
                table_logger.warning(f"오버플로우 y2 조정: row={cell['row']}, col={cell['col']}, y2: {original_y2} -> {cell['overflow_y2']}")
        
        # 기본 y 좌표 검증 
        original_y1, original_y2 = cell['y1'], cell['y2']
        cell['y1'] = max(cell['y1'], table_bbox[1])
        cell['y2'] = min(cell['y2'], table_bbox[3])
        
        # y 좌표 조정 
        if cell['y1'] >= cell['y2']:
            middle_y = (cell['y1'] + cell['y2']) / 2 
            cell['y1'] = max(middle_y - 1, table_bbox[1]) 
            cell['y2'] = min(middle_y + 1, table_bbox[3]) 
        
        if cell['y1'] != original_y1 or cell['y2'] != original_y2:
            table_logger.warning(f"Y 좌표 조정: row={cell['row']}, col={cell['col']}, y1: {original_y1} -> {cell['y1']}, y2: {original_y2} -> {cell['y2']}")

        # 최종 검증 
        cell['x1'] = max(min(cell['x1'], table_bbox[2]), table_bbox[0]) 
        cell['x2'] = max(min(cell['x2'], table_bbox[2]), table_bbox[0]) 
        cell['y1'] = max(min(cell['y1'], table_bbox[3]), table_bbox[1]) 
        cell['y2'] = max(min(cell['y2'], table_bbox[3]), table_bbox[1]) 

        # 최소 크기 확인 
        if (cell['x2'] - cell['x1']) < 10 or (cell['y2'] - cell['y1']) < 10:
            table_logger.warning(f"Invalid size after adjustment: {cell}")

    table_logger.warning(f"셀 위치 조정 완료: {adjusted_count}개 셀의 위치가 조정됨")
    return cells
