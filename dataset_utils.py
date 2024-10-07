import random
import os
from typing import List, Tuple, Union, Dict, Any
from dataset_constant import *
from dataset_config import config, TableGenerationConfig

from dataset_constant import *
from logging_config import  get_memory_handler, table_logger

from typing import Tuple

# config 객체를 통해 상수 관리
COLOR_BRIGHTNESS_THRESHOLD = config.color_brightness_threshold
DARK_GRAY_RANGE = config.dark_gray_range
LIGHT_GRAY_RANGE = config.light_gray_range
MEDIUM_GRAY_RANGE = config.medium_gray_range
LIGHT_MEDIUM_GRAY_RANGE = config.light_medium_gray_range
FADED_COLOR_PROBABILITY = config.faded_color_probability
CLASS_INFO_PROBABILITY = config.class_info_probability
COMMON_WORD_PROBABILITY = config.common_word_probability
from typing import Tuple
import random
def get_line_color(bg_color: Tuple[int, int, int], config: TableGenerationConfig) -> Tuple[int, int, int]:
    """배경색에 따라 적절한 선 색상을 반환합니다."""
    try:
        brightness = sum(bg_color) / 3
    except (TypeError, ValueError):
        config.table_logger.warning(f"Invalid background color format: {bg_color}. Using default.")
        return config.line_colors['black']

    # 색상 선택을 위한 가중치 조정
    weights = config.line_color_weights.copy()
    
    # 배경이 어두운 경우 밝은 색상의 가중치를 높임
    if brightness < config.color_brightness_threshold:
        weights['light_gray'] *= 2
        weights['gray'] *= 1.5
        weights['black'] = 0  # 어두운 배경에는 검은색 선을 사용하지 않음
    else:
        weights['dark_gray'] *= 1.5
        weights['black'] *= 2
        weights['light_gray'] = 0  # 밝은 배경에는 밝은 회색 선을 사용하지 않음

    # 가중치에 따라 색상 선택
    available_colors = [color for color, weight in weights.items() if weight > 0]
    color_weights = [weights[color] for color in available_colors]
    
    color_name = random.choices(available_colors, weights=color_weights, k=1)[0]
    
    # 선택된 색상 반환
    line_color = config.line_colors[color_name]
    
    # 배경색과의 대비 확인
    while not is_sufficient_contrast(bg_color, line_color):
        available_colors.remove(color_name)
        if not available_colors:
            config.table_logger.warning("No suitable line color found. Using default contrast color.")
            return get_contrast_color(bg_color)
        color_weights = [weights[color] for color in available_colors]
        color_name = random.choices(available_colors, weights=color_weights, k=1)[0]
        line_color = config.line_colors[color_name]
    

def is_sufficient_contrast(bg_color: Tuple[int, int, int], line_color: Tuple[int, int, int]) -> bool:
    """배경색과 선 색상 간의 대비가 충분한지 확인합니다."""
    bg_brightness = sum(bg_color) / 3
    line_brightness = sum(line_color) / 3
    
    return abs(bg_brightness - line_brightness) > 50  # 임계값은 조정 가능

def get_contrast_color(bg_color: Tuple[int, int, int]) -> Tuple[int, int, int]:
    """배경색과 대비되는 색상을 반환합니다."""
    brightness = sum(bg_color) / 3
    return (0, 0, 0) if brightness > 127 else (255, 255, 255)


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

def generate_random_resolution() -> Tuple[Tuple[int, int], Tuple[int, int, int, int]]:
    """랜덤한 이미지 해상도와 여백을 생성합니다."""
    width = random.randint(config.min_image_width, config.max_image_width)
    height = random.randint(config.min_image_height, config.max_image_height)
    margin_left = random.randint(config.min_margin, config.max_margin)
    margin_right = random.randint(config.min_margin, config.max_margin)
    margin_top = random.randint(config.min_margin, config.max_margin)
    margin_bottom = random.randint(config.min_margin, config.max_margin)
    width = width - (width % 32) + margin_left + margin_right
    height = height - (height % 32) + margin_top + margin_bottom
    return (width, height), (margin_left, margin_top, margin_right, margin_bottom)
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

def validate_cell(cell: Dict[str, Any], config: TableGenerationConfig) -> bool:
    if not all(key in cell for key in ['x1', 'y1', 'x2', 'y2']):
        return False
    x1, y1, x2, y2 = cell['x1'], cell['y1'], cell['x2'], cell['y2']
    return (x2 > x1 and y2 > y1 and 
            x2 - x1 >= config.min_cell_width and y2 - y1 >= config.min_cell_height and
            x2 - x1 <= config.max_cell_width and y2 - y1 <= config.max_cell_height and
            all(isinstance(coord, int) for coord in [x1, y1, x2, y2]))
def validate_color(color: Union[int, Tuple[int, int, int]]) -> Tuple[int, int, int]:
    """색상의 유효성을 검사하고 올바른 형식으로 반환합니다."""
    if isinstance(color, int):
        return (color, color, color)
    elif isinstance(color, tuple) and len(color) == 3:
        return color
    else:
        table_logger.warning(f"Invalid color format: {color}. Using default black.")
        return (0, 0, 0)