import random, os
from tqdm import tqdm
import numpy as np
import logging
import pandas as pd
import csv, json
import yaml
from typing import List, Tuple

from rollback_constant import *
# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
def get_line_color(bg_color: int) -> int:
    """배경색에 따라 적절한 선 색상을 반환합니다."""
    return random.randint(0, 26) if bg_color > 200 else random.randint(230, 256)

def random_text(min_length: int = 1, max_length: int = 10) -> str:
    """랜덤한 텍스트를 생성합니다. 40% 확률로 수업 정보, 50% 확률로 자주 사용되는 단어, 10% 확률로 무작위 글자 조합을 반환합니다."""
    if random.random() < 0.4:
        return random_class_info()
    elif random.random() < 0.5:
        return random.choice(SUBJECTS + DEPARTMENTS + CLASS_TYPES)
    else:
        return ''.join(random.choice('ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789') for _ in range(random.randint(min_length, max_length)))

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

def wrap_text(text: str, font, max_width: int) -> str:
    """주어진 최대 너비에 맞게 텍스트를 줄바꿈합니다."""
    lines = []
    for paragraph in text.split('\n'):
        words = paragraph.split()
        current_line = []
        current_width = 0
        for word in words:
            word_width = font.getbbox(word + " ")[2]
            if current_width + word_width <= max_width:
                current_line.append(word)
                current_width += word_width
            else:
                lines.append(' '.join(current_line))
                current_line = [word]
                current_width = word_width
        lines.append(' '.join(current_line))
    return '\n'.join(lines)

def safe_list_access(lst: List, index: int, default=None):
    """리스트의 안전한 접근을 위한 함수입니다."""
    try:
        return lst[index]
    except IndexError:
        logger.warning(f"Attempted to access index {index} in a list of length {len(lst)}")
        return default

def calculate_summary_stats(stats: List[dict]) -> dict:
    """이미지 생성 통계를 계산합니다."""
    summary = {
        'total_images': len(stats),
        'imperfect_ratio': sum(1 for s in stats if s['is_imperfect']) / len(stats),
        'bg_mode_ratio': {
            'light': sum(1 for s in stats if s['bg_mode'] == 'light') / len(stats),
            'dark': sum(1 for s in stats if s['bg_mode'] == 'dark') / len(stats)
        },
        'avg_image_size': {
            'width': sum(s['image_width'] for s in stats) / len(stats),
            'height': sum(s['image_height'] for s in stats) / len(stats)
        },
        'avg_num_cells': sum(s['num_cells'] for s in stats) / len(stats),
        'avg_num_rows': sum(s['rows'] for s in stats) / len(stats),
        'avg_num_cols': sum(s['cols'] for s in stats) / len(stats)
    }
    return summary
