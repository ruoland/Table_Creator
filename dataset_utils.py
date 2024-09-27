import random, os

import logging

from typing import List, Tuple

from dataset_constant import *
# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def validate_cell(cell):
    return (len(cell) >= 4 and 
            cell[2] > cell[0] and 
            cell[3] > cell[1] and 
            cell[2] - cell[0] >= 1 and
            cell[3] - cell[1] >= 1)
def validate_color(color):
    if isinstance(color, int):
        return (color, color, color)
    elif isinstance(color, tuple) and len(color) == 3:
        return color
    else:
        logger.warning(f"Invalid color format: {color}. Using default black.")
        return (0, 0, 0)


def get_line_color(bg_color):
    if isinstance(bg_color, tuple) and len(bg_color) == 3:
        brightness = sum(bg_color) / 3
    elif isinstance(bg_color, int):
        brightness = bg_color
    else:
        raise ValueError(f"Invalid background color format: {bg_color}")

    if brightness > 128:
        # 밝은 배경: 검은색에서 어두운 회색까지
        gray_value = random.randint(0, 64)
    else:
        # 어두운 배경: 흰색에서 밝은 회색까지
        gray_value = random.randint(192, 255)
    
    # 20% 확률로 약간 희미한 색상 생성
    if random.random() < 0.2:
        if brightness > 128:
            # 밝은 배경에 대해 약간 희미한 회색
            gray_value = random.randint(64, 128)
        else:
            # 어두운 배경에 대해 약간 희미한 밝은 회색
            gray_value = random.randint(128, 192)

    return (gray_value, gray_value, gray_value)

def generate_random_resolution():
    """랜덤한 이미지 해상도 생성 (여백 포함)"""
    width = random.randint(MIN_IMAGE_WIDTH, MAX_IMAGE_WIDTH)
    height = random.randint(MIN_IMAGE_HEIGHT, MAX_IMAGE_HEIGHT)
    margin_left = random.randint(MIN_MARGIN, MAX_MARGIN)
    margin_right = random.randint(MIN_MARGIN, MAX_MARGIN)
    margin_top = random.randint(MIN_MARGIN, MAX_MARGIN)
    margin_bottom = random.randint(MIN_MARGIN, MAX_MARGIN)
    width = width - (width % 32) + margin_left + margin_right
    height = height - (height % 32) + margin_top + margin_bottom
    return (width, height), (margin_left, margin_top, margin_right, margin_bottom)
def count_existing_images(output_dir):
    total_count = 0
    for subset in ['train', 'val']:
        image_dir = os.path.join(output_dir, subset, 'images')
        if os.path.exists(image_dir):
            total_count += len([f for f in os.listdir(image_dir) if f.endswith('.png')])
    return total_count

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



def is_overlapping(area1, area2):
    """두 영역이 겹치는지 확인"""
    return not (area1[2] <= area2[0] or area1[0] >= area2[2] or
                area1[3] <= area2[1] or area1[1] >= area2[3])
    
    

