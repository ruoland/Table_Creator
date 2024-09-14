import random, os
from tqdm import tqdm
import numpy as np
import logging
import pandas as pd
import csv, json
import yaml


from opti_constants import *

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# LINE_COLORS를 함수로 대체
def get_line_color(bg_color):
    if bg_color > 200:  # 배경색이 밝은 경우 (200 이상을 밝다고 가정)
        return random.randrange(0, 26)  # 어두운 색 (0-25)
    else:  # 배경색이 어두운 경우
        return random.randrange(230, 256)  # 밝은 색 (230-255)

def wrap_text(text, font, max_width):
    lines = []
    # 먼저 '\n'으로 나눕니다.
    paragraphs = text.split('\n')
    for paragraph in paragraphs:
        # 각 단락을 주어진 너비에 맞게 줄바꿈합니다.
        words = paragraph.split()
        current_line = []
        current_width = 0
        for word in words:
            word_bbox = font.getbbox(word + " ")
            word_width = word_bbox[2] - word_bbox[0]
            if current_width + word_width <= max_width:
                current_line.append(word)
                current_width += word_width
            else:
                lines.append(' '.join(current_line))
                current_line = [word]
                current_width = word_width
        lines.append(' '.join(current_line))
    return '\n'.join(lines)
def generate_random_resolution():
    # 최소 및 최대 해상도 설정 (증가된 값)
    min_width, max_width = 1024, 2048
    min_height, max_height = 768, 1536
    
    # 16:9, 4:3, 3:2, 1:1 등의 일반적인 종횡비 중 하나를 선택
    aspect_ratios = [16/9, 4/3, 3/2, 1]
    aspect_ratio = random.choice(aspect_ratios)
    
    # 너비를 먼저 선택하고, 선택된 종횡비에 맞춰 높이 계산
    width = random.randint(min_width, max_width)
    height = int(width / aspect_ratio)
    
    # 높이가 허용 범위를 벗어나면 조정
    if height < min_height:
        height = min_height
        width = int(height * aspect_ratio)
    elif height > max_height:
        height = max_height
        width = int(height * aspect_ratio)
    
    # 8의 배수로 조정 (YOLO 모델의 권장사항)
    width = width - (width % 8)
    height = height - (height % 8)
    
    return (width, height)
def create_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def random_text(min_length=1, max_length=10):
    if random.random() < 0.4:  # 40% 확률로 수업 정보 생성
        return random_class_info()
    elif random.random() < 0.5:  # 50% 확률로 자주 사용되는 단어 선택 (기존 COMMON_WORDS 사용)
        return random.choice(COMMON_WORDS)
    else:  # 10% 확률로 무작위 글자 조합
        return ''.join(random.choice(''.join(COMMON_WORDS)) for _ in range(random.randint(min_length, max_length)))
    
def random_department():
    return random.choice(DEPARTMENTS)

def random_subject():
    return random.choice(SUBJECTS)

def random_professor():
    return random.choice(PROFESSORS) + "교수"

def random_building():
    return random.choice(BUILDINGS)

def random_room():
    return f"{random_building()} {random.randint(1, 5)}0{random.randint(1, 9)}"

def random_time():
    return random.choice(TIMES)

def random_class_type():
    return random.choice(CLASS_TYPES)

def random_academic_term():
    return random.choice(ACADEMIC_TERMS)

def random_event():
    return random.choice(EVENTS)

def random_class_info():
    formats = [
        "{subject}\n{professor}\n{room}",
        "{subject}\n{room}\n{professor}",
        "{subject}\n{time}",
        "{subject}\n{professor}",
        "{subject}\n{class_type}\n{room}",
        "{class_type} {subject}\n{professor}\n{time}",
        "{subject}\n{professor}\n{room}\n{time}",
        "{class_type}\n{subject}\n{room}",
        "{subject} ({class_type})\n{professor}",
        "{subject}\n{room}\n{time}",
        "{department}\n{subject}\n{professor}",
        "{subject}\n{professor}\n{academic_term}",
        "{event}\n{time}\n{room}",
        "{department} {class_type}\n{subject}\n{time}",
        "{subject}\n{professor}\n{department}"
    ]
    
    format_string = random.choice(formats)
    return format_string.format(
        subject=random_subject(),
        professor=random_professor(),
        room=random_room(),
        time=random_time(),
        class_type=random_class_type(),
        department=random_department(),
        academic_term=random_academic_term(),
        event=random_event()
    )
