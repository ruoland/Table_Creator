import random
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os
import json
import math

# 상수 정의
MIN_IMAGE_WIDTH, MAX_IMAGE_WIDTH = 500, 800
MIN_IMAGE_HEIGHT, MAX_IMAGE_HEIGHT = 500, 800
MIN_COLS, MAX_COLS = 3, 5
MIN_ROWS, MAX_ROWS = 3, 5
STYLES = ['thin', 'medium', 'thick', 'double']
FONTS = ['font/NanumGothic.ttf', 'font/SANGJU Dajungdagam.ttf', 'font/SOYO Maple Regular.ttf']
COLORS = {'white': 255, 'light_gray': 220, 'dark_gray': 64, 'black': 0}

def create_cell(x, y, width, height, text, merged_rows=1, merged_cols=1):
    return {
        'x': x,
        'y': y,
        'width': width,
        'height': height,
        'text': text,
        'merged_rows': merged_rows,
        'merged_cols': merged_cols
    }


def generate_table(width, height, rows, cols):
    cell_width = width // cols
    cell_height = height // rows
    cells = []

    for i in range(rows):
        for j in range(cols):
            if any(cell['x'] == j * cell_width and cell['y'] == i * cell_height for cell in cells):
                continue  # 이미 병합된 셀

            merged_rows = random.randint(1, min(2, rows - i)) if random.random() < 0.1 else 1
            merged_cols = random.randint(1, min(2, cols - j)) if random.random() < 0.1 else 1

            x = j * cell_width
            y = i * cell_height
            w = cell_width * merged_cols
            h = cell_height * merged_rows
            
            # 병합된 셀의 크기에 따라 텍스트 길이 조정
            text_length = min(3, max(1, (merged_rows * merged_cols)))
            text = generate_random_korean_text(text_length)

            cells.append(create_cell(x, y, w, h, text, merged_rows, merged_cols))

    return cells

def generate_random_korean_text(length):
    chars = "가나다라마바사아자차카타파하"
    return ''.join(random.choice(chars) for _ in range(length))

def get_line_width(style):
    if style == 'thin':
        return 1
    elif style == 'medium':
        return 2
    elif style == 'thick':
        return 3
    else:  # double
        return 4
from PIL import ImageFont, ImageDraw
 
def adjust_font_size(draw, text, max_width, max_height, font_path, start_size=30):
    font_size = min(start_size, max_width // len(text), max_height)
    font = ImageFont.truetype(font_path, font_size)
    while font_size > 8:
        left, top, right, bottom = font.getbbox(text)
        if right - left <= max_width and bottom - top <= max_height:
            break
        font_size -= 1
        font = ImageFont.truetype(font_path, font_size)
    return font
def draw_table(image, cells, line_color, style):
    draw = ImageDraw.Draw(image)
    line_width = get_line_width(style)
    
    for cell in cells:
        x, y, w, h = cell['x'], cell['y'], cell['width'], cell['height']
        
        # 셀 테두리 그리기
        draw.rectangle([x, y, x+w, y+h], outline=line_color, width=line_width)
        
        # 텍스트를 위한 여백 설정 (셀 크기의 10%)
        padding = min(w, h) // 10
        max_text_width = w - (2 * padding)
        max_text_height = h - (2 * padding)
        
        # 셀 크기에 따라 텍스트 생성 및 폰트 크기 결정
        cell_area = w * h
        text_length = max(1, min(10, cell_area // 5000))
        base_font_size = max(8, min(50, int(math.sqrt(cell_area) / 5)))
        
        font_path = random.choice(FONTS)
        text = generate_random_korean_text(text_length)
        
        # 폰트 크기 조정 및 텍스트 배치
        font_size = base_font_size
        font = ImageFont.truetype(font_path, font_size)
        
        while True:
            bbox = font.getbbox(text)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            
            if text_width <= max_text_width and text_height <= max_text_height:
                break
            
            if font_size > 8:
                font_size -= 1
                font = ImageFont.truetype(font_path, font_size)
            else:
                text = text[:-1]
                if not text:
                    text = "가"
                    break
        
        # 텍스트 위치 계산 (정확한 중앙 정렬)
        bbox = font.getbbox(text)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        text_x = x + (w - text_width) // 2
        text_y = y + (h - text_height) // 2
        
        # 텍스트 그리기
        draw.text((text_x, text_y), text, fill=line_color, font=font)




def create_label(cell, image_width, image_height):
    x_center = (cell['x'] + cell['width'] / 2) / image_width
    y_center = (cell['y'] + cell['height'] / 2) / image_height
    width = cell['width'] / image_width
    height = cell['height'] / image_height
    return f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"

def rotate_image(image, angle):
    return image.rotate(angle, resample=Image.BICUBIC, expand=True)

def adjust_labels(labels, original_size, new_size, angle):
    adjusted_labels = []
    orig_w, orig_h = original_size
    new_w, new_h = new_size
    angle_rad = math.radians(angle)

    for label in labels:
        parts = label.split()
        x, y, w, h = map(float, parts[1:])
        
        # 회전 중심으로 이동
        x -= 0.5
        y -= 0.5
        
        # 회전 적용
        new_x = x * math.cos(angle_rad) - y * math.sin(angle_rad)
        new_y = x * math.sin(angle_rad) + y * math.cos(angle_rad)
        
        # 다시 원래 위치로 이동
        new_x += 0.5
        new_y += 0.5
        
        # 새 이미지 크기에 맞게 조정
        new_x *= orig_w / new_w
        new_y *= orig_h / new_h
        w *= orig_w / new_w
        h *= orig_h / new_h
        
        adjusted_labels.append(f"0 {new_x:.6f} {new_y:.6f} {w:.6f} {h:.6f}")

    return adjusted_labels

def generate_dataset(num_images, output_dir):
    os.makedirs(os.path.join(output_dir, 'images'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'labels'), exist_ok=True)

    for i in range(num_images):
        width = random.randint(MIN_IMAGE_WIDTH, MAX_IMAGE_WIDTH)
        height = random.randint(MIN_IMAGE_HEIGHT, MAX_IMAGE_HEIGHT)
        rows = random.randint(MIN_ROWS, MAX_ROWS)
        cols = random.randint(MIN_COLS, MAX_COLS)

        bg_color = random.choice(list(COLORS.values()))
        image = Image.new('L', (width, height), color=bg_color)
        cells = generate_table(width, height, rows, cols)
        
        line_color = COLORS['black'] if bg_color > 128 else COLORS['white']
        style = random.choice(STYLES)
        
        draw_table(image, cells, line_color, style)

        # 회전 적용
        rotation_angle = random.uniform(0, 0)
        rotated_image = rotate_image(image, rotation_angle)

        image_path = os.path.join(output_dir, 'images', f'table_{i:06d}.png')
        label_path = os.path.join(output_dir, 'labels', f'table_{i:06d}.txt')

        rotated_image.save(image_path)

        labels = [create_label(cell, width, height) for cell in cells]
        adjusted_labels = adjust_labels(labels, image.size, rotated_image.size, rotation_angle)

        with open(label_path, 'w') as f:
            for label in adjusted_labels:
                f.write(label + '\n')

        print(f"Generated image and label for table_{i:06d}")

if __name__ == '__main__':
    generate_dataset(100, 'table_dataset')
