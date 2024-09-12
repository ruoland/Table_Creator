import os
import random
from tqdm import tqdm
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageEnhance
from concurrent.futures import ProcessPoolExecutor
import yaml

import cv2
import textwrap
# 상수 정의
MIN_COLS, MAX_COLS = 3, 10
MIN_ROWS, MAX_ROWS = 3, 10
BASE_CELL_WIDTH, BASE_CELL_HEIGHT = 50, 40
MIN_CELL_SIZE = 40
MIN_IMAGE_WIDTH, MIN_IMAGE_HEIGHT = 300, 500
MAX_IMAGE_WIDTH, MAX_IMAGE_HEIGHT = 1200, 1200
STYLES = ['thin', 'medium', 'thick', 'double']
FONTS = ['font/NanumGothic.ttf', 'font/SANGJU Dajungdagam.ttf', 'font/SOYO Maple Regular.ttf']

# 그레이스케일 색상 정의
BACKGROUND_COLORS = {
    'light': {'white': 255, 'light_gray': 250},
    'dark': {'black': 0, 'dark_gray': 25}
}

LINE_COLORS = {
    'light': {'black': 0, 'dark_gray': 25},
    'dark': {'white': 255, 'light_gray': 250}
}

def create_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def random_text(min_length=1, max_length=10):
    return ''.join(random.choice('가나다라마바사아자차카타파하') for _ in range(random.randint(min_length, max_length)))
def add_text_to_cells(draw, cells, font_path, bg_mode):
    for cell in cells:
        cell_width = cell[2] - cell[0]
        cell_height = cell[3] - cell[1]
        
        # 셀 크기에 따라 최소 및 최대 폰트 크기 설정
        min_font_size = max(8, int(min(cell_width, cell_height) * 0.1))
        max_font_size = max(12, int(min(cell_width, cell_height) * 0.3))
        
        # 랜덤하게 줄 수 결정 (1~2줄)
        num_lines = random.randint(1, 2)
        
        # 각 줄에 대한 텍스트 생성
        lines = [random_text(1, 5) for _ in range(num_lines)]
        text = '\n'.join(lines)
        
        # 폰트 크기 조정
        font = adjust_font_size(font_path, text, cell_width * 0.9, cell_height * 0.9, min_font_size, max_font_size)
        
        # 텍스트 줄바꿈
        wrapped_lines = []
        for line in lines:
            wrapped_lines.extend(textwrap.wrap(line, width=max(1, int(cell_width * 0.9 / font.getbbox('가')[2]))))
        
        # 전체 텍스트 높이 계산
        line_spacing = font.size * 0.2  # 줄 간격을 폰트 크기의 20%로 설정
        total_text_height = sum(font.getbbox(line)[3] - font.getbbox(line)[1] for line in wrapped_lines) + line_spacing * (len(wrapped_lines) - 1)
        
        # 텍스트 시작 위치 계산
        x = cell[0] + cell_width * 0.05  # 셀 왼쪽에서 5% 떨어진 위치
        y = cell[1] + (cell_height - total_text_height) / 2  # 세로 중앙 정렬
        
        # 텍스트 색상 선택
        text_color = random.choice(list(LINE_COLORS[bg_mode].values()))
        
        # 텍스트 그리기
        for line in wrapped_lines:
            bbox = font.getbbox(line)
            line_width = bbox[2] - bbox[0]
            line_height = bbox[3] - bbox[1]
            
            # 가로 중앙 정렬을 위한 x 좌표 계산
            text_x = cell[0] + (cell_width - line_width) / 2
            
            draw.text((text_x, y), line, font=font, fill=text_color)
            y += line_height + line_spacing  # 다음 줄로 이동

def adjust_font_size(font_path, text, max_width, max_height, min_size, max_size):
    font_size = max_size
    font = ImageFont.truetype(font_path, font_size)
    lines = text.split('\n')
    
    while font_size > min_size:
        if all(font.getbbox(line)[2] <= max_width for line in lines) and \
           sum(font.getbbox(line)[3] - font.getbbox(line)[1] for line in lines) <= max_height:
            break
        font_size -= 1
        font = ImageFont.truetype(font_path, font_size)
    
    return font
def create_table(image_width, image_height, has_gap=False):
    cols = random.randint(MIN_COLS, MAX_COLS)
    rows = random.randint(MIN_ROWS, MAX_ROWS)
    
    cell_width = image_width // cols
    cell_height = image_height // rows
    
    gap = random.randint(1, 3) if has_gap else 0
    
    cells = []
    for row in range(rows):
        for col in range(cols):
            x1 = col * cell_width
            y1 = row * cell_height
            x2 = (col + 1) * cell_width
            y2 = (row + 1) * cell_height
            
            if has_gap:
                x1 += gap
                y1 += gap
                x2 -= gap
                y2 -= gap
            
            cells.append([x1, y1, x2, y2, row, col])
    
    return cells, [0, 0, image_width, image_height], gap, rows, cols

def draw_line(draw, start, end, color, thickness):
    x1, y1 = start
    x2, y2 = end
    
    # 항상 최소 1픽셀 두께로 선을 그립니다.
    draw.line([start, end], fill=color, width=max(1, int(thickness)))
    
    # 두께가 1보다 큰 경우 추가적인 픽셀을 그려 선을 더 굵게 만듭니다.
    if thickness > 1:
        for i in range(1, int(thickness)):
            if x1 == x2:  # 수직선
                draw.line([(x1+i, y1), (x2+i, y2)], fill=color, width=1)
                draw.line([(x1-i, y1), (x2-i, y2)], fill=color, width=1)
            else:  # 수평선
                draw.line([(x1, y1+i), (x2, y2+i)], fill=color, width=1)
                draw.line([(x1, y1-i), (x2, y2-i)], fill=color, width=1)
def draw_table(draw, cells, table_bbox, line_colors, has_gap):
    line_widths = []
    
    # 외곽선 그리기
    outer_line_thickness = random.uniform(1.0, 3.0)
    corners = [
        (table_bbox[0], table_bbox[1]),
        (table_bbox[2], table_bbox[1]),
        (table_bbox[2], table_bbox[3]),
        (table_bbox[0], table_bbox[3])
    ]
    for i in range(4):
        start = corners[i]
        end = corners[(i+1)%4]
        draw_line(draw, start, end, random.choice(line_colors), outer_line_thickness)
        line_widths.append(outer_line_thickness)
    
    # 내부 셀 그리기
    for cell in cells:
        line_thickness = random.uniform(0.5, 2.0)
        if has_gap:
            draw_line(draw, (cell[0], cell[1]), (cell[2], cell[1]), random.choice(line_colors), line_thickness)  # 상단
            draw_line(draw, (cell[0], cell[3]), (cell[2], cell[3]), random.choice(line_colors), line_thickness)  # 하단
            draw_line(draw, (cell[0], cell[1]), (cell[0], cell[3]), random.choice(line_colors), line_thickness)  # 좌측
            draw_line(draw, (cell[2], cell[1]), (cell[2], cell[3]), random.choice(line_colors), line_thickness)  # 우측
        else:
            # gap이 없는 경우, 오른쪽과 아래쪽 선만 그립니다 (겹치는 선 방지)
            draw_line(draw, (cell[2], cell[1]), (cell[2], cell[3]), random.choice(line_colors), line_thickness)  # 우측
            draw_line(draw, (cell[0], cell[3]), (cell[2], cell[3]), random.choice(line_colors), line_thickness)  # 하단
        line_widths.extend([line_thickness] * 4)
    
    return line_widths
def generate_image_and_labels(image_id, resolution, bg_mode, has_gap):
    image_width, image_height = resolution
    
    bg_color = random.choice(list(BACKGROUND_COLORS[bg_mode].values()))
    
    img = Image.new('L', (image_width, image_height), color=bg_color)
    draw = ImageDraw.Draw(img)
    
    cells, table_bbox, gap, rows, cols = create_table(image_width, image_height, has_gap)
    line_colors = list(LINE_COLORS[bg_mode].values())
    
    line_widths = draw_table(draw, cells, table_bbox, line_colors, has_gap)
    add_text_to_cells(draw, cells, random.choice(FONTS), bg_mode)
    
    # 이미지 대비 조정
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(random.uniform(1.2, 1.5))
    
    # YOLO 형식의 레이블 생성 (확장된 형식)
    labels = []
    for cell in cells:
        x_center = (cell[0] + cell[2]) / 2 / image_width
        y_center = (cell[1] + cell[3]) / 2 / image_height
        width = (cell[2] - cell[0]) / image_width
        height = (cell[3] - cell[1]) / image_height
        row, col = cell[4], cell[5]
        labels.append(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f} {row} {col}")
    
    # 전체 표에 대한 레이블
    labels.append(f"1 0.5 0.5 1.0 1.0 {rows} {cols}")
    
    # 통계 수집
    stats = {
        'image_id': image_id,
        'bg_mode': bg_mode,
        'has_gap': has_gap,
        'gap_size': gap,
        'num_cells': len(cells),
        'rows': rows,
        'cols': cols,
        'table_width': table_bbox[2] - table_bbox[0],
        'table_height': table_bbox[3] - table_bbox[1],
        'avg_cell_width': np.mean([cell[2] - cell[0] for cell in cells]),
        'avg_cell_height': np.mean([cell[3] - cell[1] for cell in cells]),
        'min_cell_width': min([cell[2] - cell[0] for cell in cells]),
        'min_cell_height': min([cell[3] - cell[1] for cell in cells]),
        'max_cell_width': max([cell[2] - cell[0] for cell in cells]),
        'max_cell_height': max([cell[3] - cell[1] for cell in cells]),
        'avg_line_width': np.mean(line_widths),
        'min_line_width': min(line_widths),
        'max_line_width': max(line_widths),
    }
    
    return img, labels, stats

def save_image_and_labels(result, image_dir, label_dir):
    img, labels, image_id = result
    img.save(os.path.join(image_dir, f"{image_id}.png"))
    with open(os.path.join(label_dir, f"{image_id}.txt"), 'w') as f:
        f.write('\n'.join(labels))
def generate_single_image_and_labels(args):
    image_id, resolution, bg_mode, has_gap = args
    try:
        return generate_image_and_labels(image_id, resolution, bg_mode, has_gap)
    except Exception as e:
        print(f"Error generating image {image_id}: {str(e)}")
        return None

import pandas as pd

def generate_dataset(light_images_with_gap, light_images_without_gap, 
                     dark_images_with_gap, dark_images_without_gap, 
                     output_dir, resolution):
    create_directory(output_dir)
    light_image_dir = os.path.join(output_dir, 'light', 'images')
    light_label_dir = os.path.join(output_dir, 'light', 'labels')
    dark_image_dir = os.path.join(output_dir, 'dark', 'images')
    dark_label_dir = os.path.join(output_dir, 'dark', 'labels')
    
    create_directory(light_image_dir)
    create_directory(light_label_dir)
    create_directory(dark_image_dir)
    create_directory(dark_label_dir)
    
    stats = []
    
    image_configs = [
        ('light', True, light_images_with_gap),
        ('light', False, light_images_without_gap),
        ('dark', True, dark_images_with_gap),
        ('dark', False, dark_images_without_gap)
    ]
    for bg_mode, has_gap, num_images in image_configs:
        args_list = [(i, resolution, bg_mode, has_gap) for i in range(num_images)]
        
        with ProcessPoolExecutor() as executor:
            results = list(tqdm(executor.map(generate_single_image_and_labels, args_list), 
                                total=num_images, 
                                desc=f"Generating {bg_mode} images ({'with' if has_gap else 'without'} gap)"))
        
        for i, result in enumerate(tqdm(results, desc=f"Saving {bg_mode} images ({'with' if has_gap else 'without'} gap)")):
            if result is not None:
                img, labels, image_stats = result
                if bg_mode == 'light':
                    save_image_and_labels((img, labels, f"light_{i}_{'gap' if has_gap else 'no_gap'}"), light_image_dir, light_label_dir)
                else:
                    save_image_and_labels((img, labels, f"dark_{i}_{'gap' if has_gap else 'no_gap'}"), dark_image_dir, dark_label_dir)
                stats.append(image_stats)
    
    # 통계 저장
    df = pd.DataFrame(stats)
    df.to_csv(os.path.join(output_dir, 'statistics.csv'), index=False)
    
    # 요약 통계 계산 및 저장
    summary_stats = df.describe()
    summary_stats.to_csv(os.path.join(output_dir, 'summary_statistics.csv'))
    
    # YAML 설정 파일 생성
    yaml_content = {
        'train': {'light': './light/images', 'dark': './dark/images'},
        'val': {'light': './light/images', 'dark': './dark/images'},
        'nc': 2,
        'names': ['cell', 'table']
    }
    
    with open(os.path.join(output_dir, 'dataset.yaml'), 'w') as f:
        yaml.dump(yaml_content, f)
    
    print(f"생성된 이미지 수:")
    print(f"밝은 배경 (간격 있음) - {light_images_with_gap}")
    print(f"밝은 배경 (간격 없음) - {light_images_without_gap}")
    print(f"어두운 배경 (간격 있음) - {dark_images_with_gap}")
    print(f"어두운 배경 (간격 없음) - {dark_images_without_gap}")
    print(f"통계 정보가 {output_dir}/statistics.csv 와 {output_dir}/summary_statistics.csv 에 저장되었습니다.")
if __name__ == "__main__":
    light_images_with_gap = 100
    light_images_without_gap = 100
    dark_images_with_gap = 50
    dark_images_without_gap = 50
    output_dir = 'table_dataset'
    resolution = (800, 600)
    generate_dataset(light_images_with_gap, light_images_without_gap,
                     dark_images_with_gap, dark_images_without_gap,
                     output_dir, resolution)