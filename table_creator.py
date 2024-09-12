import os
import random
import logging
from tqdm import tqdm
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageEnhance, ImageFilter
from concurrent.futures import ProcessPoolExecutor
import yaml
import cv2
import textwrap
import pandas as pd

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 상수 정의
MIN_COLS, MAX_COLS = 3, 10
MIN_ROWS, MAX_ROWS = 3, 10
BASE_CELL_WIDTH, BASE_CELL_HEIGHT = 50, 40
MIN_CELL_SIZE = 40
MIN_IMAGE_WIDTH, MIN_IMAGE_HEIGHT = 300, 500
MAX_IMAGE_WIDTH, MAX_IMAGE_HEIGHT = 1200, 1200
STYLES = ['thin', 'medium', 'thick', 'double']
FONTS = ['font/NanumGothic.ttf', 'font/SANGJU Dajungdagam.ttf', 'font/SOYO Maple Regular.ttf']
LINE_STYLES = ['solid', 'dashed', 'dotted']
BACKGROUND_COLORS = {
    'light': {'white': 255, 'light_gray': 250},
    'dark': {'black': 0, 'dark_gray': 25}
}
LINE_COLORS = {
    'light': {'black': 0, 'dark_gray': 25},
    'dark': {'white': 255, 'light_gray': 250}
}
COMMON_WORDS = [
    '수학', '영어', '국어', '과학', '사회', '체육', '음악', '미술',
    '월', '화', '수', '목', '금', '토', '일',
    '1교시', '2교시', '3교시', '4교시', '5교시', '6교시', '7교시',
    '점심', '휴식', '자습', '회의', '상담',
    '월요일', '화요일', '수요일', '목요일', '금요일', '토요일', '일요일',
    '김', '이', '박', '최', '정', '강', '조', '윤', '장', '임',
    '교실', '강당', '도서관', '운동장', '식당',
    '시험', '과제', '발표', '토론', '실험', '프로젝트',
    '아침', '점심', '저녁', '오전', '오후',
    '학교', '학원', '집', '회사', '병원', '은행', '마트',
    '회의', '미팅', '약속', '일정', '계획', '목표',
    '집에', '가자', '데이터', '공학', '기계', '전기', '전자', '컴퓨터', '소프트웨어',
    '하드웨어', '설계', '실험', '프로젝트', '프로그래밍', '알고리즘', '데이터', '로봇',
    '미술', '디자인', '드로잉', '페인팅', '조각', '일러스트레이션', '색채',
    '구도', '작품', '전시회', '포트폴리오',
    '농업', '식물', '작물', '토양', '비료', '농기계', '재배', '수확',
    '농촌', '환경', '생태계',
    '체육', '운동', '스포츠', '훈련', '경기', '체력', '건강', '스트레칭',
    '팀워크', '대회', '선수', '코치',
]

def create_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def safe_list_access(lst, index, default=None):
    try:
        return lst[index]
    except IndexError:
        logger.warning(f"Attempted to access index {index} in a list of length {len(lst)}")
        return default

def safe_randrange(start, stop):
    if start >= stop:
        logger.warning(f"Invalid range ({start}, {stop})")
        return start
    return random.randrange(start, stop)

def random_text(min_length=1, max_length=10):
    if random.random() < 0.7:  # 70% 확률로 자주 사용되는 단어 선택
        return random.choice(COMMON_WORDS)
    else:  # 30% 확률로 무작위 글자 조합
        return ''.join(random.choice(''.join(COMMON_WORDS)) for _ in range(random.randint(min_length, max_length)))

def add_content_to_cells(draw, cells, font_path, bg_mode, empty_cell_ratio=0.2):
    for cell in cells:
        if random.random() < empty_cell_ratio:
            continue  # 빈 셀로 남김
        
        content_type = random.choice(['text', 'shapes', 'mixed'])
        position = random.choice(['center', 'top_left'])
        
        cell_width = cell[2] - cell[0]
        cell_height = cell[3] - cell[1]
        
        if position == 'center':
            x = cell[0] + cell_width // 2
            y = cell[1] + cell_height // 2
        else:  # top_left
            x = cell[0] + int(cell_width * 0.1)
            y = cell[1] + int(cell_height * 0.1)
        
        if content_type in ['text', 'mixed']:
            add_text_to_cell(draw, cell, font_path, bg_mode, x, y, position)
        if content_type in ['shapes', 'mixed']:
            add_shapes_to_cell(draw, cell, bg_mode, x, y)

def add_text_to_cell(draw, cell, font_path, bg_mode, x, y, position):
    cell_width = cell[2] - cell[0]
    cell_height = cell[3] - cell[1]
    
    font_size = min(int(cell_height * 0.4), int(cell_width * 0.15))
    font = ImageFont.truetype(font_path, font_size)
    
    text = random_text(1, 5)
    text_color = random.choice(list(LINE_COLORS[bg_mode].values()))
    
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    
    if position == 'center':
        text_position = (x - text_width // 2, y - text_height // 2)
    else:  # top_left
        text_position = (x, y)
    
    draw.text(text_position, text, font=font, fill=text_color)
    
    add_noise_around_text(draw, text_position[0], text_position[1], text_width, text_height, bg_mode)

def add_noise_around_text(draw, x, y, width, height, bg_mode):
    noise_color = random.choice(list(LINE_COLORS[bg_mode].values()))
    num_noise = random.randint(5, 20)
    for _ in range(num_noise):
        noise_x = x + random.randint(-width//2, width//2)
        noise_y = y + random.randint(-height//2, height//2)
        noise_size = random.randint(1, 3)
        if random.random() < 0.7:  # 70% 확률로 점
            draw.ellipse([noise_x, noise_y, noise_x+noise_size, noise_y+noise_size], fill=noise_color)
        else:  # 30% 확률로 짧은 선
            end_x = noise_x + random.randint(-5, 5)
            end_y = noise_y + random.randint(-5, 5)
            draw.line([noise_x, noise_y, end_x, end_y], fill=noise_color, width=1)

def add_shapes_to_cell(draw, cell, bg_mode, x, y):
    cell_width = cell[2] - cell[0]
    cell_height = cell[3] - cell[1]
    max_size = min(cell_width, cell_height) // 2
    color = random.choice(list(LINE_COLORS[bg_mode].values()))
    
    num_shapes = random.randint(1, 3)
    for _ in range(num_shapes):
        shape_type = random.choice(['rectangle', 'ellipse', 'triangle'])
        size = random.randint(max_size // 2, max_size)
        angle = random.randint(0, 360)
        
        shape_img = Image.new('L', (size, size), color=0)
        shape_draw = ImageDraw.Draw(shape_img)
        
        if shape_type == 'rectangle':
            shape_draw.rectangle([0, 0, size-1, size-1], outline=255, width=2)
        elif shape_type == 'ellipse':
            shape_draw.ellipse([0, 0, size-1, size-1], outline=255, width=2)
        else:  # triangle
            shape_draw.polygon([(size//2, 0), (0, size-1), (size-1, size-1)], outline=255, width=2)
        
        rotated_shape = shape_img.rotate(angle, expand=True)
        
        paste_x = max(cell[0], min(x - rotated_shape.width // 2, cell[2] - rotated_shape.width))
        paste_y = max(cell[1], min(y - rotated_shape.height // 2, cell[3] - rotated_shape.height))
        
        draw.bitmap((paste_x, paste_y), rotated_shape, fill=color)

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
    
    cells = merge_cells(cells, rows, cols)  # 셀 병합 적용
    
    return cells, [0, 0, image_width, image_height], gap, rows, cols

def draw_line(draw, start, end, color, thickness, style='solid'):
    x1, y1 = start
    x2, y2 = end
    thickness = max(1, int(thickness))

    if style == 'solid':
        draw_solid_line(draw, start, end, color, thickness)
    elif style == 'dashed':
        draw_dashed_line(draw, start, end, color, thickness)
    elif style == 'dotted':
        draw_dotted_line(draw, start, end, color, thickness)

def draw_solid_line(draw, start, end, color, thickness):
    x1, y1 = start
    x2, y2 = end
    
    draw.line([start, end], fill=color, width=thickness)
    
    if thickness > 1:
        for i in range(1, thickness):
            if x1 == x2:  # 수직선
                draw.line([(x1+i, y1), (x2+i, y2)], fill=color, width=1)
                draw.line([(x1-i, y1), (x2-i, y2)], fill=color, width=1)
            else:  # 수평선
                draw.line([(x1, y1+i), (x2, y2+i)], fill=color, width=1)
                draw.line([(x1, y1-i), (x2, y2-i)], fill=color, width=1)

def draw_dashed_line(draw, start, end, color, thickness):
    x1, y1 = start
    x2, y2 = end
    dash_length = thickness * 3
    gap_length = thickness * 2
    
    if x1 == x2:  # 수직선
        y_start, y_end = min(y1, y2), max(y1, y2)
        for y in range(int(y_start), int(y_end), int(dash_length + gap_length)):
            segment_end = min(y + dash_length, y_end)
            draw_solid_line(draw, (x1, y), (x1, segment_end), color, thickness)
    else:  # 수평선
        x_start, x_end = min(x1, x2), max(x1, x2)
        for x in range(int(x_start), int(x_end), int(dash_length + gap_length)):
            segment_end = min(x + dash_length, x_end)
            draw_solid_line(draw, (x, y1), (segment_end, y1), color, thickness)

def draw_dotted_line(draw, start, end, color, thickness):
    x1, y1 = start
    x2, y2 = end
    dot_gap = thickness * 2
    
    if x1 == x2:  # 수직선
        y_start, y_end = min(y1, y2), max(y1, y2)
        for y in range(int(y_start), int(y_end), int(dot_gap)):
            draw.ellipse([(x1 - thickness/2, y - thickness/2), 
                          (x1 + thickness/2, y + thickness/2)], fill=color)
    else:  # 수평선
        x_start, x_end = min(x1, x2), max(x1, x2)
        for x in range(int(x_start), int(x_end), int(dot_gap)):
            draw.ellipse([(x - thickness/2, y1 - thickness/2), 
                          (x + thickness/2, y1 + thickness/2)], fill=color)

def draw_table(draw, cells, table_bbox, line_colors, has_gap):
    line_widths = []
    
    # 외곽선 그리기
    outer_line_thickness = random.uniform(1.0, 3.0)
    outer_line_style = random.choice(LINE_STYLES)
    corners = [
        (table_bbox[0], table_bbox[1]),
        (table_bbox[2], table_bbox[1]),
        (table_bbox[2], table_bbox[3]),
        (table_bbox[0], table_bbox[3])
    ]
    for i in range(4):
        start = corners[i]
        end = corners[(i+1)%4]
        vary_line_quality(draw, start, end, random.choice(line_colors), int(outer_line_thickness))
        line_widths.append(outer_line_thickness)
    
    # 내부 셀 그리기
    for cell in cells:
        line_thickness = random.uniform(0.5, 2.0)
        is_merged = len(cell) > 6

        if has_gap or is_merged:
            vary_line_quality(draw, (cell[0], cell[1]), (cell[2], cell[1]), random.choice(line_colors), int(line_thickness))  # 상단
            vary_line_quality(draw, (cell[0], cell[3]), (cell[2], cell[3]), random.choice(line_colors), int(line_thickness))  # 하단
            vary_line_quality(draw, (cell[0], cell[1]), (cell[0], cell[3]), random.choice(line_colors), int(line_thickness))  # 좌측
            vary_line_quality(draw, (cell[2], cell[1]), (cell[2], cell[3]), random.choice(line_colors), int(line_thickness))  # 우측
        else:
            vary_line_quality(draw, (cell[2], cell[1]), (cell[2], cell[3]), random.choice(line_colors), int(line_thickness))  # 우측
            vary_line_quality(draw, (cell[0], cell[3]), (cell[2], cell[3]), random.choice(line_colors), int(line_thickness))  # 하단

        line_widths.extend([line_thickness] * 4)
    
    return line_widths

def vary_line_quality(draw, start, end, color, thickness):
    if random.random() < 0.1:  # 10% 확률로 선을 약간 흐리게
        blur_radius = random.uniform(0.3, 0.7)
        temp_img = Image.new('L', (draw.im.size[0], draw.im.size[1]), 0)
        temp_draw = ImageDraw.Draw(temp_img)
        temp_draw.line((start, end), fill=255, width=thickness)
        blurred_img = temp_img.filter(ImageFilter.GaussianBlur(radius=blur_radius))
        draw.bitmap((0, 0), blurred_img, fill=color)
    elif random.random() < 0.05:  # 5% 확률로 선을 약간 끊기
        segments = random.randint(2, 3)
        for i in range(segments):
            seg_start = (int(start[0] + (end[0]-start[0])*i/segments), int(start[1] + (end[1]-start[1])*i/segments))
            seg_end = (int(start[0] + (end[0]-start[0])*(i+0.9)/segments), int(start[1] + (end[1]-start[1])*(i+0.9)/segments))
            draw.line((seg_start, seg_end), fill=color, width=thickness)
    else:
        draw.line((start, end), fill=color, width=thickness)

    # 선 주변에 약간의 노이즈 추가
    if random.random() < 0.2:  # 20% 확률로 노이즈 추가
        noise_intensity = 0.1
        length = int(((end[0]-start[0])**2 + (end[1]-start[1])**2)**0.5)
        for _ in range(int(length * noise_intensity)):
            noise_x = random.randint(min(start[0], end[0]), max(start[0], end[0]))
            noise_y = random.randint(min(start[1], end[1]), max(start[1], end[1]))
            draw.point((noise_x, noise_y), fill=color)

def merge_cells(cells, rows, cols):
    merged_cells = cells.copy()
    num_merges = random.randint(1, min(rows, cols) // 2)
    for _ in range(num_merges):
        start_row = random.randint(0, rows - 2)
        start_col = random.randint(0, cols - 2)
        merge_rows = random.randint(2, min(3, rows - start_row))
        merge_cols = random.randint(2, min(3, cols - start_col))
        
        # 이미 병합된 셀과 겹치는지 확인
        if any(cell is None for cell in merged_cells[start_row*cols + start_col : (start_row+merge_rows)*cols + start_col + merge_cols : cols]):
            continue
        
        new_cell = [
            cells[start_row * cols + start_col][0],
            cells[start_row * cols + start_col][1],
            cells[(start_row + merge_rows - 1) * cols + (start_col + merge_cols - 1)][2],
            cells[(start_row + merge_rows - 1) * cols + (start_col + merge_cols - 1)][3],
            start_row, start_col, merge_rows, merge_cols
        ]
        
        for r in range(start_row, start_row + merge_rows):
            for c in range(start_col, start_col + merge_cols):
                merged_cells[r * cols + c] = None
        
        merged_cells[start_row * cols + start_col] = new_cell
    
    return [cell for cell in merged_cells if cell is not None]

def create_imperfect_table(width, height, rows, cols, empty_cell_ratio=0.2):
    bg_color = random.choice(list(BACKGROUND_COLORS['light'].values()))
    image = Image.new('L', (width, height), color=bg_color)
    draw = ImageDraw.Draw(image)
    
    # 기본 선 그리기
    for i in range(rows + 1):
        y = int(i * height / rows)
        thickness = random.randint(1, 4)
        vary_line_quality(draw, (0, y), (width, y), 0, thickness)
    
    for i in range(cols + 1):
        x = int(i * width / cols)
        thickness = random.randint(1, 4)
        vary_line_quality(draw, (x, 0), (x, height), 0, thickness)
    
    # 셀 내용 추가
    cells = create_table(width, height, False)[0]
    font = ImageFont.truetype(random.choice(FONTS), 12)
    for cell in cells:
        if random.random() > empty_cell_ratio:
            content_type = random.choice(['text', 'shape', 'noise'])
            if content_type == 'text':
                text = random_text(1, 5)
                x = (cell[0] + cell[2]) // 2
                y = (cell[1] + cell[3]) // 2
                draw.text((x, y), text, fill=0, font=font, anchor="mm")
            elif content_type == 'shape':
                draw.rectangle([cell[0], cell[1], cell[2], cell[3]], outline=0)
            else:  # noise
                for _ in range(100):
                    x = random.randint(cell[0], cell[2])
                    y = random.randint(cell[1], cell[3])
                    draw.point((x, y), fill=0)
    
    # 노이즈 추가
    for _ in range(1000):
        x = random.randint(0, width-1)
        y = random.randint(0, height-1)
        draw.point((x, y), fill=random.randint(0, 255))
    
    return np.array(image)

def generate_image_and_labels(image_id, resolution, bg_mode, has_gap, is_imperfect=False, empty_cell_ratio=0.2):
    image_width, image_height = resolution
    
    try:
        if is_imperfect:
            img = Image.fromarray(create_imperfect_table(image_width, image_height, 
                                                         random.randint(MIN_ROWS, MAX_ROWS), 
                                                         random.randint(MIN_COLS, MAX_COLS),
                                                         empty_cell_ratio))
            img = img.convert('L')
            draw = ImageDraw.Draw(img)
            cells, table_bbox, gap, rows, cols = create_table(image_width, image_height, has_gap)
        else:
            bg_color = random.choice(list(BACKGROUND_COLORS[bg_mode].values()))
            img = Image.new('L', (image_width, image_height), color=bg_color)
            draw = ImageDraw.Draw(img)
            cells, table_bbox, gap, rows, cols = create_table(image_width, image_height, has_gap)
            line_colors = list(LINE_COLORS[bg_mode].values())
            line_widths = draw_table(draw, cells, table_bbox, line_colors, has_gap)
            add_content_to_cells(draw, cells, random.choice(FONTS), bg_mode, empty_cell_ratio)
        
        # 이미지 대비 조정
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(random.uniform(1.2, 1.5))
        
        # YOLO 형식의 레이블 생성 (확장된 형식)
        labels = []
        for cell in cells:
            if len(cell) > 6:  # 병합된 셀
                x_center = (cell[0] + cell[2]) / 2 / image_width
                y_center = (cell[1] + cell[3]) / 2 / image_height
                width = (cell[2] - cell[0]) / image_width
                height = (cell[3] - cell[1]) / image_height
                start_row, start_col, merge_rows, merge_cols = cell[4], cell[5], cell[6], cell[7]
                labels.append(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f} {start_row} {start_col} {merge_rows} {merge_cols}")
            else:
                x_center = (cell[0] + cell[2]) / 2 / image_width
                y_center = (cell[1] + cell[3]) / 2 / image_height
                width = (cell[2] - cell[0]) / image_width
                height = (cell[3] - cell[1]) / image_height
                row, col = cell[4], cell[5]
                labels.append(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f} {row} {col} 1 1")
        # 전체 표에 대한 레이블
        labels.append(f"1 0.5 0.5 1.0 1.0 {rows} {cols}")    
        
        # 통계 수집
        stats = {
            'image_id': image_id,
            'bg_mode': bg_mode,
            'has_gap': has_gap,
            'is_imperfect': is_imperfect,
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
        }
        
        if not is_imperfect:
            stats.update({
                'avg_line_width': np.mean(line_widths),
                'min_line_width': min(line_widths),
                'max_line_width': max(line_widths),
            })
        
        return img, labels, stats
    except Exception as e:
        logger.error(f"Error generating image {image_id}: {str(e)}")
        return None

def generate_single_image_and_labels(args):
    image_id, resolution, bg_mode, has_gap, is_imperfect = args
    try:
        return generate_image_and_labels(image_id, resolution, bg_mode, has_gap, is_imperfect)
    except Exception as e:
        logger.error(f"Error generating image {image_id}: {str(e)}")
        return None

def save_image_and_labels(result, image_dir, label_dir):
    img, labels, image_id = result
    img.save(os.path.join(image_dir, f"{image_id}.png"))
    with open(os.path.join(label_dir, f"{image_id}.txt"), 'w') as f:
        f.write('\n'.join(labels))

def generate_dataset(light_images_with_gap, light_images_without_gap,
                     dark_images_with_gap, dark_images_without_gap,
                     imperfect_images, output_dir, resolution, imperfect_ratio):
    create_directory(output_dir)
    create_directory(os.path.join(output_dir, 'images'))
    create_directory(os.path.join(output_dir, 'labels'))
    imperfect_image_dir = os.path.join(output_dir, 'imperfect', 'images')
    imperfect_label_dir = os.path.join(output_dir, 'imperfect', 'labels')
    create_directory(imperfect_image_dir)
    create_directory(imperfect_label_dir)
    
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
        args_list = [(i, resolution, bg_mode, has_gap, False) for i in range(num_images)]
        imperfect_args_list = [(i + num_images, resolution, bg_mode, has_gap, True) for i in range(int(num_images * imperfect_ratio))]
        args_list.extend(imperfect_args_list)
        
        with ProcessPoolExecutor() as executor:
            results = list(tqdm(executor.map(generate_single_image_and_labels, args_list), 
                                total=len(args_list), 
                                desc=f"Generating {bg_mode} images ({'with' if has_gap else 'without'} gap)"))
        
        for i, result in enumerate(tqdm(results, desc=f"Saving {bg_mode} images ({'with' if has_gap else 'without'} gap)")):
            if result is not None:
                img, labels, image_stats = result
                if image_stats['is_imperfect']:
                    save_image_and_labels((img, labels, f"imperfect_{bg_mode}_{i}_{'gap' if has_gap else 'no_gap'}"), imperfect_image_dir, imperfect_label_dir)
                elif bg_mode == 'light':
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
        'train': {'light': './light/images', 'dark': './dark/images', 'imperfect': './imperfect/images'},
        'val': {'light': './light/images', 'dark': './dark/images', 'imperfect': './imperfect/images'},
        'nc': 2,
        'names': ['cell', 'table']
    }
    
    with open(os.path.join(output_dir, 'dataset.yaml'), 'w') as f:
        yaml.dump(yaml_content, f)
    
    logger.info(f"생성된 이미지 수:")
    logger.info(f"밝은 배경 (간격 있음) - {light_images_with_gap}")
    logger.info(f"밝은 배경 (간격 없음) - {light_images_without_gap}")
    logger.info(f"어두운 배경 (간격 있음) - {dark_images_with_gap}")
    logger.info(f"어두운 배경 (간격 없음) - {dark_images_without_gap}")
    logger.info(f"불완전한 표 - {int((light_images_with_gap + light_images_without_gap + dark_images_with_gap + dark_images_without_gap) * imperfect_ratio)}")
    logger.info(f"통계 정보가 {output_dir}/statistics.csv 와 {output_dir}/summary_statistics.csv 에 저장되었습니다.")

if __name__ == "__main__":
    tables = 100
    light_images_with_gap = tables
    light_images_without_gap = tables
    dark_images_with_gap = tables
    dark_images_without_gap = tables
    imperfect_ratio = 0.2  # 불완전한 이미지의 비율
    output_dir = 'table_dataset'
    resolution = (800, 600)
    generate_dataset(light_images_with_gap, light_images_without_gap,
                     dark_images_with_gap, dark_images_without_gap,
                     0, output_dir, resolution, imperfect_ratio)  # imperfect_images를 0으로 설정
