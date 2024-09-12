import random
import numpy as np, cv2
import math
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance
from io import BytesIO
from constants import *
from utils import draw_line, generate_random_text, adjust_font_size
import os
import multiprocessing
import yaml, json
from tqdm import tqdm
from constants import *
from utils import create_directory
from DatasetStat import DatasetStat

def normalize_coordinates(x, y, w, h, img_width, img_height):
    return x / img_width, y / img_height, w / img_width, h / img_height
def clip_coordinates(x, y, w, h):
    x = max(0, min(1, x))
    y = max(0, min(1, y))
    w = max(0.001, min(1-x, w))  # 최소 너비를 0.001로 설정
    h = max(0.001, min(1-y, h))  # 최소 높이를 0.001로 설정
    return x, y, w, h

def validate_and_fix_label(label):
    parts = label.split()
    if len(parts) != 10:
        return None
    try:
        class_id, x, y, w, h, angle, row_start, row_end, col_start, col_end = map(float, parts)
        x, y, w, h = clip_coordinates(x, y, w, h)
        return f"{class_id:.0f} {x:.6f} {y:.6f} {w:.6f} {h:.6f} {angle:.6f} {row_start:.0f} {row_end:.0f} {col_start:.0f} {col_end:.0f}"
    except ValueError:
        return None


def adjust_rotated_coordinates(x, y, angle, width, height):
    # 회전 중심을 (0,0)으로 이동
    x -= width / 2
    y -= height / 2
    
    # 회전 적용
    angle_rad = math.radians(angle)
    x_rot = x * math.cos(angle_rad) - y * math.sin(angle_rad)
    y_rot = x * math.sin(angle_rad) + y * math.cos(angle_rad)
    
    # 다시 원래 위치로 이동
    x_rot += width / 2
    y_rot += height / 2
    
    return x_rot, y_rot

def create_cell_label(x1, y1, x2, y2, width, height, i, j, merge_down, merge_right):
    center_x = (x1 + x2) / (2 * width)
    center_y = (y1 + y2) / (2 * height)
    w = (x2 - x1) / width
    h = (y2 - y1) / height
    return f"0 {center_x:.6f} {center_y:.6f} {w:.6f} {h:.6f} {i+1} {i+merge_down} {j+1} {j+merge_right}"

def create_table_label(table_top, height, rows, cols):
    y = (table_top + height) / (2 * height)
    h = (height - table_top) / height
    y, h = clip_coordinates(0.5, y, 1.0, h)[1:3]
    return f"1 0.500000 {y:.6f} 1.000000 {h:.6f} {rows} {cols}"

def validate_label(label):
    parts = label.split()
    if len(parts) != 10:
        return False
    try:
        class_id, x, y, w, h = map(float, parts[:5])
        if not (0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1):
            return False
    except ValueError:
        return False
    return True
def generate_dataset_worker(args):
    i, output_dir, complexity, style, rotation_range, varied, with_header, stats = args
    width = random.randint(MIN_IMAGE_WIDTH, MAX_IMAGE_WIDTH)
    height = random.randint(MIN_IMAGE_HEIGHT, MAX_IMAGE_HEIGHT)

    image, labels, rows, cols = generate_table(width, height, complexity, style, rotation_range, varied, with_header)

    # 레이블 검증 및 수정
    valid_labels = []
    modified_count = 0
    for label in labels:
        original_label = label
        fixed_label = validate_and_fix_label(label)
        if fixed_label:
            valid_labels.append(fixed_label)
            if fixed_label != original_label:
                modified_count += 1
    
    if modified_count > 0:
        print(f"Warning: {modified_count} labels were modified for image {i}")
        labels = valid_labels
    avg_cell_size = (width / cols) * (height / rows)

    # 통계 업데이트
    stats.update(rows, cols, avg_cell_size, (width, height), complexity, style, rotation_range[1], with_header, modified_count)
    
    image_path = os.path.join(output_dir, 'images', f'table_{i:06d}.png')
    label_path = os.path.join(output_dir, 'labels', f'table_{i:06d}.txt')

    try:
        image.save(image_path)
        with open(label_path, 'w') as f:
            f.write('\n'.join(labels))

        if with_header:
            info_path = os.path.join(output_dir, 'info', f'table_{i:06d}.txt')
            with open(info_path, 'w') as f:
                f.write(f"Rows: {rows}\nColumns: {cols}")
    except Exception as e:
        print(f"Error saving image or labels for table_{i:06d}: {str(e)}")

def generate_table(width, height, complexity, style, rotation_range, varied=False, with_header=False):
    bg_color = random.choice(list(COLORS.values()))
    image = Image.new('L', (width, height), color=bg_color)
    draw = ImageDraw.Draw(image)

    line_color = COLORS['black'] if bg_color > 128 else COLORS['white']
    line_width = get_line_width(style)

    # 헤더 추가 (옵션)
    table_top = 0
    header_label = None
    if with_header:
        header_height = random.randint(height // 20, height // 8)
        table_top = header_height + 10
        header_text = generate_random_text()
        font, text_width, text_height = adjust_font_size(draw, header_text, width-20, header_height-10, random.choice(FONTS))

        header_x = 10 if random.choice([True, False]) else (width - text_width) // 2
        draw.text((header_x, 5), header_text, font=font, fill=line_color)
        
        draw.line([(0, table_top), (width, table_top)], fill=line_color, width=line_width)
        header_label = f"2 0.5 {header_height/(2*height):.6f} 1.0 {header_height/height:.6f}"

    table_height = height - table_top - 10
    cols, rows = get_table_dimensions(complexity)

    col_widths, row_heights = calculate_cell_sizes(width, table_height, cols, rows, varied)

    grid = [[0 for _ in range(cols)] for _ in range(rows)]
    cell_labels = []

    for i in range(rows):
        for j in range(cols):
            if grid[i][j] == 0:
                merge_right, merge_down = get_merge_size(complexity, cols - j, rows - i)
                if not check_merge_possibility(grid, i, j, merge_down, merge_right):
                    merge_right = merge_down = 0

                x1, y1, x2, y2 = calculate_cell_coordinates(col_widths, row_heights, i, j, merge_right, merge_down, table_top)
                print(f"Cell coordinates: x1={x1}, y1={y1}, x2={x2}, y2={y2}")  # 디버깅 출력

                draw_cell(draw, x1, y1, x2, y2, line_color, line_width)
                draw_cell_text(draw, x1, y1, x2, y2, line_color)

                update_grid(grid, i, j, merge_down, merge_right)
                cell_label = create_cell_label(x1, y1, x2, y2, width, height, i, j, merge_down, merge_right)
                print(f"Generated cell label: {cell_label}")  # 디버깅 출력
                cell_labels.append(cell_label)

    table_label = create_table_label(table_top, height, rows, cols)

    padding = 50
    image, cell_labels = add_padding(image, cell_labels, padding)
    
    # 패딩 추가 후 레이블 좌표 조정
    adjusted_labels = []
    for label in cell_labels:
        parts = label.split()
        class_id, x, y, w, h = map(float, parts[:5])
        new_x = (x * width + padding) / (width + 2 * padding)
        new_y = (y * height + padding) / (height + 2 * padding)
        new_w = w * width / (width + 2 * padding)
        new_h = h * height / (height + 2 * padding)
        adjusted_label = f"{class_id} {new_x:.6f} {new_y:.6f} {new_w:.6f} {new_h:.6f} {' '.join(parts[5:])}"
        adjusted_labels.append(adjusted_label)
    
    cell_labels = adjusted_labels

    all_labels = adjusted_labels
    if header_label:
        all_labels.append(header_label)

    return image, all_labels, rows, cols

def get_line_width(style):
    if style == 'thin':
        return random.randint(1, 2)
    elif style == 'medium':
        return random.randint(2, 3)
    elif style == 'thick':
        return random.randint(3, 4)
    else:  # double
        return random.randint(2, 3)

def get_table_dimensions(complexity):
    if complexity == 'simple':
        return random.randint(3, 5), random.randint(3, 5)
    elif complexity == 'complex':
        return random.randint(5, 8), random.randint(5, 8)
    else:  # nested
        return random.randint(6, 10), random.randint(6, 10)
def calculate_cell_sizes(width, table_height, cols, rows, varied):
    if varied:
        min_width = max(1, width // (cols * 10))  # 최소 너비를 1픽셀로 설정
        min_height = max(1, table_height // (rows * 10))  # 최소 높이를 1픽셀로 설정
        
        col_widths = [random.randint(min_width, width//cols) for _ in range(cols)]
        row_heights = [random.randint(min_height, table_height//rows) for _ in range(rows)]
        
        # 전체 너비와 높이에 맞게 조정
        total_width = sum(col_widths)
        total_height = sum(row_heights)
        col_widths = [int(w * width / total_width) for w in col_widths]
        row_heights = [int(h * table_height / total_height) for h in row_heights]
    else:
        col_widths = [width // cols] * cols
        row_heights = [table_height // rows] * rows
    
    return col_widths, row_heights
def get_merge_size(complexity, max_right, max_down):
    merge_right = merge_down = 0
    if complexity in ['complex', 'nested'] and random.random() < 0.3:
        merge_right = random.randint(0, min(3, max_right - 1))
        merge_down = random.randint(0, min(3, max_down - 1))
    if random.random() < 0.1:  # 큰 셀 병합
        merge_right = random.randint(0, min(6, max_right - 1))
        merge_down = random.randint(0, min(6, max_down - 1))
    return merge_right, merge_down

def check_merge_possibility(grid, i, j, merge_down, merge_right):
    rows, cols = len(grid), len(grid[0])
    for mi in range(merge_down + 1):
        for mj in range(merge_right + 1):
            if i + mi >= rows or j + mj >= cols or grid[i + mi][j + mj] != 0:
                return False
    return True

def calculate_cell_coordinates(col_widths, row_heights, i, j, merge_right, merge_down, table_top):
    x1 = sum(col_widths[:j])
    y1 = table_top + sum(row_heights[:i])
    x2 = x1 + sum(col_widths[j:j+merge_right+1])
    y2 = y1 + sum(row_heights[i:i+merge_down+1])
    return x1, y1, x2, y2

def draw_cell(draw, x1, y1, x2, y2, line_color, line_width):
    draw.rectangle([x1, y1, x2, y2], outline=line_color, width=line_width)
def draw_cell_text(draw, x1, y1, x2, y2, line_color):
    cell_text = generate_random_text()
    font, text_width, text_height = adjust_font_size(draw, cell_text, x2-x1-10, y2-y1-10, random.choice(FONTS))
    
    # 텍스트 위치 결정 (왼쪽 위 또는 가운데)
    if random.choice([True, False]):  # 50% 확률로 왼쪽 위 또는 가운데 선택
        text_x = x1 + 5
        text_y = y1 + 5
    else:
        text_x = x1 + ((x2 - x1) - text_width) // 2
        text_y = y1 + ((y2 - y1) - text_height) // 2
    
    draw.text((text_x, text_y), cell_text, font=font, fill=line_color)

def update_grid(grid, i, j, merge_down, merge_right):
    for mi in range(merge_down + 1):
        for mj in range(merge_right + 1):
            grid[i+mi][j+mj] = 1
def create_cell_label(x1, y1, x2, y2, width, height, i, j, merge_down, merge_right):
    center_x = (x1 + x2) / (2 * width)
    center_y = (y1 + y2) / (2 * height)
    w = (x2 - x1) / width
    h = (y2 - y1) / height
    
    # 좌표가 유효한 범위 내에 있는지 확인
    center_x = max(0, min(1, center_x))
    center_y = max(0, min(1, center_y))
    w = max(0.001, min(1, w))
    h = max(0.001, min(1, h))
    
    return f"0 {center_x:.6f} {center_y:.6f} {w:.6f} {h:.6f} {i+1} {i+merge_down} {j+1} {j+merge_right}"

def create_table_label(table_top, height, rows, cols):
    table_center_x = 0.5
    table_center_y = (table_top + height) / (2 * height)
    table_width = 1.0
    table_height = (height - table_top) / height
    return f"1 {table_center_x:.6f} {table_center_y:.6f} {table_width:.6f} {table_height:.6f} {rows} {cols}"

def apply_image_effects(image):
    if random.random() < 0.5:
        noise = np.random.normal(0, 10, image.size[::-1])
        noise = Image.fromarray(noise.astype('uint8'), mode='L')
        image = Image.blend(image, noise, 0.1)
    if random.random() < 0.5:
        enhancer = ImageEnhance.Brightness(image)
        image = enhancer.enhance(random.uniform(0.8, 1.2))
    if random.random() < 0.5:
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(random.uniform(0.8, 1.2))
    if random.random() < 0.3:
        image = image.filter(ImageFilter.GaussianBlur(radius=random.uniform(0, 1)))
    if random.random() < 0.3:
        buffer = BytesIO()
        image.save(buffer, format="JPEG", quality=random.randint(50, 95))
        image = Image.open(buffer)
    return image

def rotate_image(image, rotation_range):
    rotation_angle = random.uniform(*rotation_range)
    rotated = image.rotate(rotation_angle, resample=Image.BICUBIC, expand=True)
    return rotated, rotation_angle

def rotate_image_and_adjust_labels(image, labels, rotation_range):
    rotation_angle = random.uniform(*rotation_range)
    center = (image.width / 2, image.height / 2)
    rotated_image = image.rotate(rotation_angle, resample=Image.BICUBIC, expand=True, center=center)
    
    new_w, new_h = rotated_image.size
    old_w, old_h = image.size

    adjusted_labels = []
    for label in labels:
        parts = label.split()
        class_id = parts[0]
        x, y, w, h = map(float, parts[1:5])
        
        # 바운딩 박스의 모서리 좌표 계산
        corners = [
            (x - w/2, y - h/2),
            (x + w/2, y - h/2),
            (x + w/2, y + h/2),
            (x - w/2, y + h/2)
        ]
        
        # 각 모서리 회전
        rotated_corners = []
        for corner_x, corner_y in corners:
            rx, ry = adjust_rotated_coordinates(corner_x * old_w, corner_y * old_h, rotation_angle, old_w, old_h)
            rx, ry = rx / new_w, ry / new_h
            rotated_corners.append((rx, ry))
        
        # 회전된 바운딩 박스의 중심점과 치수 계산
        min_x = max(0, min(corner[0] for corner in rotated_corners))
        max_x = min(1, max(corner[0] for corner in rotated_corners))
        min_y = max(0, min(corner[1] for corner in rotated_corners))
        max_y = min(1, max(corner[1] for corner in rotated_corners))
        
        center_x = (min_x + max_x) / 2
        center_y = (min_y + max_y) / 2
        new_w = max_x - min_x
        new_h = max_y - min_y
        
        # 회전 각도 계산 (라디안)
        dx = rotated_corners[1][0] - rotated_corners[0][0]
        dy = rotated_corners[1][1] - rotated_corners[0][1]
        angle = math.atan2(dy, dx)
        
        # YOLO 형식으로 변환 (회전 정보 포함)
        adjusted_label = f"{class_id} {center_x:.6f} {center_y:.6f} {new_w:.6f} {new_h:.6f} {angle:.6f}"
        if len(parts) > 5:
            adjusted_label += " " + " ".join(parts[5:])
        
        # 레이블 검증 및 수정
        fixed_label = validate_and_fix_label(adjusted_label)
        if fixed_label:
            adjusted_labels.append(fixed_label)
    
    return rotated_image, adjusted_labels


def add_padding(image, labels, padding):
    width, height = image.size
    new_width = width + 2 * padding
    new_height = height + 2 * padding
    padded_image = Image.new(image.mode, (new_width, new_height), color=image.getpixel((0,0)))
    padded_image.paste(image, (padding, padding))
    
    adjusted_labels = []
    for label in labels:
        parts = label.split()
        class_id = parts[0]
        x, y, w, h = map(float, parts[1:5])
        
        # 새 좌표 계산
        new_x = (x * width + padding) / new_width
        new_y = (y * height + padding) / new_height
        new_w = (w * width) / new_width
        new_h = (h * height) / new_height
        
        adjusted_label = f"{class_id} {new_x:.6f} {new_y:.6f} {new_w:.6f} {new_h:.6f}"
        if len(parts) > 5:
            adjusted_label += " " + " ".join(parts[5:])
        adjusted_labels.append(adjusted_label)
        fixed_label = validate_and_fix_label(adjusted_label)
        if fixed_label:
            adjusted_labels.append(fixed_label)
    
    return padded_image, adjusted_labels
    
def generate_datasets(num_images, output_dir, complexities, styles, train_ratio=0.8, rotation_range=(-15, 15)):
    dataset_types = [
        ('uniform', False, False, rotation_range),
        ('uniform-no_rotation', False, False, (0, 0)),
        ('varied', True, False, rotation_range),
        ('header', False, True, rotation_range)
    ]

    for dataset_name, varied, with_header, rot_range in dataset_types:
        print(f"Generating {dataset_name} dataset...")
        generate_dataset(num_images, os.path.join(output_dir, dataset_name), 
        complexities, styles, train_ratio, rot_range, 
        varied=varied, with_header=with_header)

def generate_dataset(num_images, output_dir, complexities, styles, train_ratio, rotation_range=(-15,15), varied=False, with_header=False):
    create_directory(os.path.join(output_dir, 'images'))
    create_directory(os.path.join(output_dir, 'labels'))
    if with_header:
        create_directory(os.path.join(output_dir, 'info'))
    stats = DatasetStat()
    args_list = []
    for i in range(num_images):
        complexity = random.choice(complexities)
        style = random.choice(styles)
        args_list.append((i, output_dir, complexity, style, rotation_range, varied, with_header, stats))
    with multiprocessing.Pool() as pool:
        list(tqdm(pool.imap_unordered(generate_dataset_worker, args_list), total=num_images))

    # 통계 저장
    stats_summary = stats.get_summary()
    with open(os.path.join(output_dir, 'dataset_statistics.json'), 'w') as f:
        json.dump(stats_summary, f, indent=4)
        
    # 훈련 세트와 검증 세트로 분할
    all_images = os.listdir(os.path.join(output_dir, 'images'))
    random.shuffle(all_images)
    split_index = int(len(all_images) * train_ratio)
    train_images = all_images[:split_index]
    val_images = all_images[split_index:]

    # YAML 파일 생성
    yaml_content = {
        'train': os.path.join(output_dir, 'images'),
        'val': os.path.join(output_dir, 'images'),
        'nc': 2 if not with_header else 1,
        'names': ['cell', 'table'] if not with_header else ['cell']
    }

    with open(os.path.join(output_dir, 'dataset.yaml'), 'w') as f:
        yaml.dump(yaml_content, f)

    print(f"생성된 이미지 수: {num_images}")
    print(f"훈련 세트: {len(train_images)}")
    print(f"검증 세트: {len(val_images)}")


    print("All datasets generated successfully.")
    # 시각화 함수 (수정된 레이블이 있는 이미지만 시각화)
def visualize_modified_images(dataset_dir):
    image_dir = os.path.join(dataset_dir, 'images')
    label_dir = os.path.join(dataset_dir, 'labels')
    output_dir = os.path.join(dataset_dir, 'visualized')
    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(image_dir):
        if filename.endswith('.png'):
            label_path = os.path.join(label_dir, filename.replace('.png', '.txt'))
            with open(label_path, 'r') as f:
                labels = f.readlines()
            
            if any('modified' in label for label in labels):
                image_path = os.path.join(image_dir, filename)
                image = cv2.imread(image_path)
                for label in labels:
                    if 'modified' in label:
                        parts = label.split()
                        x, y, w, h = map(float, parts[1:5])
                        x1, y1 = int(x * image.shape[1]), int(y * image.shape[0])
                        x2, y2 = int((x + w) * image.shape[1]), int((y + h) * image.shape[0])
                        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                cv2.imwrite(os.path.join(output_dir, filename), image)
if __name__ == '__main__':
    generate_datasets(100, 'table_dataset', 
    complexities=['simple', 'complex', 'nested'],
    styles=STYLES,
    train_ratio=0.8,
    rotation_range=(-10, 10))  # 회전 각도 범위 축소

    visualize_modified_images('./table_dataset/uniform')
