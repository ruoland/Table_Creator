import os
import random
import numpy as np
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance
from tqdm import tqdm
import multiprocessing
import yaml

# 상수 정의
MIN_IMAGE_WIDTH, MAX_IMAGE_WIDTH = 800, 1200
MIN_IMAGE_HEIGHT, MAX_IMAGE_HEIGHT = 600, 1000
MIN_COLS, MAX_COLS = 3, 10
MIN_ROWS, MAX_ROWS = 3, 10
STYLES = ['thin', 'medium', 'thick', 'double']
FONTS = ['font/NanumGothic.ttf', 'font/SANGJU Dajungdagam.ttf', 'font/SOYO Maple Regular.ttf']
COLORS = {'white': 255, 'light_gray': 220, 'dark_gray': 64, 'black': 0}
SHAPES = ['rectangle', 'ellipse', 'triangle', 'line', 'polygon']

def create_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def draw_line(draw, start, end, width, style, color):
    if style == 'double':
        draw.line([start, end], fill=color, width=width)
        offset = max(1, width // 3)
        draw.line([(start[0]+offset, start[1]+offset), (end[0]+offset, end[1]+offset)], fill=color, width=1)
    else:
        draw.line([start, end], fill=color, width=width)

def generate_random_text():
    return ''.join(random.choices('가나다라마바사아자차카타파하', k=random.randint(2, 5)))

def draw_shape(draw, x1, y1, x2, y2, color):
    shape = random.choice(SHAPES)
    if shape == 'rectangle':
        draw.rectangle([x1, y1, x2, y2], outline=color)
    elif shape == 'ellipse':
        draw.ellipse([x1, y1, x2, y2], outline=color)
    elif shape == 'triangle':
        draw.polygon([(x1, y2), ((x1+x2)//2, y1), (x2, y2)], outline=color)
    elif shape == 'line':
        draw.line([x1, y1, x2, y2], fill=color)
    elif shape == 'polygon':
        points = [(random.randint(x1, x2), random.randint(y1, y2)) for _ in range(random.randint(3, 6))]
        draw.polygon(points, outline=color)
def generate_table(width, height, complexity, style, rotation_range):
    # 배경 생성
    bg_color = random.choice(list(COLORS.values()))
    image = Image.new('L', (width, height), color=bg_color)
    draw = ImageDraw.Draw(image)

    # 표 스타일 설정
    if style == 'thin':
        line_width = random.randint(1, 2)
    elif style == 'medium':
        line_width = random.randint(2, 3)
    elif style == 'thick':
        line_width = random.randint(3, 4)
    else:  # double
        line_width = random.randint(2, 3)

    # 선 색상 설정
    line_color = COLORS['black'] if bg_color > 128 else COLORS['white']

    # 행과 열 수 결정
    if complexity == 'simple':
        cols, rows = random.randint(3, 5), random.randint(3, 5)
    elif complexity == 'complex':
        cols, rows = random.randint(5, 8), random.randint(5, 8)
    else:  # nested
        cols, rows = random.randint(6, 10), random.randint(6, 10)

     # 다양한 셀 크기 생성
    col_widths = [random.randint(width//(cols*2), width//cols) for _ in range(cols)]
    row_heights = [random.randint(height//(rows*2), height//rows) for _ in range(rows)]

    # 전체 너비와 높이에 맞게 조정
    total_width = sum(col_widths)
    total_height = sum(row_heights)
    col_widths = [int(w * width / total_width) for w in col_widths]
    row_heights = [int(h * height / total_height) for h in row_heights]

    # 표 그리기 및 셀 병합
    grid = [[0 for _ in range(cols)] for _ in range(rows)]
    labels = []

    for i in range(rows):
        for j in range(cols):
            if grid[i][j] == 0:
                # 셀 병합 결정
                merge_right = merge_down = 0
                if complexity in ['complex', 'nested'] and random.random() < 0.3:
                    merge_right = random.randint(0, min(3, cols - j - 1))
                    merge_down = random.randint(0, min(3, rows - i - 1))
                
                if random.random() < 0.1:  # 큰 셀 병합
                    merge_right = random.randint(0, min(6, cols - j - 1))
                    merge_down = random.randint(0, min(6, rows - i - 1))

                # 병합 가능 여부 확인
                can_merge = True
                for mi in range(merge_down + 1):
                    for mj in range(merge_right + 1):
                        if i + mi >= rows or j + mj >= cols or grid[i + mi][j + mj] != 0:
                            can_merge = False
                            break
                    if not can_merge:
                        break

                # 병합이 불가능하면 단일 셀로 처리
                if not can_merge:
                    merge_right = merge_down = 0

                # 셀 그리기
                x1 = sum(col_widths[:j])
                y1 = sum(row_heights[:i])
                x2 = x1 + sum(col_widths[j:j+merge_right+1])
                y2 = y1 + sum(row_heights[i:i+merge_down+1])

                draw.rectangle([x1, y1, x2, y2], outline=line_color, width=line_width)

                draw.rectangle([x1, y1, x2, y2], outline=line_color, width=line_width)

                # 셀에 텍스트 추가
                cell_text = generate_random_text()
                font_size = 10  # 초기 폰트 크기
                font = ImageFont.truetype(random.choice(FONTS), size=font_size)
                bbox = draw.textbbox((0, 0), cell_text, font=font)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]

                # 텍스트 크기 조정
                while text_width < (x2 - x1 - 10) * 0.8 and text_height < (y2 - y1 - 10) * 0.8 and font_size < 40:
                    font_size += 1
                    font = ImageFont.truetype(random.choice(FONTS), size=font_size)
                    bbox = draw.textbbox((0, 0), cell_text, font=font)
                    text_width = bbox[2] - bbox[0]
                    text_height = bbox[3] - bbox[1]

                # 텍스트가 너무 크면 한 단계 줄임
                if text_width > (x2 - x1 - 10) * 0.9 or text_height > (y2 - y1 - 10) * 0.9:
                    font_size -= 1
                    font = ImageFont.truetype(random.choice(FONTS), size=font_size)
                    bbox = draw.textbbox((0, 0), cell_text, font=font)
                    text_width = bbox[2] - bbox[0]
                    text_height = bbox[3] - bbox[1]

                # 텍스트 그리기
                text_x = x1 + ((x2 - x1) - text_width) // 2
                text_y = y1 + ((y2 - y1) - text_height) // 2
                draw.text((text_x, text_y), cell_text, font=font, fill=line_color)

                # 병합된 셀 표시
                for mi in range(merge_down + 1):
                    for mj in range(merge_right + 1):
                        grid[i+mi][j+mj] = 1

                # 레이블 생성
                center_x = (x1 + x2) / (2 * width)
                center_y = (y1 + y2) / (2 * height)
                w = (x2 - x1) / width
                h = (y2 - y1) / height
                labels.append(f"0 {center_x:.6f} {center_y:.6f} {w:.6f} {h:.6f}")
    # 이미지 처리 및 왜곡
    if random.random() < 0.5:  # 가우시안 노이즈
        noise = np.random.normal(0, 10, image.size[::-1])
        noise = Image.fromarray(noise.astype('uint8'), mode='L')
        image = Image.blend(image, noise, 0.1)

    if random.random() < 0.5:  # 밝기 조정
        enhancer = ImageEnhance.Brightness(image)
        image = enhancer.enhance(random.uniform(0.8, 1.2))

    if random.random() < 0.5:  # 대비 조정
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(random.uniform(0.8, 1.2))

    if random.random() < 0.3:  # 블러 효과
        image = image.filter(ImageFilter.GaussianBlur(radius=random.uniform(0, 1)))

    if random.random() < 0.3:  # 이미지 압축 시뮬레이션
        buffer = BytesIO()
        image.save(buffer, format="JPEG", quality=random.randint(50, 95))
        image = Image.open(buffer)

    # 표 회전
    rotation_angle = random.uniform(*rotation_range)
    image = image.rotate(rotation_angle, resample=Image.BICUBIC, expand=True)

    return image, labels

def generate_dataset_worker(args):
    i, output_dir, complexity, style, rotation_range = args
    width = random.randint(MIN_IMAGE_WIDTH, MAX_IMAGE_WIDTH)
    height = random.randint(MIN_IMAGE_HEIGHT, MAX_IMAGE_HEIGHT)

    image, labels = generate_table(width, height, complexity, style, rotation_range)

    image_path = os.path.join(output_dir, 'images', f'table_{i:06d}.png')
    label_path = os.path.join(output_dir, 'labels', f'table_{i:06d}.txt')

    image.save(image_path)
    with open(label_path, 'w') as f:
        f.write('\n'.join(labels))

def generate_dataset(num_images, output_dir, complexities, styles, train_ratio=0.8, rotation_range=(-15, 15)):
    create_directory(output_dir)
    create_directory(os.path.join(output_dir, 'images'))
    create_directory(os.path.join(output_dir, 'labels'))

    args_list = []
    for i in range(num_images):
        complexity = random.choice(complexities)
        style = random.choice(styles)
        args_list.append((i, output_dir, complexity, style, rotation_range))

    with multiprocessing.Pool() as pool:
        list(tqdm(pool.imap_unordered(generate_dataset_worker, args_list), total=num_images))

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
        'nc': 1,
        'names': ['cell']
    }

    with open(os.path.join(output_dir, 'dataset.yaml'), 'w') as f:
        yaml.dump(yaml_content, f)

    print(f"생성된 이미지 수: {num_images}")
    print(f"훈련 세트: {len(train_images)}")
    print(f"검증 세트: {len(val_images)}")

if __name__ == '__main__':
    generate_dataset(1000, 'table_dataset', 
    complexities=['simple', 'complex', 'nested'],
    styles=STYLES,
    train_ratio=0.8,
    rotation_range=(-15, 15))
