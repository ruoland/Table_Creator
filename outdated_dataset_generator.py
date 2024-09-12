import os
import random
import multiprocessing
import yaml, json
from tqdm import tqdm
from constants import *
from PIL import Image, ImageDraw
from utils import create_directory
from DatasetStat import DatasetStat
from outdated_table_generator import generate_table, get_table_dimensions, get_merge_size, generate_random_text, get_line_width,apply_image_effects, adjust_font_size, rotate_image
def generate_dataset_worker(args):
    i, output_dir, complexity, style, rotation_range, varied, with_header, stats = args
    width = random.randint(MIN_IMAGE_WIDTH, MAX_IMAGE_WIDTH)
    height = random.randint(MIN_IMAGE_HEIGHT, MAX_IMAGE_HEIGHT)

    if varied:
        image, labels, rows, cols = generate_table(width, height, complexity, style, varied, with_header)
    else:
        image, labels = generate_uniform_table(width, height, complexity, style)
        rows, cols = get_table_dimensions(complexity)

    # 회전 적용
    rotation_angle = random.uniform(rotation_range[0], rotation_range[1])
    image = rotate_image(image, rotation_angle)

    # 통계 업데이트
    avg_cell_size = (width / cols) * (height / rows)
    stats.update(rows, cols, avg_cell_size, (width, height), complexity, style, rotation_angle, with_header)
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

def generate_uniform_table(width, height, complexity, style):
    bg_color = random.choice(list(COLORS.values()))
    image = Image.new('L', (width, height), color=bg_color)
    draw = ImageDraw.Draw(image)

    line_color = COLORS['black'] if bg_color > 128 else COLORS['white']
    line_width = get_line_width(style)

    if complexity == 'simple':
        cols, rows = random.randint(3, 5), random.randint(3, 5)
    elif complexity == 'complex':
        cols, rows = random.randint(5, 8), random.randint(5, 8)
    else:  # nested
        cols, rows = random.randint(6, 10), random.randint(6, 10)

    cell_width = width // cols
    cell_height = height // rows

    labels = []

    # 표 그리기
    for i in range(rows + 1):
        y = i * cell_height
        draw.line([(0, y), (width, y)], fill=line_color, width=line_width)

    for j in range(cols + 1):
        x = j * cell_width
        draw.line([(x, 0), (x, height)], fill=line_color, width=line_width)

    # 셀 내용 추가 및 레이블 생성
    for i in range(rows):
        for j in range(cols):
            x1 = j * cell_width
            y1 = i * cell_height
            x2 = (j + 1) * cell_width
            y2 = (i + 1) * cell_height

            cell_text = generate_random_text()
            font, text_width, text_height = adjust_font_size(draw, cell_text, cell_width-10, cell_height-10, random.choice(FONTS))
            text_x = x1 + (cell_width - text_width) // 2
            text_y = y1 + (cell_height - text_height) // 2
            draw.text((text_x, text_y), cell_text, font=font, fill=line_color)

            center_x = (x1 + x2) / (2 * width)
            center_y = (y1 + y2) / (2 * height)
            w = cell_width / width
            h = cell_height / height
            labels.append(f"0 {center_x:.6f} {center_y:.6f} {w:.6f} {h:.6f}")

    image = apply_image_effects(image)

    return image, labels

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
if __name__ == '__main__':
    generate_datasets(100, 'table_dataset', 
    complexities=['simple', 'complex', 'nested'],
    styles=STYLES,
    train_ratio=0.8,
    rotation_range=(-15, 15))
