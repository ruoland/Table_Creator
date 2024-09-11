import multiprocessing, math, os, random
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance, ImageFont
import string
from tqdm import tqdm
from io import BytesIO
MIN_IMAGE_WIDTH = 800
MAX_IMAGE_WIDTH = 1200
MIN_IMAGE_HEIGHT = 600
MAX_IMAGE_HEIGHT = 1000

SIMPLE_MIN_COLS = 4
SIMPLE_MAX_COLS = 8
SIMPLE_MIN_ROWS = 4
SIMPLE_MAX_ROWS = 8

COMPLEX_MIN_COLS = 4
COMPLEX_MAX_COLS = 12
COMPLEX_MIN_ROWS = 4
COMPLEX_MAX_ROWS = 12

NESTED_MIN_COLS = 3
NESTED_MAX_COLS = 10
NESTED_MIN_ROWS = 3
NESTED_MAX_ROWS = 10

MERGE_PROBABILITY = 0.5
MAX_MERGE_RIGHT = 10
MAX_MERGE_DOWN = 10
LARGE_MERGE_PROBABILITY = 0.4
LARGE_MERGE_MIN = 4
LARGE_MERGE_MAX = 10

MIN_CELL_COLOR = 160
MAX_CELL_COLOR = 255
LINE_COLORS = [0, 25, 50, 75, 100, 125]

LINE_WIDTHS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
LINE_STYLES = ['solid', 'double']

BORDER_WIDTHS = [3, 4, 5, 6, 7, 8, 9 ,10]

BORDER_REMOVE_PROBABILITY = 0.1

MIN_ROTATION = -0
MAX_ROTATION = 0

TOTAL_IMAGES = 30000
TRAIN_RATIO = 0.8

COMPLEXITIES = ['simple', 'complex', 'nested']
COMPLEXITY_WEIGHTS = [0.3, 0.3, 0.3]

GAUSSIAN_NOISE_PROBABILITY = 0.5
GAUSSIAN_NOISE_MEAN = 0
GAUSSIAN_NOISE_STD = 10
GAUSSIAN_NOISE_BLEND = 0.1

BRIGHTNESS_ADJUST_PROBABILITY = 0.5
BRIGHTNESS_MIN = 0.8
BRIGHTNESS_MAX = 1.2

CONTRAST_ADJUST_PROBABILITY = 0.5
CONTRAST_MIN = 0.8
CONTRAST_MAX = 1.2

BLUR_PROBABILITY = 0.3
BLUR_MAX_RADIUS = 1

COMPRESSION_PROBABILITY = 0.3
COMPRESSION_MIN_QUALITY = 50
COMPRESSION_MAX_QUALITY = 95

# New constants
GRAY_BG_MIN = 200
GRAY_BG_MAX = 240
BLACK_BG_PROBABILITY = 0.2
WHITE_LINE_MIN = 200
WHITE_LINE_MAX = 255
TEXT_PROBABILITY = 0.7
SHAPE_PROBABILITY = 0.3

# 새로운 상수들
SHAPE_TYPES = ['rectangle', 'ellipse', 'triangle', 'line', 'polygon']
FONT_SIZES = range(8, 24)  # 8부터 23까지의 폰트 크기
FONTS = ['./font/NanumGothic.ttf', './font/SOYO Maple Regular.ttf', './font/SANGJU Dajungdagam.ttf']  # 사용 가능한 폰트 파일들

def generate_random_text():
    """랜덤한 텍스트 생성"""
    length = random.randint(3, 10)
    return ''.join(random.choices(string.ascii_letters + string.digits, k=length))

def draw_random_shape(draw, x1, y1, x2, y2, color):
    """랜덤한 도형 그리기"""
    shape = random.choice(SHAPE_TYPES)
    if shape == 'rectangle':
        draw.rectangle([x1, y1, x2, y2], outline=color)
    elif shape == 'ellipse':
        draw.ellipse([x1, y1, x2, y2], outline=color)
    elif shape == 'triangle':
        points = [(x1, y2), ((x1+x2)//2, y1), (x2, y2)]
        draw.polygon(points, outline=color)
    elif shape == 'line':
        draw.line([x1, y1, x2, y2], fill=color)
    elif shape == 'polygon':
        num_points = random.randint(5, 8)
        points = [(random.randint(x1, x2), random.randint(y1, y2)) for _ in range(num_points)]
        draw.polygon(points, outline=color)
def add_content_to_cell(draw, x1, y1, x2, y2, bg_type):
    """셀에 내용 추가"""
    content_type = random.choices(['text', 'shape', 'none'], weights=[0.4, 0.3, 0.3])[0]
    color = 0 if bg_type != 'black' else 255

    if content_type == 'text':
        cell_width = x2 - x1
        cell_height = y2 - y1
        max_font_size = min(cell_width, cell_height) // 2  # 셀 크기의 절반을 최대 폰트 크기로 설정
        font_size = random.randint(8, max_font_size)
        font = ImageFont.truetype(random.choice(FONTS), font_size)
        
        text = generate_random_text()
        text_bbox = draw.textbbox((0, 0), text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        
        # 텍스트가 셀을 벗어나는 경우 크기 조정
        if text_width > cell_width or text_height > cell_height:
            scale_factor = min(cell_width / text_width, cell_height / text_height)
            font_size = int(font_size * scale_factor)
            font = ImageFont.truetype(random.choice(FONTS), font_size)
            text_bbox = draw.textbbox((0, 0), text, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
        
        # 텍스트 위치 계산 (셀 내에서 랜덤하게 위치)
        text_x = random.randint(x1, max(x1, x2 - text_width))
        text_y = random.randint(y1, max(y1, y2 - text_height))
        
        draw.text((text_x, text_y), text, font=font, fill=color)
    
    elif content_type == 'shape':
        shape_x1 = x1 + (x2 - x1) // 4
        shape_y1 = y1 + (y2 - y1) // 4
        shape_x2 = x2 - (x2 - x1) // 4
        shape_y2 = y2 - (y2 - y1) // 4
        draw_random_shape(draw, shape_x1, shape_y1, shape_x2, shape_y2, color)

def create_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
def draw_line(draw, start, end, color, width, style='solid'):
    if style == 'solid':
        draw.line([start, end], fill=color, width=width)
    elif style == 'double':
        draw.line([start, end], fill=color, width=width)
        offset = width // 2
        if start[0] == end[0]:  # vertical line
            draw.line([(start[0]+offset, start[1]), (end[0]+offset, end[1])], fill=color, width=1)
            draw.line([(start[0]-offset, start[1]), (end[0]-offset, end[1])], fill=color, width=1)
        else:  # horizontal line
            draw.line([(start[0], start[1]+offset), (end[0], end[1]+offset)], fill=color, width=1)
            draw.line([(start[0], start[1]-offset), (end[0], end[1]-offset)], fill=color, width=1)
def generate_table(width, height, complexity, rotation_range):
    # 배경 타입 선택 (흰색 배경의 비중을 줄이고 검은색/회색 배경의 비중을 늘림)
    bg_type = random.choices(['white', 'gray', 'black'], weights=[0.2, 0.4, 0.4])[0]
    
    if bg_type == 'white':
        image = Image.new('L', (width, height), color=255)
    elif bg_type == 'gray':
        bg_color = random.randint(GRAY_BG_MIN, GRAY_BG_MAX)
        image = Image.new('L', (width, height), color=bg_color)
    else:  # black
        image = Image.new('L', (width, height), color=0)
    
    draw = ImageDraw.Draw(image)
    if complexity == 'simple':
        cols, rows = random.randint(SIMPLE_MIN_COLS, SIMPLE_MAX_COLS), random.randint(SIMPLE_MIN_ROWS, SIMPLE_MAX_ROWS)
    elif complexity == 'complex':
        cols, rows = random.randint(COMPLEX_MIN_COLS, COMPLEX_MAX_COLS), random.randint(COMPLEX_MIN_ROWS, COMPLEX_MAX_ROWS)
    else:  # nested
        cols, rows = random.randint(NESTED_MIN_COLS, NESTED_MAX_COLS), random.randint(NESTED_MIN_ROWS, NESTED_MAX_ROWS)

    col_widths = [random.randint(width//(cols*2), width//cols) for _ in range(cols)]
    row_heights = [random.randint(height//(rows*2), height//rows) for _ in range(rows)]

    total_width = sum(col_widths)
    total_height = sum(row_heights)
    col_widths = [int(w * width / total_width) for w in col_widths]
    row_heights = [int(h * height / total_height) for h in row_heights]

    grid = [[0 for _ in range(cols)] for _ in range(rows)]
    labels = []

    border_color = random.choice(LINE_COLORS)
    border_width = random.choice(BORDER_WIDTHS)
    border_style = random.choice(LINE_STYLES)

    for row in range(rows):
        for col in range(cols):
            if grid[row][col] == 0:
                x1 = sum(col_widths[:col])
                y1 = sum(row_heights[:row])

                merge_right = merge_down = 0
                if complexity in ['complex', 'nested'] and random.random() < MERGE_PROBABILITY:
                    merge_right = random.randint(0, min(MAX_MERGE_RIGHT, cols - col - 1))
                    merge_down = random.randint(0, min(MAX_MERGE_DOWN, rows - row - 1))

                if random.random() < LARGE_MERGE_PROBABILITY:
                    if random.choice([True, False]):
                        merge_right = min(cols - col - 1, random.randint(LARGE_MERGE_MIN, LARGE_MERGE_MAX))
                        merge_down = 0
                    else:
                        merge_right = 0
                        merge_down = min(rows - row - 1, random.randint(LARGE_MERGE_MIN, LARGE_MERGE_MAX))

                x2 = sum(col_widths[:col+merge_right+1])
                y2 = sum(row_heights[:row+merge_down+1])

                for r in range(row, row + merge_down + 1):
                    for c in range(col, col + merge_right + 1):
                        grid[r][c] = 1

                cell_color = random.randint(MIN_CELL_COLOR, MAX_CELL_COLOR)
                draw.rectangle([x1, y1, x2, y2], fill=cell_color)
# 선 색상 선택 로직 변경
                if bg_type == 'black':
                    # 검은 배경일 때는 항상 흰색~밝은 회색 선 사용
                    line_color = random.randint(200, 255)
                else:
                    # 흰색/회색 배경일 때 흰색~밝은 회색 선을 더 자주 사용
                    if random.random() < 0.7:  # 70% 확률로 흰색~밝은 회색 선
                        line_color = random.randint(200, 255)
                    else:  # 30% 확률로 검은색~어두운 회색 선
                        line_color = random.randint(0, 100)

                line_width = random.choice(LINE_WIDTHS)
                line_style = random.choice(LINE_STYLES)
# Add text or shape to cell
                add_content_to_cell(draw, x1, y1, x2, y2, bg_type)
                borders = [True, True, True, True]  # 상, 우, 하, 좌
                for i in range(4):
                    if (i == 0 and row == 0) or (i == 1 and col == cols - 1) or \
                       (i == 2 and row == rows - 1) or (i == 3 and col == 0):
                        if random.random() < BORDER_REMOVE_PROBABILITY:
                            borders[i] = False

                for i, border in enumerate(borders):
                    if border:
                        if i == 0:
                            draw_line(draw, (x1, y1), (x2, y1), line_color, line_width, line_style)
                        elif i == 1:
                            draw_line(draw, (x2, y1), (x2, y2), line_color, line_width, line_style)
                        elif i == 2:
                            draw_line(draw, (x2, y2), (x1, y2), line_color, line_width, line_style)
                        else:
                            draw_line(draw, (x1, y2), (x1, y1), line_color, line_width, line_style)

                center_x, center_y = (x1 + x2) / (2 * width), (y1 + y2) / (2 * height)
                box_width, box_height = (x2 - x1) / width, (y2 - y1) / height
                labels.append(f"0 {center_x:.6f} {center_y:.6f} {box_width:.6f} {box_height:.6f}")

    draw_line(draw, (0, 0), (width, 0), border_color, border_width, border_style)
    draw_line(draw, (width, 0), (width, height), border_color, border_width, border_style)
    draw_line(draw, (width, height), (0, height), border_color, border_width, border_style)
    draw_line(draw, (0, height), (0, 0), border_color, border_width, border_style)

    rotation_angle = random.uniform(*rotation_range)
    image = image.rotate(rotation_angle, resample=Image.BICUBIC, expand=True)

    new_width, new_height = image.size

    # 레이블 조정
    adjusted_labels = []
    for label in labels:
        parts = label.split()
        x, y, w, h = map(float, parts[1:5])
        angle_rad = math.radians(rotation_angle)
        cos_angle = math.cos(angle_rad)
        sin_angle = math.sin(angle_rad)
        
        cx = x + w/2 - 0.5
        cy = y + h/2 - 0.5
        new_cx = cx * cos_angle - cy * sin_angle + 0.5
        new_cy = cx * sin_angle + cy * cos_angle + 0.5
        
        new_w = w * (width / new_width)
        new_h = h * (height / new_height)
        
        new_x = max(0, min(1, new_cx - new_w/2))
        new_y = max(0, min(1, new_cy - new_h/2))
        new_w = max(0, min(1, new_w))
        new_h = max(0, min(1, new_h))
        
        adjusted_label = f"0 {new_x:.6f} {new_y:.6f} {new_w:.6f} {new_h:.6f}"
        adjusted_labels.append(adjusted_label)

    image = apply_noise_and_distortions(image)
    return image, adjusted_labels

def apply_noise_and_distortions(image):
    try:
        if random.random() < GAUSSIAN_NOISE_PROBABILITY:
            noise = np.random.normal(GAUSSIAN_NOISE_MEAN, GAUSSIAN_NOISE_STD, image.size[::-1])
            noise = Image.fromarray(noise.astype('uint8'), mode='L').resize(image.size)
            image = Image.blend(image.convert('L'), noise, GAUSSIAN_NOISE_BLEND)
        
        if random.random() < BRIGHTNESS_ADJUST_PROBABILITY:
            enhancer = ImageEnhance.Brightness(image)
            image = enhancer.enhance(random.uniform(BRIGHTNESS_MIN, BRIGHTNESS_MAX))
        if random.random() < CONTRAST_ADJUST_PROBABILITY:
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(random.uniform(CONTRAST_MIN, CONTRAST_MAX))
        
        if random.random() < BLUR_PROBABILITY:
            image = image.filter(ImageFilter.GaussianBlur(radius=random.uniform(0, BLUR_MAX_RADIUS)))
        
        if random.random() < COMPRESSION_PROBABILITY:
            buffer = BytesIO()
            image.save(buffer, format="JPEG", quality=random.randint(COMPRESSION_MIN_QUALITY, COMPRESSION_MAX_QUALITY))
            image = Image.open(buffer)
   
        
    except Exception as e:
        print(f"Error in apply_noise_and_distortions: {e}")
        print(f"Image mode: {image.mode}, size: {image.size}")
        raise

    return image.convert('L')

def generate_table_wrapper(args):
    width, height, complexity, rotation_range = args
    return generate_table(width, height, complexity, rotation_range)

def generate_dataset(num_images, output_dir, complexities, train_ratio=TRAIN_RATIO, rotation_range=(MIN_ROTATION, MAX_ROTATION), complexity_weights=None):
    create_directory(output_dir)
    train_dir = os.path.join(output_dir, "train")
    val_dir = os.path.join(output_dir, "val")
    create_directory(train_dir)
    create_directory(val_dir)

    for data_dir in [train_dir, val_dir]:
        for complexity in complexities:
            create_directory(os.path.join(data_dir, "images", complexity))
            create_directory(os.path.join(data_dir, "labels", complexity))

    if complexity_weights is None:
        complexity_weights = COMPLEXITY_WEIGHTS

    with multiprocessing.Pool() as pool:
        tasks = [(random.randint(MIN_IMAGE_WIDTH, MAX_IMAGE_WIDTH), 
                  random.randint(MIN_IMAGE_HEIGHT, MAX_IMAGE_HEIGHT), 
                  random.choices(complexities, weights=complexity_weights)[0], 
                  rotation_range) for _ in range(num_images)]
        
        for i, (image, labels) in enumerate(tqdm(pool.imap_unordered(generate_table_wrapper, tasks), total=num_images)):
            is_train = random.random() < train_ratio
            data_dir = train_dir if is_train else val_dir
            complexity = tasks[i][2]

            image_filename = f"table_{i+1}.png"
            label_filename = f"table_{i+1}.txt"

            image_path = os.path.join(data_dir, "images", complexity, image_filename)
            label_path = os.path.join(data_dir, "labels", complexity, label_filename)
            extra_info_path = os.path.join(data_dir, "extra_info", complexity, f"table_{i+1}.txt")

            image.save(image_path)
            with open(label_path, 'w') as f:
                for label in labels:
                    parts = label.split()
                    f.write(f"{' '.join(parts[:5])}\n")

            create_directory(os.path.join(data_dir, "extra_info", complexity))
            with open(extra_info_path, 'w') as f:
                for label in labels:
                    parts = label.split()
                    f.write(f"{' '.join(parts[5:])}\n")

    yaml_content = f"""
    path: {output_dir}
    train: train/images
    val: val/images
    nc: 1
    names: ['cell']

    # 추가 정보 경로
    train_extra_info: train/extra_info
    val_extra_info: val/extra_info
    """

    with open(os.path.join(output_dir, 'dataset.yaml'), 'w') as f:
        f.write(yaml_content)

    print("Dataset YAML file created.")
    
def main():
    generate_dataset(TOTAL_IMAGES, "enhanced_tables_dataset", COMPLEXITIES, 
                     complexity_weights=COMPLEXITY_WEIGHTS)

if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()
