import random, string
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from dataset_utils import *
from dataset_constant import *
from opti_calc import *
def draw_table(draw, cells, table_bbox, bg_color, has_gap, is_imperfect):
    line_widths = []
    line_color = get_line_color(bg_color)
    merged_cells = set()
    removed_lines = set()
    
    # 병합된 셀 정보 추적
    for cell in cells:
        if len(cell) > 6 and cell[6]:
            for r in range(cell[7], cell[9] + 1):
                for c in range(cell[8], cell[10] + 1):
                    merged_cells.add((r, c))
    
    # 외곽선 그리기
    if random.random() > 0.2:
        outer_line_thickness = random.randint(1, 5)
        corners = [
            (table_bbox[0], table_bbox[1]),
            (table_bbox[2], table_bbox[1]),
            (table_bbox[2], table_bbox[3]),
            (table_bbox[0], table_bbox[3])
        ]
        for i in range(4):
            start, end = corners[i], corners[(i+1)%4]
            draw_cell_line(draw, start, end, line_color, outer_line_thickness, is_imperfect)
            line_widths.append(outer_line_thickness)

    # 각 셀에 대한 선 그리기
    for cell in cells:
        line_thickness = random.randint(1, 3)
        row, col = cell[4], cell[5]
        is_merged = (row, col) in merged_cells
        
        # 셀의 각 면에 대한 좌표
        top = ((cell[0], cell[1]), (cell[2], cell[1]))
        bottom = ((cell[0], cell[3]), (cell[2], cell[3]))
        left = ((cell[0], cell[1]), (cell[0], cell[3]))
        right = ((cell[2], cell[1]), (cell[2], cell[3]))
        
        # 상단과 하단 선은 항상 그리기
        draw_cell_line(draw, *top, line_color, line_thickness, is_imperfect)
        draw_cell_line(draw, *bottom, line_color, line_thickness, is_imperfect)
        
        # 병합된 셀, 갭이 있는 표, 또는 주변에 선이 제거된 셀이 있는 경우 모든 선 그리기
        if is_merged or has_gap or any((row+dr, col+dc) in removed_lines for dr, dc in [(0,1), (0,-1), (1,0), (-1,0)]):
            draw_cell_line(draw, *left, line_color, line_thickness, is_imperfect)
            draw_cell_line(draw, *right, line_color, line_thickness, is_imperfect)
        else:
            # 좌우 선 중 하나 또는 둘 다 그리기
            sides_to_draw = random.sample([left, right], random.randint(1, 2))
            for side in sides_to_draw:
                draw_cell_line(draw, *side, line_color, line_thickness, is_imperfect)
            
            if len(sides_to_draw) < 2:
                removed_lines.add((row, col))

        line_widths.extend([line_thickness] * 4)

    # 불완전한 테이블 생성 (일부 셀 숨기기)
    if is_imperfect and len(cells) > 4:
        num_cells_to_hide = random.randint(1, max(1, len(cells) // 4))
        cells_to_hide = random.sample([cell for cell in cells if (cell[4], cell[5]) not in merged_cells and (cell[4], cell[5]) not in removed_lines], num_cells_to_hide)
        for cell in cells_to_hide:
            draw.rectangle([cell[0], cell[1], cell[2], cell[3]], fill=bg_color)

    return line_widths, bool(line_widths)  # 외곽선이 그려졌는지 여부 반환



def draw_cell_line(draw, start, end, color, thickness, is_imperfect):
    """
    셀의 경계선을 그리는 함수입니다.
    
    :param draw: PIL.ImageDraw 객체
    :param start: 선의 시작점 (x, y) 튜플
    :param end: 선의 끝점 (x, y) 튜플
    :param color: 선의 색상 (RGB 튜플 또는 정수)
    :param thickness: 선의 두께
    :param is_imperfect: 불완전한 선을 그릴지 여부 (True/False)
    """
    x1, y1 = start
    x2, y2 = end
    
    # 기본 선 그리기
    draw.line((start, end), fill=color, width=thickness)
    
    if is_imperfect:
        # 20% 확률로 돌출부 추가
        if random.random() < 0.2:
            mid_x, mid_y = (x1 + x2) // 2, (y1 + y2) // 2  # 선의 중점 계산
            protrusion_thickness = random.randint(1, 5)  # 돌출부 두께 랜덤 설정
            protrusion_length = random.randint(1, 3)  # 돌출부 길이 랜덤 설정
            
            # color가 튜플인 경우 각 채널별로 255를 초과하지 않도록 처리
            if isinstance(color, tuple):
                protrusion_color = tuple(min(255, c) for c in color)
            else:
                protrusion_color = min(255, color)
            
            # 수직선 또는 수평선에 따라 돌출부 방향 결정
            if x1 == x2:  # 수직선
                draw.line((mid_x, mid_y, mid_x + protrusion_length, mid_y), fill=protrusion_color, width=protrusion_thickness)
            else:  # 수평선
                draw.line((mid_x, mid_y, mid_x, mid_y + protrusion_length), fill=protrusion_color, width=protrusion_thickness)
        
        # 20% 확률로 점 추가
        if random.random() < 0.2:
            num_dots = random.randint(1, 3)  # 추가할 점의 개수 랜덤 설정
            for _ in range(num_dots):
                # 선 위의 랜덤한 위치에 점 추가
                dot_x = random.randint(min(x1, x2), max(x1, x2))
                dot_y = random.randint(min(y1, y2), max(y1, y2))
                dot_size = random.randint(1, 2)  # 점의 크기 랜덤 설정
                draw.ellipse((dot_x, dot_y, dot_x + dot_size, dot_y + dot_size), fill=color)
def apply_imperfections(img, cells):
    draw = ImageDraw.Draw(img)
    width, height = img.size

    if random.random() < 0.3:
        img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.3, 0.7)))
    
    if random.random() < 0.3:
        for _ in range(random.randint(1, 3)):
            x1, y1 = random.randint(0, width), random.randint(0, height)
            x2, y2 = random.randint(0, width), random.randint(0, height)
            gap_start = random.uniform(0.3, 0.7)
            gap_end = gap_start + random.uniform(0.1, 0.3)
            
            draw.line((x1, y1, x1 + (x2-x1)*gap_start, y1 + (y2-y1)*gap_start), fill=255, width=1)
            draw.line((x1 + (x2-x1)*gap_end, y1 + (y2-y1)*gap_end, x2, y2), fill=255, width=1)
    
    for cell in cells:
        if random.random() < 0.1:
            x1, y1, x2, y2 = cell[:4]
            start_x, start_y = random.randint(x1, x2), random.randint(y1, y2)
            end_x, end_y = random.randint(x1, x2), random.randint(y1, y2)
            draw.line((start_x, start_y, end_x, end_y), fill=0, width=random.randint(1, 3))
    
    return img
def add_shapes_to_cell(draw, cell, bg_color, x, y):
    cell_width, cell_height = cell[2] - cell[0], cell[3] - cell[1]
    max_size = min(cell_width, cell_height) // 2
    
    color = get_line_color(bg_color)
    
    for _ in range(random.randint(1, 3)):
        shape_type = random.choice(['rectangle', 'ellipse', 'triangle', 'line', 'arc', 'polygon'])
        size = random.randint(max_size // 2, max_size)
        angle = random.randint(0, 360)
        
        shape_img = Image.new('L', (size, size), color=0)
        shape_draw = ImageDraw.Draw(shape_img)
        
        draw_shape(shape_draw, shape_type, size)
        
        rotated_shape = shape_img.rotate(angle, expand=True)
        
        paste_x = max(cell[0], min(x - rotated_shape.width // 2, cell[2] - rotated_shape.width))
        paste_y = max(cell[1], min(y - rotated_shape.height // 2, cell[3] - rotated_shape.height))
        
        draw.bitmap((paste_x, paste_y), rotated_shape, fill=color)
        #print(f"도형 추가됨, {paste_x}, {paste_y}, {color}")  # 디버깅용 출력
def adjust_font_size(draw, text, font, max_width, max_height):
    while font.size > 8:  # 최소 폰트 크기 설정
        bbox = draw.multiline_textbbox((0, 0), text, font=font)
        if bbox[2] - bbox[0] <= max_width and bbox[3] - bbox[1] <= max_height:
            break
        font = ImageFont.truetype(font.path, font.size - 1)
    return font
def add_text_to_cell(draw, cell, font_path, text_color, x, y, position):
    cell_width, cell_height = cell[2] - cell[0], cell[3] - cell[1]
    padding = 4  # 셀 경계로부터의 여백
    
    # 폰트 크기 조정
    font_size = max(8, min(int(cell_height * 0.2), int(cell_width * 0.1)))
    font = ImageFont.truetype(font_path, font_size)
    
    # 텍스트 생성
    text = random_text(min_length=1, max_length=max(1, min(20, cell_width // font_size)))
    
    # 텍스트 줄바꿈
    wrapped_text = wrap_text(text, font, cell_width - 2*padding)
    
    # 텍스트 크기 계산
    bbox = draw.multiline_textbbox((0, 0), wrapped_text, font=font)
    text_width, text_height = bbox[2] - bbox[0], bbox[3] - bbox[1]
    
    # 텍스트가 셀을 벗어나는 경우 폰트 크기 조정
    while text_height > cell_height - 2*padding or text_width > cell_width - 2*padding:
        font_size -= 1
        if font_size < 8:  # 최소 폰트 크기
            font_size = 8
            break
        font = ImageFont.truetype(font_path, font_size)
        wrapped_text = wrap_text(text, font, cell_width - 2*padding)
        bbox = draw.multiline_textbbox((0, 0), wrapped_text, font=font)
        text_width, text_height = bbox[2] - bbox[0], bbox[3] - bbox[1]
    
    # 텍스트 위치 계산
    text_position = calculate_text_position(cell[0]+padding, cell[1]+padding, 
                                            cell_width-2*padding, cell_height-2*padding, 
                                            text_width, text_height, position)
    
    # 텍스트 그리기
    draw.multiline_text(text_position, wrapped_text, font=font, fill=text_color, align='center')
    
    
    return wrapped_text
def add_content_to_cells(draw, cells, font_path, bg_color, empty_cell_ratio=0.2):
    #print(f"Debug: add_content_to_cells - bg_color = {bg_color}, type = {type(bg_color)}")
    for cell in cells:
        if random.random() < empty_cell_ratio:
            continue
        
        content_type = random.choice(['text', 'shapes', 'mixed'])
        position = random.choice(['center', 'top_left', 'top_right', 'bottom_left', 'bottom_right'])
        
        cell_width = cell[2] - cell[0]
        cell_height = cell[3] - cell[1]
        
        x, y = calculate_position(cell, position)
        
        content_color = get_line_color(bg_color)
        
        if content_type in ['text', 'mixed']:
            text = add_text_to_cell(draw, cell, font_path, content_color, x, y, position)
        
        if content_type in ['shapes', 'mixed']:
            add_shapes_to_cell(draw, cell, bg_color, x, y)
        
        #print(f"셀 정보: 위치({cell[0]}, {cell[1]}, {cell[2]}, {cell[3]}), 크기({cell_width}x{cell_height})")


def add_noise_around_text(draw, x, y, width, height, bg_color):
    noise_color = get_line_color(bg_color)
    num_noise = random.randint(5, 20)
    for _ in range(num_noise):
        noise_x = x + random.randint(-width//2, width//2)
        noise_y = y + random.randint(-height//2, height//2)
        noise_size = random.randint(1, 3)
        if random.random() < 0.7:
            draw.ellipse([noise_x, noise_y, noise_x+noise_size, noise_y+noise_size], fill=noise_color)
        else:
            end_x = noise_x + random.randint(-5, 5)
            end_y = noise_y + random.randint(-5, 5)
            draw.line([noise_x, noise_y, end_x, end_y], fill=noise_color, width=1)
def add_shapes_to_cell(draw, cell, bg_color, x, y):
    cell_width, cell_height = cell[2] - cell[0], cell[3] - cell[1]
    max_size = min(cell_width, cell_height) // 2
    
    color = get_line_color(bg_color)
    
    for _ in range(random.randint(1, 3)):
        shape_type = random.choice(['rectangle', 'ellipse', 'triangle', 'line', 'arc', 'polygon'])
        size = random.randint(max_size // 2, max_size)
        angle = random.randint(0, 360)
        
        shape_img = Image.new('L', (size, size), color=0)
        shape_draw = ImageDraw.Draw(shape_img)
        
        draw_shape(shape_draw, shape_type, size)
        
        rotated_shape = shape_img.rotate(angle, expand=True)
        
        paste_x = max(cell[0], min(x - rotated_shape.width // 2, cell[2] - rotated_shape.width))
        paste_y = max(cell[1], min(y - rotated_shape.height // 2, cell[3] - rotated_shape.height))
        
        draw.bitmap((paste_x, paste_y), rotated_shape, fill=color)
        #print(f"도형 추가됨: 유형({shape_type}), 위치({paste_x}, {paste_y}), 크기({size}), 색상{color}")

def draw_shape(shape_draw, shape_type, size):
    if shape_type == 'rectangle':
        shape_draw.rectangle([0, 0, size-1, size-1], outline=255, width=2)
    elif shape_type == 'ellipse':
        shape_draw.ellipse([0, 0, size-1, size-1], outline=255, width=2)
    elif shape_type == 'triangle':
        shape_draw.polygon([(size//2, 0), (0, size-1), (size-1, size-1)], outline=255, width=2)
    elif shape_type == 'line':
        shape_draw.line([(0, 0), (size-1, size-1)], fill=255, width=2)
    elif shape_type == 'arc':
        shape_draw.arc([0, 0, size-1, size-1], 0, 270, fill=255, width=2)
    elif shape_type == 'polygon':
        points = [(random.randint(0, size-1), random.randint(0, size-1)) for _ in range(5)]
        shape_draw.polygon(points, outline=255, width=2)