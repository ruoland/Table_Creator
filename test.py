from PIL import Image, ImageDraw, ImageFilter
import cv2
import numpy as np
import random

def create_simple_table(width, height, rows, cols):
    image = Image.new('RGB', (width, height), color='white')
    draw = ImageDraw.Draw(image)

    cell_width = width // cols
    cell_height = height // rows

    cell_corners = []

    for i in range(rows + 1):
        y = i * cell_height
        draw.line([(0, y), (width, y)], fill='black', width=2)
        for j in range(cols + 1):
            x = j * cell_width
            if i == 0:
                draw.line([(x, 0), (x, height)], fill='black', width=2)
            cell_corners.append((x, y))

    return image, cell_corners

def draw_cell_corner_points(image, corners, radius=3, color=(255, 0, 0)):
    draw = ImageDraw.Draw(image)
    for corner in corners:
        x, y = corner
        draw.ellipse([x-radius, y-radius, x+radius, y+radius], fill=color)
    return image

def apply_perspective_transform(image, corners, intensity=0.2):
    width, height = image.size
    pts1 = np.float32([[0, 0], [width-1, 0], [width-1, height-1], [0, height-1]])
    pts2 = np.float32([[random.uniform(0, width*intensity), random.uniform(0, height*intensity)],
                       [random.uniform(width*(1-intensity), width-1), random.uniform(0, height*intensity)],
                       [random.uniform(width*(1-intensity), width-1), random.uniform(height*(1-intensity), height-1)],
                       [random.uniform(0, width*intensity), random.uniform(height*(1-intensity), height-1)]])
    
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    cv2_image = pil_to_cv2(image)
    result = cv2.warpPerspective(cv2_image, matrix, (width, height))
    
    # Transform cell corners
    corners_array = np.array(corners, dtype=np.float32).reshape(-1, 1, 2)
    transformed_corners = cv2.perspectiveTransform(corners_array, matrix).reshape(-1, 2)
    
    return cv2_to_pil(result), transformed_corners.tolist()

def process_image(image, corners, apply_wrinkle=True, apply_perspective=True, apply_distortion=True, draw_corners=True):
    if apply_wrinkle:
        image = add_enhanced_wrinkle_effect(image, intensity=0.8, num_wrinkles=10)
    
    if apply_perspective:
        image, corners = apply_perspective_transform(image, corners, intensity=0.2)
    
    if apply_distortion:
        cv2_image = pil_to_cv2(image)
        cv2_image = apply_lens_distortion(cv2_image, k1=0.1, k2=0.1)
        image = cv2_to_pil(cv2_image)
    
    if draw_corners:
        image = draw_cell_corner_points(image, corners)
    
    return image, corners


def add_enhanced_wrinkle_effect(image, intensity=0.3, num_wrinkles=10):
    width, height = image.size
    wrinkle = Image.new('L', (width, height), 255)
    draw = ImageDraw.Draw(wrinkle)
    
    for _ in range(num_wrinkles):
        start = (random.randint(0, width), random.randint(0, height))
        end = (random.randint(0, width), random.randint(0, height))
        draw.line([start, end], fill=100, width=random.randint(1, 3))
    
    wrinkle = wrinkle.filter(ImageFilter.GaussianBlur(radius=3))
    
    img_array = np.array(image)
    wrinkle_array = np.array(wrinkle)
    
    result_array = img_array * (wrinkle_array[:,:,np.newaxis] / 255.0 * intensity + (1 - intensity))
    result_array = result_array + np.random.randint(-10, 10, result_array.shape)
    result_array = np.clip(result_array, 0, 255).astype(np.uint8)
    
    result = Image.fromarray(result_array)
    return result

def pil_to_cv2(pil_image):
    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

def cv2_to_pil(cv2_image):
    return Image.fromarray(cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB))

def apply_lens_distortion(image, k1=0.1, k2=0.1):
    height, width = image.shape[:2]
    camera_matrix = np.array([[width, 0, width/2],
                              [0, height, height/2],
                              [0, 0, 1]], dtype=np.float32)
    dist_coeffs = np.array([k1, k2, 0, 0], dtype=np.float32)
    
    undistorted = cv2.undistort(image, camera_matrix, dist_coeffs)
    return undistorted

def draw_corner_points(image, corners, radius=5, color=(0, 255, 0), thickness=-1):
    for corner in corners:
        cv2.circle(image, tuple(map(int, corner)), radius, color, thickness)
    return image

if __name__ == "__main__":
    width, height = 800, 600
    rows, cols = 6, 4
    
    table_image, cell_corners = create_simple_table(width, height, rows, cols)
    
    # 모든 효과 적용
    processed_image, processed_corners = process_image(table_image, cell_corners, 
                                                       apply_wrinkle=True, 
                                                       apply_perspective=True, 
                                                       apply_distortion=True, 
                                                       draw_corners=True)
    
    # 원본 이미지 저장 (셀 모서리 점 포함)
    draw_cell_corner_points(table_image, cell_corners).save("original_table_with_corners.png")
    
    # 처리된 이미지 저장
    processed_image.save("processed_table_with_corners.png")
    
    print("이미지가 생성되었습니다: original_table_with_corners.png, processed_table_with_corners.png")

    # 각 효과를 개별적으로 적용한 이미지 생성
    wrinkle_only, _ = process_image(table_image, cell_corners, apply_wrinkle=True, apply_perspective=False, apply_distortion=False, draw_corners=True)
    perspective_only, _ = process_image(table_image, cell_corners, apply_wrinkle=False, apply_perspective=True, apply_distortion=False, draw_corners=True)
    distortion_only, _ = process_image(table_image, cell_corners, apply_wrinkle=False, apply_perspective=False, apply_distortion=True, draw_corners=True)

    wrinkle_only.save("wrinkle_only_with_corners.png")
    perspective_only.save("perspective_only_with_corners.png")
    distortion_only.save("distortion_only_with_corners.png")

    print("개별 효과 이미지가 생성되었습니다: wrinkle_only_with_corners.png, perspective_only_with_corners.png, distortion_only_with_corners.png")
