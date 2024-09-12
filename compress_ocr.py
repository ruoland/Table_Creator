from PIL import Image
import os, cv2, numpy as np

def compress_png(input_path, output_path, quality=50):
    # 이미지 열기
    with Image.open(input_path) as img:
        # 이미지를 RGB 모드로 변환 (알파 채널 제거)
        rgb_im = img.convert('RGB')
        rgb_im = preprocess_image(rgb_im)
        # 압축된 이미지 저장
        rgb_im.save(output_path, 'JPEG', quality=quality, optimize=True)
def to_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def apply_gaussian_blur(image, kernel_size=(5,5)):
    return cv2.GaussianBlur(image, kernel_size, 0)
def apply_adaptive_thresholding(image):
    return cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

def apply_thresholding(image):
    return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

def enhance_contrast(image):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    return clahe.apply(image)

def apply_morphology(image):
    kernel = np.ones((3,3), np.uint8)
    image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

#이미지 기울임 보정
def correct_skew(image):
    coords = np.column_stack(np.where(image > 0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

def preprocess_image(image):
    gray = to_grayscale(image)
    blurred = apply_gaussian_blur(gray)
    binary = apply_adaptive_thresholding(blurred)
    enhanced = enhance_contrast(binary)
    cleaned = apply_morphology(enhanced)
    #corrected = correct_skew(cleaned)
    return cleaned

# 사용 예시
input_folder = "table_dataset\\uniform\\images\\"
output_folder = "compressed_images"

# 출력 폴더가 없으면 생성
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 입력 폴더의 모든 PNG 파일 처리
for filename in os.listdir(input_folder):
    if filename.endswith(".png"):
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename.replace(".png", ".jpg"))
        compress_png(input_path, output_path)
        print(f"Compressed: {filename}")

print("Compression completed.")
