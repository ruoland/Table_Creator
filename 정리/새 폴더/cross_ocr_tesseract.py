from PIL import Image, ImageFilter
import pytesseract

# Tesseract 설치 경로 설정 (Windows 사용자만 해당)
pytesseract.pytesseract.tesseract_cmd = r'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

# 이미지 파일 경로
image_path = './OCR/young/OCR5.png'

# 이미지 열기
img = Image.open(image_path)

# 이미지 전처리: 그레이스케일 변환, 노이즈 제거, 명암비 조절 등
img = img.convert('L')  # 그레이스케일 변환
img = img.filter(ImageFilter.SHARPEN)  # 샤프닝 필터 적용

# 이미지에서 텍스트 추출
custom_config = r'--oem 3 --psm 6'  # OCR 엔진 모드 및 페이지 세그멘테이션 모드 설정
text = pytesseract.image_to_string(img, lang='kor', config=custom_config)

print(text)
