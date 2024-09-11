# config.py

# 데이터베이스 설정
DATABASE_URL = "postgresql://postgres.rhidgqfdbgyqxvoylrsv:QBUXKN8B8LwnB7zb@aws-0-ap-northeast-2.pooler.supabase.com:6543/postgres"
# 로깅 설정
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"

# 기타 설정
DEFAULT_SLEEP_START = "23:00"
DEFAULT_SLEEP_END = "07:00"


#OCR
IMAGES_COLLEGE = ["OCR/College/OCR1.jpg", "OCR/College/ocr2.jpg", "OCR/College/ocr3.jpg", "results/tables/ocr8.jpg"]
IMAGES_YOUNG = ["OCR/young/OCR2.png", "OCR/young/OCR4.png", "OCR/young/OCR5-temp.png", "results/tables/OCR6.png", "OCR/young/OCR7.png"]
RESULT_COLLEGE = "results/college/"
RESULT_YOUNG = "results/young/"
RESULT_TABLES = "results/tables/"

YOUNG_TABLES = "results/young/"
COLLEGE_TABLES = "results/college/"
