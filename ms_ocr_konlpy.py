import logging
import numpy as np
import cv2
import easyocr
import pandas as pd
import yaml
from difflib import SequenceMatcher
from konlpy.tag import Okt

def load_config(config_path='config.yaml'):
    with open(config_path, 'r', encoding='UTF-8') as file:
        return yaml.safe_load(file)

def setup_logging():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_image(image_path):
    logging.info(f"이미지 로드 중: {image_path}")
    return cv2.imread(image_path)

def string_similarity(a, b):
    return SequenceMatcher(None, a, b).ratio()

def post_process_text(text, known_words, okt, similarity_threshold=0.8):
    # 형태소 분석
    morphs = okt.morphs(text)
    
    # 명사 추출
    nouns = okt.nouns(text)
    
    # 알려진 단어와 직접 비교
    for word in known_words:
        if word in morphs or word in nouns:
            return word
    
    # 유사도 비교
    best_match = max(known_words, key=lambda x: string_similarity(text, x))
    if string_similarity(text, best_match) > similarity_threshold:
        return best_match
    
    return text

def perform_ocr(image, reader, known_words, okt):
    logging.info("OCR 수행 중")
    results = reader.readtext(image, paragraph=False, detail=1, contrast_ths=0.1, adjust_contrast=0.8)
    
    processed_results = []
    for (bbox, text, prob) in results:
        processed_text = post_process_text(text, known_words, okt)
        processed_results.append((bbox, processed_text, prob))
    
    return processed_results

def structure_timetable(ocr_result, image_shape):
    logging.info("시간표 구조화 중")
    height, width = image_shape[:2]
    table_structure = []
    for (box, text, prob) in ocr_result:
        (top_left, _, bottom_right, _) = box
        row = int(top_left[1] / height * 10)  # 임시로 10개 구간으로 나눔
        col = int(top_left[0] / width * 10)   # 임시로 10개 구간으로 나눔
        table_structure.append({"row": row, "col": col, "text": text})
    
    df = pd.DataFrame(table_structure)
    df = df.sort_values(['row', 'col'])
    
    # 실제 행과 열 수 계산
    unique_rows = df['row'].nunique()
    unique_cols = df['col'].nunique()
    
    # row와 col 값을 0부터 시작하는 연속된 정수로 변환
    df['row'] = pd.factorize(df['row'])[0]
    df['col'] = pd.factorize(df['col'])[0]
    
    timetable = df.pivot(index='row', columns='col', values='text')
    return timetable.fillna(''), (unique_rows, unique_cols)

def validate_timetable(timetable, expected_shape):
    logging.info("시간표 검증 중")
    actual_shape = timetable.shape
    if actual_shape != expected_shape:
        logging.warning(f"시간표 형태가 예상과 다릅니다. 예상: {expected_shape}, 실제: {actual_shape}")
    return actual_shape

def main():
    config = load_config()
    setup_logging()

    logging.info("프로그램 시작")

    # 알려진 단어 목록
    known_words = ['창체', '국어', '수학', '영어', '과학', '사회', '체육', '음악', '미술', '기술가정', '일어', ''
                   '월', '화', '수', '목', '금', '1교시', '2교시', '3교시', '4교시', '5교시', '6교시', '7교시']

    # OCR 및 형태소 분석기 설정
    reader = easyocr.Reader(['ko', 'en'], gpu=False)
    okt = Okt()

    # 이미지 로드
    image = load_image(config['image_path'])

    # OCR 수행
    ocr_result = perform_ocr(image, reader, known_words, okt)

    # 시간표 구조화
    timetable, actual_shape = structure_timetable(ocr_result, image.shape)

    # 결과 검증
    expected_shape = tuple(config['expected_shape'])
    validated_shape = validate_timetable(timetable, expected_shape)

    logging.info(f"시간표 구조화 완료. 크기: {validated_shape}")
    print("시간표:")
    print(timetable.to_string(index=False))

    # CSV 파일로 저장
    csv_file_path = config['/timetable.csv']
    timetable.to_csv(csv_file_path, index=False)
    logging.info(f"시간표가 {csv_file_path}에 저장되었습니다.")

if __name__ == "__main__":
    main()