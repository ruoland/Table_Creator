import json
import cv2
import numpy as np
def load_json(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        return json.load(f)

def create_grid(results):
    # 모든 텍스트의 중심점 좌표 추출
    points = np.array([[
        (r['coordinates']['x_min'] + r['coordinates']['x_max']) / 2,
        (r['coordinates']['y_min'] + r['coordinates']['y_max']) / 2
    ] for r in results])
    
    # X 좌표와 Y 좌표의 고유한 값을 찾아 정렬
    unique_x = np.unique(points[:, 0])
    unique_y = np.unique(points[:, 1])
    
    # 그리드 크기 결정
    rows = len(unique_y)
    cols = len(unique_x)
    
    # 빈 그리드 생성
    grid = [['' for _ in range(cols)] for _ in range(rows)]
    
    # 각 텍스트를 그리드에 배치
    for result in results:
        x = (result['coordinates']['x_min'] + result['coordinates']['x_max']) / 2
        y = (result['coordinates']['y_min'] + result['coordinates']['y_max']) / 2
        
        # 가장 가까운 그리드 위치 찾기
        col = np.argmin(np.abs(unique_x - x))
        row = np.argmin(np.abs(unique_y - y))
        
        grid[row][col] = result['text']
    
    return grid

def compare_results(paddle_results, easy_results):
    paddle_words = set(r['text'] for r in paddle_results)
    easy_words = set(r['text'] for r in easy_results)
    
    comparison = {
        "paddle_word_count": len(paddle_results),
        "easy_word_count": len(easy_results),
        "common_words": list(paddle_words.intersection(easy_words)),
        "paddle_unique": list(paddle_words - easy_words),
        "easy_unique": list(easy_words - paddle_words),
        "paddle_avg_confidence": sum(r['confidence'] for r in paddle_results) / len(paddle_results),
        "easy_avg_confidence": sum(r['confidence'] for r in easy_results) / len(easy_results)
    }
    
    return comparison

def main():
    # JSON 파일 로드
    paddle_results = load_json('paddle_ocr_results.json')
    easy_results = load_json('easy_ocr_results.json')
    
    # 결과 비교
    comparison = compare_results(paddle_results, easy_results)
    
    # 그리드 생성
    paddle_grid = create_grid(paddle_results)
    easy_grid = create_grid(easy_results)
    
    # 결과 출력
    print("OCR 비교 결과:")
    print(json.dumps(comparison, ensure_ascii=False, indent=2))
    
    print("\nPaddleOCR 그리드:")
    for row in paddle_grid:
        print(" | ".join(cell or "<빈>" for cell in row))
    
    print("\nEasyOCR 그리드:")
    for row in easy_grid:
        print(" | ".join(cell or "<빈>" for cell in row))

if __name__ == "__main__":
    main()