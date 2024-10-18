import ujson as json
import numpy as np

def analyze_cell_sizes(annotation_file):
    with open(annotation_file, 'r') as f:
        data = json.load(f)
    
    cell_areas = []
    cell_widths = []
    cell_heights = []
    
    for annotation in data['annotations']:
        if annotation['category_id'] == 0:
            x, y, width, height = annotation['bbox']
            area = width * height
            cell_areas.append(area)
            cell_widths.append(width)
            cell_heights.append(height)
    
    # 통계 계산
    avg_area = np.mean(cell_areas)
    median_area = np.median(cell_areas)
    std_area = np.std(cell_areas)
    
    avg_width = np.mean(cell_widths)
    avg_height = np.mean(cell_heights)
    
    # 크기별 분류
    small_cells = sum(1 for area in cell_areas if area < 5000)
    medium_cells = sum(1 for area in cell_areas if 5000 <= area < 30000)
    large_cells = sum(1 for area in cell_areas if area >= 30000)
    
    print(f"평균 셀 면적: {avg_area:.2f} 픽셀²")
    print(f"중앙값 셀 면적: {median_area:.2f} 픽셀²")
    print(f"셀 면적 표준편차: {std_area:.2f} 픽셀²")
    print(f"평균 셀 너비: {avg_width:.2f} 픽셀")
    print(f"평균 셀 높이: {avg_height:.2f} 픽셀")
    print(f"소형 셀 (< 5000 픽셀²): {small_cells}")
    print(f"중형 셀 (5000-30000 픽셀²): {medium_cells}")
    print(f"대형 셀 (>= 30000 픽셀²): {large_cells}")
    
    # 히스토그램 데이터 반환 (시각화에 사용 가능)
    return np.histogram(cell_areas, bins=[0, 5000, 30000, np.inf])

# 사용 예시
annotation_file = r"C:\project\table_overflow\train_annotations.json"
hist, bin_edges = analyze_cell_sizes(annotation_file)

# 히스토그램 데이터 출력
print("셀 크기 분포:")
for i, count in enumerate(hist):
    print(f"{bin_edges[i]:.0f} - {bin_edges[i+1]:.0f} 픽셀²: {count}")
