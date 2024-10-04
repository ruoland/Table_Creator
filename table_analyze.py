import json
from collections import defaultdict

def analyze_annotations(annotation_file):
    with open(annotation_file, 'r') as f:
        data = json.load(f)
    
    stats = defaultdict(int)
    table_stats = defaultdict(list)
    cell_stats = defaultdict(int)
    
    for ann in data['annotations']:
        category_name = ann['category_name']
        stats[f'total_{category_name}s'] += 1
        
        if category_name == 'table':
            attrs = ann['attributes']
            table_stats['rows'].append(attrs['total_rows'])
            table_stats['cols'].append(attrs['total_cols'])
            table_stats['types'].append(attrs['table_type'])
            stats['tables_with_outer_border'] += attrs['has_outer_border']
            stats['tables_with_rounded_corners'] += attrs['has_rounded_corners']
            stats['tables_with_cell_gap'] += attrs['has_cell_gap']
            stats['tables_with_overflow'] += attrs['has_overflow']
            stats['tables_with_merged_cells'] += attrs['has_merged_cells']
            stats['tables_with_gray_cells'] += attrs['has_gray_cells']
        
        elif category_name == 'cell':
            attrs = ann['attributes']
            cell_stats['header_cells'] += attrs['is_header']
            cell_stats['merged_cells'] += attrs['is_merged']
            cell_stats['cells_with_overflow'] += attrs['has_overflow']
            cell_stats['cells_affected_by_overflow'] += attrs['is_affected_by_overflow']
            cell_stats['gray_cells'] += attrs['is_gray']
    
    # 테이블 통계 계산
    stats['avg_rows'] = sum(table_stats['rows']) / len(table_stats['rows'])
    stats['avg_cols'] = sum(table_stats['cols']) / len(table_stats['cols'])
    stats['max_rows'] = max(table_stats['rows'])
    stats['max_cols'] = max(table_stats['cols'])
    stats['min_rows'] = min(table_stats['rows'])
    stats['min_cols'] = min(table_stats['cols'])
    
    # 테이블 유형 분포
    type_distribution = defaultdict(int)
    for t in table_stats['types']:
        type_distribution[t] += 1
    stats['table_type_distribution'] = dict(type_distribution)
    
    # 셀 통계를 전체 통계에 추가
    stats.update(cell_stats)
    
    return stats

def print_stats(stats):
    print("=== 어노테이션 통계 ===")
    print(f"총 테이블 수: {stats['total_tables']}")
    print(f"총 셀 수: {stats['total_cells']}")
    print(f"총 행 수: {stats['total_rows']}")
    print(f"총 열 수: {stats['total_columns']}")
    
    print("\n=== 테이블 구조 ===")
    print(f"평균 행 수: {stats['avg_rows']:.2f}")
    print(f"평균 열 수: {stats['avg_cols']:.2f}")
    print(f"최대 행 수: {stats['max_rows']}")
    print(f"최대 열 수: {stats['max_cols']}")
    print(f"최소 행 수: {stats['min_rows']}")
    print(f"최소 열 수: {stats['min_cols']}")
    
    print("\n=== 테이블 유형 분포 ===")
    for t, count in stats['table_type_distribution'].items():
        print(f"{t}: {count}")
    
    print("\n=== 테이블 특성 ===")
    print(f"외곽선 있는 테이블: {stats['tables_with_outer_border']}")
    print(f"둥근 모서리 테이블: {stats['tables_with_rounded_corners']}")
    print(f"셀 간격 있는 테이블: {stats['tables_with_cell_gap']}")
    print(f"오버플로우 있는 테이블: {stats['tables_with_overflow']}")
    print(f"병합된 셀 있는 테이블: {stats['tables_with_merged_cells']}")
    print(f"회색 셀 있는 테이블: {stats['tables_with_gray_cells']}")
    
    print("\n=== 셀 특성 ===")
    print(f"헤더 셀: {stats['header_cells']}")
    print(f"병합된 셀: {stats['merged_cells']}")
    print(f"오버플로우 있는 셀: {stats['cells_with_overflow']}")
    print(f"오버플로우의 영향을 받는 셀: {stats['cells_affected_by_overflow']}")
    print(f"회색 셀: {stats['gray_cells']}")

if __name__ == "__main__":
    annotation_file = r"C:\project\table_color\train_annotations.json"  # 실제 파일 경로로 변경하세요
    stats = analyze_annotations(annotation_file)
    print_stats(stats)
