import os, yaml, json, csv
import numpy as np

def save_dataset_info(output_dir, dataset_info, stats):
    total_images = sum(len(info) for info in dataset_info.values())
    print(f"총 생성된 이미지 수: {total_images}")

    for subset, info in dataset_info.items():
        print(f"{subset.capitalize()} 세트:")
        print(f"  총 이미지 수: {len(info)}")
        print(f"  밝은 이미지 수: {sum(1 for item in info if item['bg_mode'] == 'light')}")
        print(f"  어두운 이미지 수: {sum(1 for item in info if item['bg_mode'] == 'dark')}")
        print(f"  간격 있는 이미지 수: {sum(1 for item in info if item['has_gap'])}")
        print(f"  불완전 이미지 수: {sum(1 for item in info if item['is_imperfect'])}")
        print()

    yaml_content = {
        'train': os.path.join(output_dir, 'train', 'images'),
        'val': os.path.join(output_dir, 'val', 'images'),
        'test': os.path.join(output_dir, 'test', 'images'),
        'nc': 5,
        'names': ['cell', 'merged_cell', 'row', 'column', 'table']
    }
    
    with open(os.path.join(output_dir, 'dataset.yaml'), 'w') as f:
        yaml.dump(yaml_content, f)

    with open(os.path.join(output_dir, 'dataset_stats.csv'), 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=stats[0].keys())
        writer.writeheader()
        writer.writerows(stats)

    summary_stats = calculate_summary_stats(stats)
    with open(os.path.join(output_dir, 'summary_stats.json'), 'w', encoding='utf-8') as f:
        json.dump(summary_stats, f, indent=4, ensure_ascii=False)
def calculate_summary_stats(stats):
    summary = {
        '총 이미지 수': len(stats),
        '불완전 이미지 비율': sum(1 for s in stats if s['is_imperfect']) / len(stats),
        '배경 모드 비율': {
            '밝음': sum(1 for s in stats if s['bg_mode'] == 'light') / len(stats),
            '어두움': sum(1 for s in stats if s['bg_mode'] == 'dark') / len(stats)
        },
        '간격 있는 이미지 비율': sum(1 for s in stats if s['has_gap']) / len(stats),
        '데이터셋 분포': {
            '학습': sum(1 for s in stats if s['subset'] == 'train') / len(stats),
            '검증': sum(1 for s in stats if s['subset'] == 'val') / len(stats),
            '테스트': sum(1 for s in stats if s['subset'] == 'test') / len(stats)
        },
        '평균 이미지 크기': {
            '너비': np.mean([s['image_width'] for s in stats]),
            '높이': np.mean([s['image_height'] for s in stats])
        },
        '평균 셀 수': np.mean([s['num_cells'] for s in stats]),
        '평균 행 수': np.mean([s['rows'] for s in stats]),
        '평균 열 수': np.mean([s['cols'] for s in stats]),
        '평균 병합된 셀 수': np.mean([s['num_merged_cells'] for s in stats]),
        '평균 선 두께': np.mean([s['avg_line_width'] for s in stats]),
        '최소 선 두께': min([s['min_line_width'] for s in stats]),
        '최대 선 두께': max([s['max_line_width'] for s in stats]),
        '외곽선 있는 테이블 비율': sum(1 for s in stats if s['has_outer_lines']) / len(stats),
        '평균 테이블 크기': {
            '너비': np.mean([s['table_width'] for s in stats]),
            '높이': np.mean([s['table_height'] for s in stats])
        },
        '평균 셀 크기': {
            '너비': np.mean([s['avg_cell_width'] for s in stats]),
            '높이': np.mean([s['avg_cell_height'] for s in stats])
        }
    }
    return summary