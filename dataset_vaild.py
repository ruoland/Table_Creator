import ujson as json
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import os, numpy as np

def validate_annotations(image_path, annotation_path, output_dir, specific_image_id=None):
    with open(annotation_path, 'r') as f:
        coco_data = json.load(f)

    image_id_to_filename = {img['id']: img['file_name'] for img in coco_data['images']}
    category_id_to_name = {cat['id']: cat['name'] for cat in coco_data['categories']}

    if specific_image_id is not None:
        selected_images = [img for img in coco_data['images'] if img['id'] == specific_image_id]
        if not selected_images:
            print(f"이미지 ID {specific_image_id}를 찾을 수 없습니다.")
            return
    else:
        selected_images = random.sample(coco_data['images'], min(10, len(coco_data['images'])))

    for image_info in selected_images:
        image_id = image_info['id']
        filename = image_id_to_filename[image_id]
        img_path = os.path.join(image_path, filename)

        print(f"\n=== 선택된 이미지 정보 (ID: {image_id}) ===")
        print("이미지 메타데이터:")
        print(json.dumps(image_info, indent=2, ensure_ascii=False))

        img = Image.open(img_path)
        print(f"이미지 크기: {img.size}")
        print(f"이미지 모드: {img.mode}")

        annotations = [ann for ann in coco_data['annotations'] if ann['image_id'] == image_id]
        print(f"\n이 이미지의 어노테이션 수: {len(annotations)}")

        # 사용자에게 어떤 요소를 보고 싶은지 물어봅니다
        print("\n어떤 요소를 보시겠습니까? (여러 개 선택 가능)")
        print("1: 셀, 2: 행, 3: 열, 4: 테이블, 5: 모두")
        choices = input("선택한 번호를 공백으로 구분하여 입력하세요 (예: 1 3 4): ").split()
        
        elements_to_show = set()
        if '5' in choices:
            elements_to_show = {'cell', 'row', 'column', 'table'}
        else:
            if '1' in choices: elements_to_show.add('cell')
            if '2' in choices: elements_to_show.add('row')
            if '3' in choices: elements_to_show.add('column')
            if '4' in choices: elements_to_show.add('table')

        # 선택된 요소의 수에 따라 서브플롯 구성
        n_plots = len(elements_to_show)
        if n_plots == 0:
            print("선택된 요소가 없습니다.")
            continue

        fig, axes = plt.subplots(1, n_plots, figsize=(6*n_plots, 8))
        if n_plots == 1:
            axes = [axes]  # 단일 축을 리스트로 변환

        for ax, element in zip(axes, elements_to_show):
            ax.imshow(img)
            ax.set_title(f"Image ID: {image_id} - {element.capitalize()}")
            ax.axis('off')

            for ann in annotations:
                bbox = ann['bbox']
                category_name = category_id_to_name[ann['category_id']]
                
                if category_name == element:
                    if category_name == 'cell':
                        has_overflow = ann.get('attributes', {}).get('has_overflow', False)
                        color = 'g' if has_overflow else 'r'
                        rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], 
                                                 linewidth=2, edgecolor=color, facecolor='none')
                        ax.add_patch(rect)
                        label = f"{category_name} (Overflow)" if has_overflow else category_name
                        ax.text(bbox[0], bbox[1], label, color=color, fontsize=8, 
                                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
                    elif category_name == 'row':
                        # 행의 왼쪽과 오른쪽 경계를 테이블의 경계로 설정
                        table_bbox = next(ann['bbox'] for ann in annotations if category_id_to_name[ann['category_id']] == 'table')
                        left_border = table_bbox[0]
                        right_border = table_bbox[0] + table_bbox[2]
                        
                        # 행의 상단 경계만 그립니다
                        ax.plot([left_border, right_border], [bbox[1], bbox[1]], 
                                color='b', linestyle='--', linewidth=2)
                        
                        # 행 레이블 추가
                        ax.text(left_border, bbox[1], f"Row {ann['attributes']['row_id']}", 
                                color='b', fontsize=8, verticalalignment='bottom',
                                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

                    elif category_name == 'column':
                        # 열의 상단과 하단 경계를 테이블의 경계로 설정
                        table_bbox = next(ann['bbox'] for ann in annotations if category_id_to_name[ann['category_id']] == 'table')
                        top_border = table_bbox[1]
                        bottom_border = table_bbox[1] + table_bbox[3]
                        
                        # 열의 왼쪽 경계만 그립니다
                        ax.plot([bbox[0], bbox[0]], [top_border, bottom_border], 
                                color='y', linestyle='--', linewidth=2)
                        
                        # 열 레이블 추가
                        ax.text(bbox[0], top_border, f"Col {ann['attributes']['column_id']}", 
                                color='y', fontsize=8, rotation=90, verticalalignment='top',
                                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

                    elif category_name == 'table':
                        rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], 
                                                 linewidth=3, edgecolor='m', facecolor='none')
                        ax.add_patch(rect)
                        ax.text(bbox[0], bbox[1], category_name, color='m', fontsize=10, 
                                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

        plt.tight_layout()
        output_path = os.path.join(output_dir, f"validated_{image_id}.jpg")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig)

        print(f"검증된 이미지가 저장되었습니다. 이미지 ID: {image_id}")

    print("검증이 완료되었습니다. 출력된 이미지를 확인해주세요.")
def validate_row_column_ids(annotations):
    rows = sorted(set(ann['attributes']['row_id'] for ann in annotations if 'row_id' in ann['attributes']))
    cols = sorted(set(ann['attributes']['column_id'] for ann in annotations if 'column_id' in ann['attributes']))
    
    print(f"Unique row IDs: {rows}")
    print(f"Unique column IDs: {cols}")
    
    # 연속성 검사
    if rows != list(range(min(rows), max(rows)+1)):
        print("Warning: Row IDs are not continuous")
    if cols != list(range(min(cols), max(cols)+1)):
        print("Warning: Column IDs are not continuous")

# 사용 예:

def check_annotation_consistency(annotation_path):
    with open(annotation_path, 'r') as f:
        coco_data = json.load(f)

    print("\n=== JSON 데이터 통계 ===")
    print(f"이미지 수: {len(coco_data['images'])}")
    print(f"어노테이션 수: {len(coco_data['annotations'])}")
    print(f"카테고리 수: {len(coco_data['categories'])}")

    print("\n이미지 ID 샘플:")
    print(json.dumps([img['id'] for img in coco_data['images'][:5]], indent=2))
    print("\n카테고리 ID 샘플:")
    print(json.dumps([cat['id'] for cat in coco_data['categories']], indent=2))

    image_ids = set(img['id'] for img in coco_data['images'])
    annotation_image_ids = set(ann['image_id'] for ann in coco_data['annotations'])
    category_ids = set(cat['id'] for cat in coco_data['categories'])
    annotation_category_ids = set(ann['category_id'] for ann in coco_data['annotations'])

    print("Checking annotation consistency...")
    print(f"Number of images: {len(coco_data['images'])}")
    print(f"Number of annotations: {len(coco_data['annotations'])}")
    print(f"Number of categories: {len(coco_data['categories'])}")
    print(f"All annotation image IDs exist in images: {annotation_image_ids.issubset(image_ids)}")
    print(f"All annotation category IDs exist in categories: {annotation_category_ids.issubset(category_ids)}")

    for ann in coco_data['annotations']:
        img_info = next(img for img in coco_data['images'] if img['id'] == ann['image_id'])
        bbox = ann['bbox']
        if bbox[0] < 0 or bbox[1] < 0 or bbox[0] + bbox[2] > img_info['width'] or bbox[1] + bbox[3] > img_info['height']:
            print(f"Warning: Bounding box out of image bounds for annotation {ann['id']}")

    print("Consistency check complete.")

def check_annotation_consistency(annotation_path):
    with open(annotation_path, 'r') as f:
        coco_data = json.load(f)

    print("\n=== JSON 데이터 통계 ===")
    print(f"이미지 수: {len(coco_data['images'])}")
    print(f"어노테이션 수: {len(coco_data['annotations'])}")
    print(f"카테고리 수: {len(coco_data['categories'])}")

    print("\n이미지 ID 샘플:")
    print(json.dumps([img['id'] for img in coco_data['images'][:5]], indent=2))
    print("\n카테고리 ID 샘플:")
    print(json.dumps([cat['id'] for cat in coco_data['categories']], indent=2))

    image_ids = set(img['id'] for img in coco_data['images'])
    annotation_image_ids = set(ann['image_id'] for ann in coco_data['annotations'])
    category_ids = set(cat['id'] for cat in coco_data['categories'])
    annotation_category_ids = set(ann['category_id'] for ann in coco_data['annotations'])

    print("Checking annotation consistency...")
    print(f"Number of images: {len(coco_data['images'])}")
    print(f"Number of annotations: {len(coco_data['annotations'])}")
    print(f"Number of categories: {len(coco_data['categories'])}")
    print(f"All annotation image IDs exist in images: {annotation_image_ids.issubset(image_ids)}")
    print(f"All annotation category IDs exist in categories: {annotation_category_ids.issubset(category_ids)}")

    category_counts = {cat['name']: 0 for cat in coco_data['categories']}
    for ann in coco_data['annotations']:
        img_info = next(img for img in coco_data['images'] if img['id'] == ann['image_id'])
        bbox = ann['bbox']
        if bbox[0] < 0 or bbox[1] < 0 or bbox[0] + bbox[2] > img_info['width'] or bbox[1] + bbox[3] > img_info['height']:
            print(f"Warning: Bounding box out of image bounds for annotation {ann['id']}")
        
        category_name = next(cat['name'] for cat in coco_data['categories'] if cat['id'] == ann['category_id'])
        category_counts[category_name] += 1

    print("\nCategory counts:")
    for category, count in category_counts.items():
        print(f"{category}: {count}")

    print("Consistency check complete.")
def validate_table_structure(annotations, image_id):
    # 카테고리별로 어노테이션 분류
    cells = [ann for ann in annotations if ann['category_name'] == 'cell']
    rows = [ann for ann in annotations if ann['category_name'] == 'row']
    cols = [ann for ann in annotations if ann['category_name'] == 'column']
    tables = [ann for ann in annotations if ann['category_name'] == 'table']

    print(f"\n=== 이미지 ID {image_id} 검증 결과 ===")
    print(f"총 어노테이션 수: {len(annotations)}")
    print(f"셀 수: {len(cells)}, 행 수: {len(rows)}, 열 수: {len(cols)}, 테이블 수: {len(tables)}")

    if not cells or not rows or not cols or not tables:
        print("경고: 필수 어노테이션(셀, 행, 열, 테이블)이 누락되었습니다.")
        return

    # 행, 열, 셀 ID 추출
    row_ids = sorted(set(ann['attributes']['row_id'] for ann in rows))
    col_ids = sorted(set(ann['attributes']['column_id'] for ann in cols))
    cell_row_ids = sorted(set(ann['attributes']['row'] for ann in cells))
    cell_col_ids = sorted(set(ann['attributes']['col'] for ann in cells))

    print(f"\n행 ID: {row_ids}")
    print(f"열 ID: {col_ids}")
    print(f"셀 행 ID: {cell_row_ids}")
    print(f"셀 열 ID: {cell_col_ids}")

    # 연속성 검사 함수
    def check_continuity(ids, name):
        if ids != list(range(min(ids), max(ids)+1)):
            print(f"경고: {name} ID가 연속적이지 않습니다.")
            missing = set(range(min(ids), max(ids)+1)) - set(ids)
            if missing:
                print(f"  누락된 {name} ID: {sorted(missing)}")
        else:
            print(f"{name} ID가 연속적입니다.")

    # 행, 열, 셀 ID 연속성 검사
    check_continuity(row_ids, "행")
    check_continuity(col_ids, "열")
    check_continuity(cell_row_ids, "셀 행")
    check_continuity(cell_col_ids, "셀 열")

    # 셀과 행/열 ID 일치 검사
    if set(cell_row_ids) != set(row_ids):
        print("경고: 셀의 행 ID와 행 어노테이션의 ID가 일치하지 않습니다.")
        print(f"  셀에만 있는 행 ID: {set(cell_row_ids) - set(row_ids)}")
        print(f"  행 어노테이션에만 있는 ID: {set(row_ids) - set(cell_row_ids)}")
    
    if set(cell_col_ids) != set(col_ids):
        print("경고: 셀의 열 ID와 열 어노테이션의 ID가 일치하지 않습니다.")
        print(f"  셀에만 있는 열 ID: {set(cell_col_ids) - set(col_ids)}")
        print(f"  열 어노테이션에만 있는 ID: {set(col_ids) - set(cell_col_ids)}")

    # 테이블 구조 검사
    if tables:
        table = tables[0]  # 첫 번째 테이블 사용
        table_rows = table['attributes']['total_rows']
        table_cols = table['attributes']['total_cols']
        print(f"\n테이블 구조: {table_rows}행 x {table_cols}열")
        
        if table_rows != len(set(cell_row_ids)) or table_cols != len(set(cell_col_ids)):
            print("경고: 테이블 구조와 실제 셀 수가 일치하지 않습니다.")
            print(f"  예상: {table_rows}행 x {table_cols}열")
            print(f"  실제: {len(set(cell_row_ids))}행 x {len(set(cell_col_ids))}열")
    else:
        print("경고: 테이블 어노테이션이 없습니다.")


def add_images_and_annotations(annotation_path):
    with open(annotation_path, 'r') as f:
        coco_data = json.load(f)

    num_images = len(coco_data['images'])
    num_annotations = len(coco_data['annotations'])

    print("\n=== 이미지 및 어노테이션 통계 ===")
    print(f"총 이미지 수: {num_images}")
    print(f"총 어노테이션 수: {num_annotations}")

    category_counts = {cat['name']: 0 for cat in coco_data['categories']}
    for ann in coco_data['annotations']:
        category_name = next(cat['name'] for cat in coco_data['categories'] if cat['id'] == ann['category_id'])
        category_counts[category_name] += 1

    print("\n카테고리별 어노테이션 수:")
    for category, count in category_counts.items():
        print(f"{category}: {count}")

    print("\n이미지 데이터 샘플:")
    print(json.dumps(coco_data['images'][:2], indent=2, ensure_ascii=False))
    print("\n어노테이션 데이터 샘플:")
    print(json.dumps(coco_data['annotations'][:2], indent=2, ensure_ascii=False))
from collections import Counter

def generate_table_statistics(annotation_path):
    with open(annotation_path, 'r') as f:
        coco_data = json.load(f)
    
    category_id_to_name = {cat['id']: cat['name'] for cat in coco_data['categories']}
    
    cell_counts = {cat['name']: 0 for cat in coco_data['categories']}
    table_sizes = []
    row_counts = []
    col_counts = []
    tables_with_merged_cells = 0
    tables_with_overflow = 0
    overflow_directions = []

    for image in coco_data['images']:
        image_id = image['id']
        annotations = [ann for ann in coco_data['annotations'] if ann['image_id'] == image_id]
        
        table_ann = next((ann for ann in annotations if category_id_to_name[ann['category_id']] == 'table'), None)
        if table_ann:
            table_sizes.append((table_ann['bbox'][2], table_ann['bbox'][3]))
            row_counts.append(table_ann['attributes']['total_rows'])
            col_counts.append(table_ann['attributes']['total_cols'])

        has_merged_cells = False
        has_overflow = False
        
        for ann in annotations:
            category = category_id_to_name[ann['category_id']]
            cell_counts[category] += 1
            
            if category in ['merged_cell', 'merged_overflow_cell']:
                has_merged_cells = True
            
            if category in ['overflow_cell', 'merged_overflow_cell']:
                has_overflow = True
                overflow_directions.append(ann['attributes'].get('direction', 'unknown'))

        if has_merged_cells:
            tables_with_merged_cells += 1
        if has_overflow:
            tables_with_overflow += 1

    total_tables = len(table_sizes)
    total_cells = sum(cell_counts.values())

    print("\n=== 테이블 통계 ===")
    print(f"총 테이블 수: {total_tables}")
    print(f"총 셀 수: {total_cells}")
    
    for category, count in cell_counts.items():
        print(f"{category} 수: {count} ({count/total_cells:.2%})")
    
    print(f"\n평균 테이블 크기: {np.mean(table_sizes, axis=0)}")
    print(f"평균 행 수: {np.mean(row_counts):.2f} (최소: {min(row_counts)}, 최대: {max(row_counts)})")
    print(f"평균 열 수: {np.mean(col_counts):.2f} (최소: {min(col_counts)}, 최대: {max(col_counts)})")

    overflow_cell_ratio = (cell_counts['overflow_cell'] + cell_counts['merged_overflow_cell']) / total_cells

    print(f"오버플로우 셀 비율: {overflow_cell_ratio:.2%}")
    print(f"병합된 셀이 있는 테이블 비율: {tables_with_merged_cells / total_tables:.2%}")
    print(f"오버플로우가 있는 테이블 비율: {tables_with_overflow / total_tables:.2%}")

    # 테이블 크기 분포
    small_tables = sum(1 for r, c in zip(row_counts, col_counts) if r * c <= 25)
    medium_tables = sum(1 for r, c in zip(row_counts, col_counts) if 25 < r * c < 100)
    large_tables = sum(1 for r, c in zip(row_counts, col_counts) if r * c >= 100)
    print(f"\n작은 테이블(5x5 이하) 비율: {small_tables / total_tables:.2%}")
    print(f"중간 테이블(5x5 초과 10x10 미만) 비율: {medium_tables / total_tables:.2%}")
    print(f"큰 테이블(10x10 이상) 비율: {large_tables / total_tables:.2%}")

    # 행/열 비율
    more_rows = sum(1 for r, c in zip(row_counts, col_counts) if r > c)
    more_cols = sum(1 for r, c in zip(row_counts, col_counts) if c > r)
    equal = sum(1 for r, c in zip(row_counts, col_counts) if r == c)
    print(f"\n행이 더 많은 테이블 비율: {more_rows / total_tables:.2%}")
    print(f"열이 더 많은 테이블 비율: {more_cols / total_tables:.2%}")
    print(f"행과 열이 같은 테이블 비율: {equal / total_tables:.2%}")

    # 행과 열 분포
    row_distribution = Counter(row_counts)
    col_distribution = Counter(col_counts)
    print("\n행 분포:")
    for rows, count in sorted(row_distribution.items()):
        print(f"{rows} 행: {count} 테이블 ({count/total_tables:.2%})")
    print("\n열 분포:")
    for cols, count in sorted(col_distribution.items()):
        print(f"{cols} 열: {count} 테이블 ({count/total_tables:.2%})")

    # 오버플로우 방향 분포
    direction_distribution = Counter(overflow_directions)
    print("\n오버플로우 방향 분포:")
    for direction, count in direction_distribution.items():
        print(f"{direction}: {count} 셀 ({count/len(overflow_directions):.2%})")

    # 복잡도 분석
    complexity_scores = [(r * c, (cell_counts['merged_cell'] + cell_counts['merged_overflow_cell']) / (r * c), 
                          (cell_counts['overflow_cell'] + cell_counts['merged_overflow_cell']) / (r * c))
                         for r, c in zip(row_counts, col_counts)]
    avg_size_complexity = np.mean([score[0] for score in complexity_scores])
    avg_merge_complexity = np.mean([score[1] for score in complexity_scores])
    avg_overflow_complexity = np.mean([score[2] for score in complexity_scores])
    
    print(f"\n평균 크기 복잡도: {avg_size_complexity:.2f}")
    print(f"평균 병합 복잡도: {avg_merge_complexity:.2f}")
    print(f"평균 오버플로우 복잡도: {avg_overflow_complexity:.2f}")

    # 추가적인 통계
    print(f"\n행 대 열의 비율: {np.mean(row_counts) / np.mean(col_counts):.2f}")
    print(f"셀 당 평균 면적: {np.mean([w*h/(r*c) for (w,h), r, c in zip(table_sizes, row_counts, col_counts)]):.2f}")

    # 클래스 간 상관관계
    correlation_merged_overflow = np.corrcoef(
        [ann['category_id'] == category_id_to_name.index('merged_cell') for ann in coco_data['annotations']],
        [ann['category_id'] == category_id_to_name.index('overflow_cell') for ann in coco_data['annotations']]
    )[0, 1]
    print(f"\n병합된 셀과 오버플로우 셀 간의 상관계수: {correlation_merged_overflow:.2f}")


if __name__ == "__main__":
    image_path = r"C:\project\table_color_7_classes_simple\train\images"
    annotation_path = r"C:\project\table_color_7_classes_simple\train_annotations.json"
    output_dir = r"valid"

    os.makedirs(output_dir, exist_ok=True)

    # 어노테이션 데이터 로드
    with open(annotation_path, 'r') as f:
        coco_data = json.load(f)

    while True:
        print("\nMenu:")
        print("1. Validate annotations")
        print("2. Check annotation consistency")
        print("3. Add images and annotations count")
        print("4. Validate table structure")
        print("5. Generate table statistics")
        print("6. Exit")
        choice = input("Enter your choice (1-6): ")


        if choice == '1':
            specific_image_id = int(input("Enter the specific image ID to validate (or press Enter to validate random images): ") or 0)
            validate_annotations(image_path, annotation_path, output_dir, specific_image_id if specific_image_id != 0 else None)
        elif choice == '2':
            check_annotation_consistency(annotation_path)
        elif choice == '3':
            add_images_and_annotations(annotation_path)
        if choice == '4':
            print("\n테이블 구조 검증 옵션:")
            print("1. 특정 이미지 검증")
            print("2. 모든 이미지 검증")
            print("3. 돌아가기")
            sub_choice = input("선택하세요 (1-3): ")

            if sub_choice == '1':
                specific_image_id = int(input("검증할 이미지 ID를 입력하세요: "))
                annotations = [ann for ann in coco_data['annotations'] if ann['image_id'] == specific_image_id]
                if annotations:
                    validate_table_structure(annotations, specific_image_id)
                else:
                    print(f"이미지 ID {specific_image_id}에 대한 어노테이션을 찾을 수 없습니다.")
            elif sub_choice == '2':
                print("\n모든 이미지의 테이블 구조 검증:")
                image_ids = set(ann['image_id'] for ann in coco_data['annotations'])
                for image_id in image_ids:
                    annotations = [ann for ann in coco_data['annotations'] if ann['image_id'] == image_id]
                    validate_table_structure(annotations, image_id)
                    print("\n" + "="*50)  # 구분선 추가
            elif sub_choice == '3':
                continue
            else:
                print("잘못된 선택입니다. 다시 시도해주세요.")
        elif choice == '5':
            generate_table_statistics(annotation_path)
        elif choice == '6':
            print("Exiting program.")
            break
        else:
            print("Invalid choice. Please try again.")
