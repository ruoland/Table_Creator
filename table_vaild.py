import json
import cv2
import numpy as np
import os
from typing import Dict, List

class COCOValidator:
    def __init__(self, json_file: str, image_dir: str):
        self.data = self.load_json(json_file)
        self.image_dir = image_dir
        self.category_map = {cat['id']: cat['name'] for cat in self.data['categories']}
        self.colors = {
            'table': (255, 0, 0),
            'row': (0, 255, 0),
            'column': (0, 0, 255),
            'cell': (255, 255, 0),
            'merged_cell': (255, 0, 255),
            'overflow_cell': (0, 255, 255),
            'merged_overflow_cell': (255, 165, 0)
        }
        self.visibility = {category: True for category in self.colors.keys()}
        
        self.category_id_map = {
            0: 'cell',
            1: 'table',
            2: 'row',
            3: 'column',
            4: 'merged_cell',
            5: 'overflow_cell',
            6: 'merged_overflow_cell'
        }

    @staticmethod
    def load_json(file_path: str) -> Dict:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def analyze(self):
        print(f"데이터셋 분석 결과")
        print("=" * 50)
        print(f"총 이미지 수: {len(self.data['images'])}")
        print(f"총 어노테이션 수: {len(self.data['annotations'])}")

        category_counts = {}
        for ann in self.data['annotations']:
            category = self.category_id_map[ann['category_id']]
            category_counts[category] = category_counts.get(category, 0) + 1

        print("\n카테고리별 어노테이션 수:")
        for category, count in category_counts.items():
            print(f"  {category}: {count}")

    def print_annotation_info(self, annotations):
        category_counts = {}
        for ann in annotations:
            category = self.category_id_map[ann['category_id']]
            category_counts[category] = category_counts.get(category, 0) + 1

        print("\n현재 이미지의 어노테이션 정보:")
        print(f"총 어노테이션 수: {len(annotations)}")
        print("카테고리별 어노테이션 수:")
        for category, count in category_counts.items():
            print(f"  {category}: {count}")

        print("\n각 카테고리의 첫 번째 어노테이션 예시:")
        for category in self.category_id_map.values():
            ann = next((a for a in annotations if self.category_id_map[a['category_id']] == category), None)
            if ann:
                print(f"\n{category}:")
                print(json.dumps(ann, indent=2))

    def visualize_annotations(self, image_id: int):
        image_info = next((img for img in self.data['images'] if img['id'] == image_id), None)
        if not image_info:
            print(f"이미지 ID {image_id}를 찾을 수 없습니다.")
            return

        image_path = os.path.join(self.image_dir, image_info['file_name'])
        if not os.path.exists(image_path):
            print(f"이미지 파일을 찾을 수 없습니다: {image_path}")
            return

        original_image = cv2.imread(image_path)
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

        annotations = [ann for ann in self.data['annotations'] if ann['image_id'] == image_id]

        # 어노테이션 정보 출력
        self.print_annotation_info(annotations)

        def draw_annotations(image):
            for ann in annotations:
                category = self.category_id_map[ann['category_id']]
                if self.visibility[category]:
                    color = self.colors.get(category, (0, 0, 0))
                    x, y, w, h = map(int, ann['bbox'])
                    cv2.rectangle(image, (x, y), (x+w, y+h), color, 2)
                    cv2.putText(image, category, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            return image

        def toggle_category(category):
            self.visibility[category] = not self.visibility[category]
            image = original_image.copy()
            image = draw_annotations(image)
            cv2.imshow(f"Image ID: {image_id}", image)

        def on_mouse_click(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                for ann in annotations:
                    bbox = ann['bbox']
                    if bbox[0] <= x <= bbox[0] + bbox[2] and bbox[1] <= y <= bbox[1] + bbox[3]:
                        print(f"\n선택한 어노테이션 정보:")
                        print(json.dumps(ann, indent=2))
                        break

        image = original_image.copy()
        image = draw_annotations(image)
        cv2.imshow(f"Image ID: {image_id}", image)
        cv2.setMouseCallback(f"Image ID: {image_id}", on_mouse_click)

        print("\n카테고리 토글 키:")
        print("t: 테이블, r: 행, c: 열, e: 일반 셀, m: 병합 셀, o: 오버플로우 셀, v: 병합 오버플로우 셀")
        print("a: 모든 카테고리 토글, q: 이미지 닫기")

        while True:
            key = cv2.waitKey(0) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('t'):
                toggle_category('table')
            elif key == ord('r'):
                toggle_category('row')
            elif key == ord('c'):
                toggle_category('column')
            elif key == ord('e'):
                toggle_category('cell')
            elif key == ord('m'):
                toggle_category('merged_cell')
            elif key == ord('o'):
                toggle_category('overflow_cell')
            elif key == ord('v'):
                toggle_category('merged_overflow_cell')
            elif key == ord('a'):
                for category in self.visibility:
                    self.visibility[category] = not self.visibility[category]
                image = original_image.copy()
                image = draw_annotations(image)
                cv2.imshow(f"Image ID: {image_id}", image)

        cv2.destroyAllWindows()

    def validate_dataset(self):
        self.analyze()
        while True:
            image_id = input("확인할 이미지 ID를 입력하세요 (종료하려면 'q' 입력): ")
            if image_id.lower() == 'q':
                break
            try:
                self.visualize_annotations(int(image_id))
            except ValueError:
                print("올바른 이미지 ID를 입력해주세요.")

if __name__ == "__main__":
    json_file = r"C:\project\table_color\train_annotations.json"  # JSON 파일 경로
    image_dir = r"C:\project\table_color\train\images"  # 이미지 디렉토리 경로
    validator = COCOValidator(json_file, image_dir)
    validator.validate_dataset()
