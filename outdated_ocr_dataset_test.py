import cv2
import os
import random
import numpy as np

def draw_box(image, center, width, height, color=(0, 255, 0), thickness=2):
    x1 = int(center[0] - width / 2)
    y1 = int(center[1] - height / 2)
    x2 = int(center[0] + width / 2)
    y2 = int(center[1] + height / 2)
    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)

def visualize_labeled_image(dataset_dir, num_samples=5):
    image_dir = os.path.join(dataset_dir, 'images')
    label_dir = os.path.join(dataset_dir, 'labels')
    
    image_files = os.listdir(image_dir)
    selected_images = random.sample(image_files, min(num_samples, len(image_files)))

    for selected_image in selected_images:
        image_path = os.path.join(image_dir, selected_image)
        label_path = os.path.join(label_dir, selected_image.replace('.png', '.txt'))

        print(f"\nProcessing image: {selected_image}")
        print(f"Image path: {image_path}")
        print(f"Label path: {label_path}")

        image = cv2.imread(image_path)
        height, width = image.shape[:2]
        print(f"Image dimensions: {width}x{height}")

        with open(label_path, 'r') as f:
            labels = f.readlines()

        print(f"Total labels: {len(labels)}")

        for idx, label in enumerate(labels):
            parts = label.strip().split()
            print(f"\nProcessing label {idx}: {parts}")
            
            if len(parts) != 10:
                print(f"Warning: Invalid label format for label {idx}")
                continue

            class_id, x, y, w, h, angle, row_start, row_end, col_start, col_end = map(float, parts)
            
            print(f"Parsed values: class_id={class_id}, x={x}, y={y}, w={w}, h={h}")

            center_x = int(x * width)
            center_y = int(y * height)
            box_width = max(int(w * width), 5)  # 최소 너비 5픽셀
            box_height = max(int(h * height), 5)  # 최소 높이 5픽셀

            print(f"Calculated values: center_x={center_x}, center_y={center_y}, box_width={box_width}, box_height={box_height}")

            color = (0, 255, 0) if class_id == 0 else (0, 0, 255)

            draw_box(image, (center_x, center_y), box_width, box_height, color)
            
            # 셀 정보 표시
            cv2.putText(image, f"{idx}: ({x:.2f}, {y:.2f})", (center_x, center_y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1)

            print(f"Drew box for label {idx}")

        cv2.imshow(f"Labeled Image: {selected_image}", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        output_dir = os.path.join(dataset_dir, 'visualized')
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"visualized_{selected_image}")
        cv2.imwrite(output_path, image)
        print(f"Visualized image saved to: {output_path}")

# 사용 예:
visualize_labeled_image('./table_dataset/uniform-no_rotation', num_samples=5)
