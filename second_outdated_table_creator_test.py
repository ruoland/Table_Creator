import cv2
import os
import random
import numpy as np

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
            
            if len(parts) != 5:  # class, x_center, y_center, width, height
                print(f"Warning: Invalid label format for label {idx}")
                continue

            class_id, x_center, y_center, w, h = map(float, parts)
            
            print(f"Parsed values: class_id={class_id}, x_center={x_center}, y_center={y_center}, w={w}, h={h}")

            # Convert normalized coordinates to pixel coordinates
            x_center = int(x_center * width)
            y_center = int(y_center * height)
            box_width = int(w * width)
            box_height = int(h * height)

            # Calculate top-left and bottom-right corners
            x1 = int(x_center - box_width / 2)
            y1 = int(y_center - box_height / 2)
            x2 = int(x_center + box_width / 2)
            y2 = int(y_center + box_height / 2)

            print(f"Calculated values: x1={x1}, y1={y1}, x2={x2}, y2={y2}")

            # Draw bounding box
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw label
            label_text = f"{idx}: ({x_center/width:.2f}, {y_center/height:.2f})"
            cv2.putText(image, label_text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            print(f"Drew box for label {idx}")

        cv2.imshow(f"Labeled Image: {selected_image}", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        output_dir = os.path.join(dataset_dir, 'visualized')
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"visualized_{selected_image}")
        cv2.imwrite(output_path, image)
        print(f"Visualized image saved to: {output_path}")

if __name__ == '__main__':
    visualize_labeled_image('table_dataset', num_samples=5)
