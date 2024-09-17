#계산


import cv2
import numpy as np
import os
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

def process_image(image_path):
    try:
        image = cv2.imread(image_path)
        if image is None:
            print(f"Failed to read image: {image_path}")
            return None
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if image.size == 0:
            print(f"Empty image: {image_path}")
            return None
        
        mean = np.mean(image, axis=(0, 1))
        std = np.std(image, axis=(0, 1))
        
        return mean, std
    except Exception as e:
        print(f"Error processing {image_path}: {str(e)}")
        return None

def calculate_mean_std(image_folder):
    image_files = [os.path.join(image_folder, f) for f in os.listdir(image_folder) 
                   if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'))]

    if not image_files:
        print(f"No image files found in {image_folder}")
        return None, None

    # 사용할 프로세스 수 결정 (CPU 코어 수의 절반 사용)
    num_processes = max(1, cpu_count() // 2)

    with Pool(processes=num_processes) as pool:
        results = list(tqdm(pool.imap(process_image, image_files), total=len(image_files), desc="Processing images"))

    # None 값 제거 및 유효한 결과만 추출
    valid_results = [r for r in results if r is not None]

    if not valid_results:
        print("No valid images processed")
        return None, None

    means, stds = zip(*valid_results)
    overall_mean = np.mean(means, axis=0)
    overall_std = np.mean(stds, axis=0)

    return overall_mean, overall_std

if __name__ == '__main__':
    # 이미지 폴더 경로 지정
    image_folder = r"D:\Projects\OCR-LEARNIGN-PROJECT\OCR-PROJECT_OLD\yolox_table_dataset_simple\train\images"

    mean, std = calculate_mean_std(image_folder)

    if mean is not None and std is not None:
        print(f"Mean: {mean}")
        print(f"Std: {std}")
    else:
        print("Failed to calculate mean and std")

