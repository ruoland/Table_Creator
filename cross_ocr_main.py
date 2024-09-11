import subprocess
import os
import json
import logging as logger
def load_json(final_output_path):
    with open(final_output_path, 'r', encoding='UTF-8') as f:
        data = json.load(f)
        logger.log(logger.INFO, "데이터를 불러옵니다, ", final_output_path)
        return data
        
def run_ocr(image_path):
    final_output_json = "final_output.json"
    final_boxes_image = "cross_image_final_ocr_boxes.png"
    crop_output = "results/crop/"

    # 표 자르기
    subprocess.run([
        "python", "crop_table.py", image_path, crop_output
    ], check=True)
    logger.log(logger.INFO, f"{image_path} 에서 표를 찾아냅니다, ", image_path)

    crop_output = ("results/crop/"+os.path.basename(image_path))
    if not os.path.isfile(crop_output):
        logger.log(logger.INFO, f"{image_path} 에서 표를 찾지 못했습니다, ", image_path)
    
    logger.log(logger.INFO, f"{crop_output} 에서 내용을 검색합니다 ", crop_output)
    # 표 인식하기 (OCR)

    subprocess.run([
        "conda", "run", "-n", "time-ocr", "python",
        "chn_ocr.py", crop_output
    ], check=True)

    # 최종 결과 로드
    img_output = ("results/img/"+os.path.basename(image_path))

    final_results = load_json(final_output_json)
   
    print(f"최종 병합 결과 박스가 {final_boxes_image}에 그려졌습니다")
    print(f"최종 결과가 {final_results}에 저장되었습니다")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("사용법: python main.py <image_path> [x_threshold] [y_threshold] [vertical_threshold]")
        sys.exit(1)
    
    image_path = sys.argv[1]
    x_threshold = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    y_threshold = int(sys.argv[3]) if len(sys.argv) > 3 else 20
    vertical_threshold = int(sys.argv[4]) if len(sys.argv) > 4 else 10
    
    run_ocr(image_path)