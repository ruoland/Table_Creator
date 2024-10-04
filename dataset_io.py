from PIL import Image
from logging_config import  get_memory_handler, table_logger
import os
from dataset_config import TableGenerationConfig, CELL_CATEGORY_ID, TABLE_CATEGORY_ID, COLUMN_CATEGORY_ID, ROW_CATEGORY_ID, MERGED_CELL_CATEGORY_ID, OVERFLOW_CELL_CATEGORY_ID,MERGED_OVERFLOW_CELL_CATEGORY_ID
import ujson as json
from io import BytesIO
import numpy as np
from typing import List, Dict, Any

def numpy_to_python(obj: Any) -> Any:
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj

def compress_and_save_image(img: Image.Image, path: str, quality: int = 95) -> None:
    """
    이미지를 압축하여 JPEG 형식으로 저장합니다.
    :param img: PIL Image 객체
    :param path: 저장할 경로
    :param quality: 압축 품질 (1-100, 높을수록 품질이 좋고 파일 크기가 커짐)
    """
    buffer = BytesIO()
    img.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    with open(path, 'wb') as f:
        f.write(buffer.getvalue())

def convert_floats(item: Any) -> Any:
    if isinstance(item, dict):
        return {k: convert_floats(v) for k, v in item.items()}
    elif isinstance(item, list):
        return [convert_floats(v) for v in item]
    elif isinstance(item, (np.integer, np.floating)):
        return numpy_to_python(item)
    return item
def save_subset_results(output_dir: str, subset: str, dataset_info: List[Dict[str, Any]], coco_annotations: List[Dict[str, Any]]) -> None:
    images = [{"id": int(info['image_id']), 
               "file_name": f"{int(info['image_id']):06d}.jpeg", 
               "width": int(info['image_width']), 
               "height": int(info['image_height'])} 
              for info in dataset_info]
    
    coco_annotations = convert_floats(coco_annotations)
    
    coco_format = {
        "images": images,
        "annotations": coco_annotations,
        "categories": [
            {"id": CELL_CATEGORY_ID, "name": "cell"},
            {"id": TABLE_CATEGORY_ID, "name": "table"},
            {"id": ROW_CATEGORY_ID, "name": "row"},
            {"id": COLUMN_CATEGORY_ID, "name": "column"},
            {"id": MERGED_CELL_CATEGORY_ID, "name": "merged_cell"},
            {"id": OVERFLOW_CELL_CATEGORY_ID, "name": "overflow_cell"},
            {"id": MERGED_OVERFLOW_CELL_CATEGORY_ID, "name": "merged_overflow_cell"}
        ]
    }
    annotation_file = os.path.join(output_dir, f'{subset}_annotations.json')
    
    with open(annotation_file, 'w') as f:
        json.dump(coco_format, f, default=numpy_to_python)
    table_logger.info(f"Saved annotations for {subset} to {annotation_file}")
