import ujson as json
import gc
import psutil
from PIL import Image
from typing import List, Dict, Any
import os
from io import BytesIO
from typing import List, Dict, Any
from logging_config import table_logger
from dataset_config import (CELL_CATEGORY_ID, TABLE_CATEGORY_ID, 
                            COLUMN_CATEGORY_ID, ROW_CATEGORY_ID, MERGED_CELL_CATEGORY_ID, 
                            OVERFLOW_CELL_CATEGORY_ID, HEADER_COLUMN_CATEGORY_ID, HEADER_ROW_CATEGORY_ID)

def log_memory_usage():
    process = psutil.Process()
    mem_info = process.memory_info()
    table_logger.info(f"Memory usage: {mem_info.rss / 1024 / 1024:.2f} MB")
    
def compress_and_save_image(img: Image.Image, path: str, quality: int = 95) -> None:
    with BytesIO() as output:
        img.save(output, format="JPEG", quality=quality)
        with open(path, "wb") as f:
            f.write(output.getvalue())

def to_python_type(obj: Any) -> Any:
    if isinstance(obj, (int, float, str, bool, type(None))):
        return obj
    elif isinstance(obj, (list, tuple)):
        return [to_python_type(item) for item in obj]
    elif isinstance(obj, dict):
        return {to_python_type(k): to_python_type(v) for k, v in obj.items()}
    else:
        return str(obj)
def save_subset_results(output_dir: str, subset: str, dataset_info: List[Dict[str, Any]], coco_annotations: List[Dict[str, Any]]) -> None:
    annotation_file = os.path.join(output_dir, f'{subset}_annotations.json')
    try:
        with open(annotation_file, 'w') as f:
            f.write('{\n"images": [\n')
            for i, info in enumerate(dataset_info):
                image_data = json.dumps({"id": int(info['image_id']), "file_name": f"{int(info['image_id']):06d}.jpeg", "width": int(info['image_width']), "height": int(info['image_height'])})
                f.write(image_data + (',' if i < len(dataset_info) - 1 else '') + '\n')
            f.write('],\n"annotations": [\n')
            for i, ann in enumerate(coco_annotations):
                annotation_data = json.dumps(to_python_type(ann))
                f.write(annotation_data + (',' if i < len(coco_annotations) - 1 else '') + '\n')
            f.write('],\n"categories": [\n')
            categories = [
                {"id": CELL_CATEGORY_ID, "name": "cell"},
                {"id": TABLE_CATEGORY_ID, "name": "table"},
                {"id": ROW_CATEGORY_ID, "name": "row"},
                {"id": COLUMN_CATEGORY_ID, "name": "column"},
                {"id": MERGED_CELL_CATEGORY_ID, "name": "merged_cell"},
                {"id": OVERFLOW_CELL_CATEGORY_ID, "name": "overflow_cell"},
                {"id": HEADER_ROW_CATEGORY_ID, "name": "header_row"},
                {"id": HEADER_COLUMN_CATEGORY_ID, "name": "header_column"}
            ]
            for i, category in enumerate(categories):
                category_data = json.dumps(category)
                f.write(category_data + (',' if i < len(categories) - 1 else '') + '\n')
            f.write(']\n}')
        
        table_logger.info(f"Saved annotations for {subset} to {annotation_file}")
        log_memory_usage()
        
    except Exception as e:
        table_logger.error(f"Failed to save annotations for {subset}: {str(e)}", exc_info=True)
