# dataset_creator.py
from logging_config import table_logger
from dataset_config import TableGenerationConfig, DatasetCellTypeCounter
from multiprocessing import Pool, cpu_count, freeze_support

from dataset_generator import batch_dataset

import asyncio
def setup_config():
    config = TableGenerationConfig()
    config.dataset_counter = DatasetCellTypeCounter(config)
    
    return config

def run_dataset_generation(output_dir, num_images, imperfect_ratio, train_ratio, num_processes, config):
    try:
        return batch_dataset(
            output_dir=output_dir, 
            total_num_images=num_images,
            imperfect_ratio=imperfect_ratio, 
            train_ratio=train_ratio,
            num_processes=num_processes,
            config=config
        )
    except Exception as e:
        table_logger.error(f"데이터셋 생성 중 오류 발생: {str(e)}", exc_info=True)
        raise

def main():
    output_dir = r'C:\project\table_color2'
    
    imperfect_ratio = 0.3
    train_ratio = 0.8
    num_processes = cpu_count() - 1

    config = setup_config()
    num_images = config.total_images
    try:
        batch_dataset(
            output_dir, num_images, imperfect_ratio, train_ratio, num_processes, config
        )
        table_logger.info("프로그램 실행 완료")
    except Exception as e:
        table_logger.error(f"메인 프로그램 실행 중 오류 발생: {str(e)}", exc_info=True)
import cProfile
if __name__ == "__main__":

    freeze_support()  # Windows에서 필요
    main()
