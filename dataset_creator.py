# dataset_creator.py
from logging_config import table_logger
from dataset_config import TableGenerationConfig
from multiprocessing import Pool, cpu_count, freeze_support
import time
from dataset_image_maker import batch_dataset

import asyncio
def setup_config():
    config = TableGenerationConfig()
    
    return config

def run_dataset_generation(output_dir, num_images, train_ratio, num_processes, config):
    try:
        start_time = time.time()
        
        result = batch_dataset(
            output_dir=output_dir, 
            total_num_images=num_images,
            train_ratio=train_ratio,
            num_processes=num_processes,
            config=config
        )
        end_time = time.time()
        print(f"데이터셋 생성 시간: {end_time - start_time:.2f} 초")
        return result
    except Exception as e:
        table_logger.error(f"데이터셋 생성 중 오류 발생: {str(e)}", exc_info=True)
        raise

def main():
    start_time = time.time()

    output_dir = r'C:\project\table_ver6-for_overflow'
    train_ratio = 0.8
    num_processes = cpu_count() - 2

    config = setup_config()
    num_images = config.total_images
    try:
        run_dataset_generation(
            output_dir, num_images, train_ratio, num_processes, config
        )
        end_time = time.time()
        table_logger.info(f"전체 프로그램 실행 시간: {end_time - start_time:.2f} 초")
        table_logger.info("프로그램 실행 완료")
    except Exception as e:
        table_logger.error(f"메인 프로그램 실행 중 오류 발생: {str(e)}", exc_info=True)

if __name__ == "__main__":

    freeze_support()  # Windows에서 필요
    main()
