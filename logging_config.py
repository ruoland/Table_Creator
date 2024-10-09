# logging_config.py
import logging
from logging.handlers import MemoryHandler
import sys

def setup_logger():
    logger = logging.getLogger('table_generator')
    logger.setLevel(logging.INFO)

    # 스트림 핸들러 설정 (콘솔 출력용)
    stream_handler = logging.StreamHandler(sys.stderr)
    stream_handler.setLevel(logging.WARNING)

    # 포맷터 설정
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    stream_handler.setFormatter(formatter)

    # 핸들러를 로거에 추가
    logger.addHandler(stream_handler)

    return logger

table_logger = setup_logger()
import functools

def log_function(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        table_logger.debug(f"Entering function: {func.__name__}")
        try:
            result = func(*args, **kwargs)
            table_logger.debug(f"Exiting function: {func.__name__}")
            return result
        except Exception as e:
            table_logger.exception(f"Exception in {func.__name__}: {str(e)}")
            raise
    return wrapper
def get_memory_handler():
    # 메모리 핸들러 설정
    memory_handler = MemoryHandler(capacity=1000, flushLevel=logging.ERROR)
    memory_handler.setLevel(logging.DEBUG)
    
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    memory_handler.setFormatter(formatter)
    
    return memory_handler
