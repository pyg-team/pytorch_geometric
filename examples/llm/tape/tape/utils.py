import hashlib
import time
from typing import Callable


def profile_execution(func: Callable) -> Callable:
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        execution_time = time.perf_counter() - start_time
        minutes = int(execution_time // 60)
        seconds = execution_time % 60
        print(f"Function '{func.__name__}' executed in "
              f'{minutes} minutes and {seconds:.2f} seconds.\n')
        return result

    return wrapper


def generate_string_hash(input_string: str, algorithm: str = 'sha256'):
    input_bytes = input_string.encode('utf-8')
    hash_obj = hashlib.new(algorithm)
    hash_obj.update(input_bytes)
    return hash_obj.hexdigest()
