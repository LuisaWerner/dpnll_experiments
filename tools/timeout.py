from concurrent.futures import ThreadPoolExecutor
from subprocess import TimeoutExpired
from typing import Callable
import multiprocessing

from tools import process


class Timeout(Exception):
    def __init__(self, timeout: float):
        super().__init__(f"Timeout({timeout:.2f})")
        self.timeout = timeout


def process_timeout_call(timeout: float, func: Callable, *args, **kwargs):

    queue = multiprocessing.Queue()
    process = multiprocessing.Process(target=func, args=args, kwargs=kwargs)
    process.start()
    process.join(timeout)

    if process.is_alive():
        process.terminate()
        process.join()
        return Timeout(timeout)
    else:
        return func()


executor = ThreadPoolExecutor()


def timeout_call(timeout: float, func: Callable, *args, **kwargs):
    future = executor.submit(func, *args, **kwargs)
    try:
        return future.result(timeout=timeout)
    except TimeoutError:
        return Timeout(timeout)


python_interpreter_path = "C:/Users/Thomas Valentin/AppData/Local/Programs/Python/Python312/python.exe"


def subprocess_timeout_call(timeout: float, python_file: str, *args):
    proc = process.start_subprocess([python_interpreter_path, python_file] + [arg for arg in args])
    try:

        stdout, stderr = proc.communicate(timeout=timeout)
        return stdout

    except TimeoutExpired:
        process.terminate(proc)
        return Timeout(timeout)
