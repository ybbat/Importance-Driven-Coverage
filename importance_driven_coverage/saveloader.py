import functools
import os
from typing import Any, Callable

import torch


def file_saveloader(filepath: str, key: Any) -> Callable:
    """
    A decorator that saves or loads function results to/from a .pt file.
    File may save one key:value pair.
    Currently very naive, suggested to delete the file if recalculation is vital.

    Args:
        filepath (str): The path to the file where the function results will be saved.
        key (Any): The key to save the function results under.

    Returns:
        function: The decorated function.
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if os.path.exists(filepath):
                val = torch.load(filepath)
                if val["key"] == str(key):
                    return val["return"]
            ret = func(*args, **kwargs)
            torch.save({"key": str(key), "return": ret}, filepath)
            return ret

        return wrapper

    return decorator
