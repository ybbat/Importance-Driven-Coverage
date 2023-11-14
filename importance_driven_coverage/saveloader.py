import functools
import os
from typing import Callable

import torch


def file_saveloader(filepath: str) -> Callable:
    """
    A decorator that saves or loads function results to/from a .pt file.
    Only one combination of function arguments is saved per file for memory reasons.

    Args:
        filepath (str): The path to the file where the function results will be saved.

    Returns:
        function: The decorated function.
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            call = (func.__name__, args, kwargs)
            if os.path.exists(filepath):
                val = torch.load(filepath)
                if val["call"] == call:
                    return val["return"]
            ret = func(*args, **kwargs)
            torch.save({"call": call, "return": ret}, filepath)
            return ret

        return wrapper

    return decorator
