# Modified from napari.utils.migrations.py
import warnings
from functools import wraps
from typing import Callable


def rename_argument(from_name: str, to_name: str) -> Callable:
    """
    This is decorator for simple rename function argument
    without break backward compatibility.

    Parameters
    ----------
    from_name : str
        old name of argument
    to_name : str
        new name of argument
    """

    def _wrapper(func):
        @wraps(func)
        def _update_from_dict(*args, **kwargs):
            if from_name in kwargs:
                if to_name in kwargs:
                    raise ValueError(
                        f"Argument {to_name} already defined, please do not mix {from_name} and {to_name} in one call."
                    )

                warnings.warn(
                    f"Argument {from_name} is deprecated, please use {to_name} instead.",
                    category=DeprecationWarning,
                    stacklevel=2,
                )
                kwargs = kwargs.copy()
                kwargs[to_name] = kwargs.pop(from_name)
            return func(*args, **kwargs)

        return _update_from_dict

    return _wrapper
