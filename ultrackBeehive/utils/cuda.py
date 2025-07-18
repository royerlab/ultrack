import functools
import importlib
import logging
from contextlib import contextmanager
from types import ModuleType
from typing import Callable, Generator, Optional

import dask.array as da
import numpy as np
from numpy.typing import ArrayLike

LOG = logging.getLogger(__name__)

try:
    import cupy as cp
    from cupy.cuda.memory import malloc_managed

    LOG.info("cupy found.")
except (ModuleNotFoundError, ImportError):
    LOG.info("cupy not found.")
    cp = None

try:
    import torch as th

    LOG.info("torch found.")
except (ModuleNotFoundError, ImportError):
    LOG.info("torch not found.")
    th = None


CUPY_MODULES = {
    "scipy": "cupyx.scipy",
    "skimage": "cucim.skimage",
}


def is_cupy_array(arr: ArrayLike) -> bool:
    """
    Checks if array is a cupy array.

    Parameters
    ----------
    arr : ArrayLike
        Array to be checked.
    """
    return cp is not None and isinstance(arr, cp.ndarray)


@contextmanager
def unified_memory() -> Generator:
    """
    Initializes cupy's unified memory allocation.
    cupy's functions will run slower but it will spill memory memory into cpu without crashing.
    """
    # starts unified memory
    if cp is not None:
        previous_allocator = cp.cuda.get_allocator()
        cp.cuda.set_allocator(malloc_managed)

    yield

    # ends unified memory
    if cp is not None:
        cp.clear_memo()
        cp.cuda.set_allocator(previous_allocator)


def torch_default_device() -> "th.device":
    """
    Returns "gpu", "mps" or "cpu" devices depending on their availability.

    Returns
    -------
    th.device
        Torch fastest device.
    """
    if th is None:
        raise ImportError("torch not found.")

    if th.cuda.is_available():
        device = th.cuda.device_count() - 1
    elif th.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    return th.device(device)


def import_module(
    module: str,
    submodule: str,
    arr: Optional[ArrayLike] = None,
) -> ModuleType:
    """Import GPU accelerated module if available and matches optional array, otherwise returns CPU version.

    Parameters
    ----------
    module : str
        Main python module (e.g. skimage, scipy)
    submodule : str
        Secondary python module (e.g. morphology, ndimage)
    arr : ArrayLike
        If provided, it will be used to determine if GPU module should be imported.

    Returns
    -------
    ModuleType
        Imported submodule.
    """
    is_gpu_array = cp is not None and isinstance(arr, cp.ndarray)

    if arr is None or is_gpu_array:
        cupy_module_name = f"{CUPY_MODULES[module]}.{submodule}"
        try:
            pkg = importlib.import_module(cupy_module_name)
            LOG.info(f"{cupy_module_name} found.")
            return pkg

        except (ModuleNotFoundError, ImportError):
            LOG.info(f"{cupy_module_name} not found. Using cpu equivalent")

    pkg = importlib.import_module(f"{module}.{submodule}")

    return pkg


def to_cpu(arr: ArrayLike) -> ArrayLike:
    """Moves array to cpu, if it's already there nothing is done."""
    if hasattr(arr, "cpu"):
        arr = arr.cpu()
    elif hasattr(arr, "get"):
        arr = arr.get()
    return arr


def on_gpu(func: Callable) -> Callable:
    """Decorator to run a function on GPU if available, otherwise it will run on CPU.

    Parameters
    ----------
    func : Callable
        Function to be decorated.

    Returns
    -------
    Callable
        Decorated function.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if cp is not None:
            args = [
                cp.asarray(a) if isinstance(a, (np.ndarray, da.Array)) else a
                for a in args
            ]
            kwargs = {
                k: cp.asarray(v) if isinstance(v, (np.ndarray, da.Array)) else v
                for k, v in kwargs.items()
            }
        return func(*args, **kwargs)

    if not hasattr(func, "__name__"):
        # if it instead a class
        wrapper.__name__ = func.__class__.__name__

    return wrapper
