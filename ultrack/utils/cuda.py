import importlib
import logging
from contextlib import contextmanager
from types import ModuleType
from typing import Generator

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


def import_module(module: str, submodule: str) -> ModuleType:
    """Import GPU accelerated module if available, otherwise returns CPU version.

    Parameters
    ----------
    module : str
        Main python module (e.g. skimage, scipy)
    submodule : str
        Secondary python module (e.g. morphology, ndimage)

    Returns
    -------
    ModuleType
        Imported submodule.
    """
    cupy_module_name = f"{CUPY_MODULES[module]}.{submodule}"
    try:
        pkg = importlib.import_module(cupy_module_name)
        LOG.info(f"{cupy_module_name} found.")

    except (ModuleNotFoundError, ImportError):

        pkg = importlib.import_module(f"{module}.{submodule}")
        LOG.info(f"{cupy_module_name} not found. Using cpu equivalent")

    return pkg


def to_cpu(arr: ArrayLike) -> ArrayLike:
    """Moves array to cpu, if it's already there nothing is done."""
    if hasattr(arr, "cpu"):
        arr = arr.cpu()
    elif hasattr(arr, "get"):
        arr = arr.get()
    return arr
