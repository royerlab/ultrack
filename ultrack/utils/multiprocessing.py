import multiprocessing as mp
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

import fasteners
from tqdm import tqdm

from ultrack.config.config import DataConfig


def multiprocessing_apply(
    func: Callable[[Any], None],
    sequence: Sequence[Any],
    n_workers: int,
    desc: Optional[str] = None,
) -> None:
    """Applies `func` for each item in `sequence`.

    Parameters
    ----------
    func : Callable[[Any], NoneType]
        Function to be executed.
    sequence : Sequence[Any]
        Sequence of parameters.
    n_workers : int
        Number of workers for multiprocessing.
    desc : Optional[str], optional
        Description to tqdm progress bar, by default None
    """
    length = len(sequence)
    if n_workers > 1:
        with mp.Pool(min(n_workers, length)) as pool:
            list(tqdm(pool.imap(func, sequence), desc=desc, total=length))
    else:
        for t in tqdm(sequence, desc=desc):
            func(t)


@contextmanager
def multiprocessing_sqlite_lock(
    data_config: DataConfig,
) -> Optional[fasteners.InterProcessLock]:
    """Write lock for writing on `sqlite` with multiprocessing. No lock otherwise."""

    lock = None
    if data_config.database == "sqlite":
        identifier = uuid.uuid4().hex
        lock = fasteners.InterProcessLock(
            path=data_config.working_dir / f"{identifier}.lock"
        )

    try:
        yield lock

    finally:
        if lock is not None:
            Path(lock.path.decode("ascii")).unlink(missing_ok=True)
