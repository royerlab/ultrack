import logging
import pickle
from contextlib import nullcontext
from typing import List, Optional

import fasteners
import numpy as np
import sqlalchemy as sqla
import zarr
from numpy.typing import ArrayLike
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session
from toolz import curry

from ultrack.config.config import MainConfig, SegmentationConfig
from ultrack.core.database import Base, NodeDB, OverlapDB, clear_all_data
from ultrack.core.segmentation.hierarchy import create_hierarchies
from ultrack.utils.array import check_array_chunk
from ultrack.utils.deprecation import rename_argument
from ultrack.utils.multiprocessing import (
    batch_index_range,
    multiprocessing_apply,
    multiprocessing_sqlite_lock,
)

logging.basicConfig()
logging.getLogger("sqlachemy.engine").setLevel(logging.INFO)

LOG = logging.getLogger(__name__)


def _generate_id(index: int, time: int, max_segments: int) -> int:
    """Generates a unique id.

    Parameters
    ----------
    index : int
        Current index per time.
    time_point : int
        Current time.
    max_segments : int
        Upper bound of number of segments.

    Returns
    -------
    int
        Unique id.
    """
    return index + (time + 1) * max_segments


def _insert_db(
    engine: Engine, time: int, nodes: List[NodeDB], overlaps: List[OverlapDB]
) -> None:
    """
    Helper function to insert data into database.
    IMPORTANT: This function resets `nodes` and `overlaps`.

    Parameters
    ----------
    engine : Engine
        SQLAlchemy connection engine.
    time : int
        Current time.
    nodes : List[NodeDB]
        List of nodes to insert.
    overlaps : List[OverlapDB]
        List of overlap of nodes to insert.
    """
    LOG.info(f"Pushing some nodes from hier time {time} to database.")
    with Session(engine) as session:
        session.add_all(nodes)
        session.add_all(overlaps)
        session.commit()
    engine.dispose()
    nodes.clear()
    overlaps.clear()


@curry
def _process(
    time: int,
    foreground: ArrayLike,
    contours: ArrayLike,
    config: SegmentationConfig,
    db_path: str,
    max_segments_per_time: int,
    write_lock: Optional[fasteners.InterProcessLock] = None,
    catch_duplicates_expection: bool = False,
    insertion_throttle_rate: int = 50,
) -> None:
    """Process `foreground` and `edge` of current time and add data to database.

    Parameters
    ----------
    time : int
        Current time.
    foreground : ArrayLike
        Foreground array.
    contours : ArrayLike
        Contours array.
    config : SegmentationConfig
        Segmentation configuration parameters.
    db_path : str
        Path to database including type prefix.
    max_segments_per_time : int
        Upper bound of segments per time point.
    lock : Optional[fasteners.InterProcessLock], optional
        Lock object for SQLite multiprocessing, optional otherwise, by default None.
    catch_duplicates_expection : bool
        If True, catches duplicates exception, by default False.
    insertion_throttle_rate : int
        Throttling rate for insertion, by default 50.
    """
    np.random.seed(time)

    edge_map = contours[time]
    if config.max_noise > 0:
        noise = np.random.uniform(0, config.max_noise, size=edge_map.shape)
        # promoting edge_map to smallest float
        noise = noise.astype(np.result_type(edge_map.dtype, np.float16))
        edge_map = edge_map + noise

    hiers = create_hierarchies(
        foreground[time] > config.threshold,
        edge_map,
        hierarchy_fun=config.ws_hierarchy,
        max_area=config.max_area,
        min_area=config.min_area,
        min_frontier=config.min_frontier,
    )

    LOG.info(f"Computing nodes of time {time}")

    engine = sqla.create_engine(db_path, hide_parameters=True)

    index = 1
    nodes = []
    overlaps = []

    for h, hierarchy in enumerate(hiers):
        hierarchy.cache = True

        hier_index_map = {}
        for hier_node in hierarchy.nodes:
            hier_node.id = _generate_id(index, time, max_segments_per_time)
            hier_node.time = time
            hier_node._parent = None  # avoiding pickling parent hierarchy
            centroid = hier_node.centroid

            if len(centroid) == 2:
                y, x = centroid
                z = 0
            else:
                z, y, x = centroid

            node = NodeDB(
                id=hier_node.id,
                t_node_id=index,
                t_hier_id=h + 1,
                t=time,
                z=int(z),
                y=int(y),
                x=int(x),
                area=int(hier_node.area),
                pickle=pickle.dumps(hier_node),  # pickling to reduce memory usage
            )

            hier_index_map[hier_node._h_node_index] = node
            nodes.append(node)

            index += 1
            del hier_node

        tree = hierarchy.tree

        for h_index, node in hier_index_map.items():

            h_parent_index = tree.parent(h_index)
            # checking if h_index is root or parent is not in hier_index_map
            if h_index != h_parent_index:
                try:
                    # assign hierarchy parent to node
                    node.hier_parent_id = hier_index_map[h_parent_index].id
                except KeyError:
                    pass

            # find overlapping segments by iterating through hierarchy (tree)
            for a in tree.ancestors(h_index):
                if a in hier_index_map and a != h_index:
                    overlaps.append(
                        OverlapDB(
                            node_id=hier_index_map[h_index].id,
                            ancestor_id=hier_index_map[a].id,
                        )
                    )

            LOG.info(
                f"{len(overlaps)} overlaps found for node {hier_index_map[h_index].id}."
            )

        hierarchy.cache = False
        hierarchy.free_props()

        if index > max_segments_per_time:
            raise ValueError(
                f"Number of segments exceeds upper bound of {max_segments_per_time} per time."
            )

        try:
            # if lock is None it inserts at every iteration
            if write_lock is None:
                if (h + 1) % insertion_throttle_rate == 0:  # throttling insertions
                    _insert_db(engine, time, nodes, overlaps)

            # otherwise it inserts only when it can acquire the lock
            elif write_lock.acquire(blocking=False):
                _insert_db(engine, time, nodes, overlaps)
                write_lock.release()

        except sqla.exc.IntegrityError as e:
            if (
                catch_duplicates_expection
                and "duplicate key value violates unique constraint" in str(e).lower()
            ):
                LOG.warning(
                    f"IntegrityError: {e}\nSkipping batch {time}.\n"
                    "Cannot guarantee image was entirely processed.\nSEGMENTS MIGHT BE MISSING!"
                )
                return
            else:
                raise e

    # pushes any remaning data
    with write_lock if write_lock is not None else nullcontext():
        _insert_db(engine, time, nodes, overlaps)

    LOG.info(f"DONE with time {time}.")


def _check_zarr_memory_store(arr: ArrayLike) -> None:
    if isinstance(arr, zarr.Array) and isinstance(arr.store, zarr.MemoryStore):
        LOG.warning(
            "Found zarr with MemoryStore. "
            "Using an zarr with MemoryStore can lead to considerable memory usage."
        )


@rename_argument("detection", "foreground")
@rename_argument("edge", "contours")
def segment(
    foreground: ArrayLike,
    contours: ArrayLike,
    config: MainConfig,
    max_segments_per_time: int = 1_000_000,
    batch_index: Optional[int] = None,
    overwrite: bool = False,
    insertion_throttle_rate: int = 50,
) -> None:
    """Add candidate segmentation (nodes) from `foreground` and `edge` to database.

    Parameters
    ----------
    foreground : ArrayLike
        Foreground probability array of shape (T, (Z), Y, X)
    contours : ArrayLike
        Contours array of shape (T, (Z), Y, X)
    config : MainConfig
        Configuration parameters.
    max_segments_per_time : int
        Upper bound of segments per time point.
    batch_index : Optional[int], optional
        Batch index for processing a subset of nodes, by default everything is processed.
    overwrite : bool
        Cleans up segmentation, linking, and tracking database content before processing.
    insertion_throttle_rate : int
        Throttling rate for insertion, by default 50.
        Only used with non-sqlite databases.
    """
    LOG.info(f"Adding nodes with SegmentationConfig:\n{config.segmentation_config}")

    if foreground.shape != contours.shape:
        raise ValueError(
            f"`foreground` and `contours` shape must match. Found {foreground.shape} and {contours.shape}"
        )

    check_array_chunk(foreground)
    check_array_chunk(contours)

    _check_zarr_memory_store(foreground)
    _check_zarr_memory_store(contours)

    LOG.info(f"Foreground array with shape {foreground.shape}")
    LOG.info(f"Edge array with shape {contours.shape}")

    length = foreground.shape[0]
    time_points = batch_index_range(
        length, config.segmentation_config.n_workers, batch_index
    )
    LOG.info(f"Segmenting time points {time_points}")

    if batch_index is None or batch_index == 0:
        engine = sqla.create_engine(config.data_config.database_path)

        if overwrite:
            clear_all_data(config.data_config.database_path)

        Base.metadata.create_all(engine)
        config.data_config.metadata_add({"shape": foreground.shape})

    with multiprocessing_sqlite_lock(config.data_config) as lock:
        process = _process(
            foreground=foreground,
            contours=contours,
            config=config.segmentation_config,
            db_path=config.data_config.database_path,
            write_lock=lock,
            max_segments_per_time=max_segments_per_time,
            catch_duplicates_expection=batch_index is not None,
            insertion_throttle_rate=insertion_throttle_rate,
        )

        multiprocessing_apply(
            process,
            time_points,
            config.segmentation_config.n_workers,
            desc="Adding nodes to database",
        )
