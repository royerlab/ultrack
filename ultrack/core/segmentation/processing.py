import logging
import os
import pickle
from contextlib import nullcontext
from typing import Callable, List, Optional

import fasteners
import numpy as np
import pandas as pd
import sqlalchemy as sqla
import zarr
from numpy.typing import ArrayLike
from skimage.measure._regionprops import RegionProperties, regionprops_table
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session
from toolz import curry

from ultrack.config.config import MainConfig, SegmentationConfig
from ultrack.core.database import (
    Base,
    NodeDB,
    OverlapDB,
    clear_all_data,
    get_node_values,
)
from ultrack.core.segmentation.hierarchy import create_hierarchies
from ultrack.core.segmentation.node import Node
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


class _ImageCachedLazyLoader:
    """
    Wrapper class to cache dask/zarr data loading for feature computation.
    """

    def __init__(self, image: ArrayLike):
        self._image = image
        self._current_t = -1
        self._frame = None

    def __getitem__(self, index: int) -> np.ndarray:
        if index != self._current_t:
            self._frame = np.asarray(self._image[index])
            self._current_t = index
        return self._frame


def create_feats_callback(
    shape: ArrayLike, image: Optional[ArrayLike], properties: List[str]
) -> Callable[[Node], np.ndarray]:
    """
    Create a callback function to compute features for each node.

    Parameters
    ----------
    shape : ArrayLike
        Volume (plane) shape.
    image : Optional[ArrayLike]
        Image array for segments properties, could have channel dimension on last axis.
    properties : List[str]
        List of properties to compute for each segment, see skimage.measure.regionprops documentation.

    Returns
    -------
    Callable[[Node], np.ndarray]
        Callback function to compute features for each node returning a numpy array.
    """
    mask = np.zeros(shape, dtype=bool)

    if image is not None:
        image = _ImageCachedLazyLoader(image)

    def _feats_callback(node: Node) -> np.ndarray:

        node.paint_buffer(mask, True, include_time=False)

        if image is None:
            frame = None
        else:
            frame = image[node.time]

        obj = RegionProperties(
            node.slice,
            label=True,
            label_image=mask,
            intensity_image=frame,
            cache_active=True,
        )

        feats = np.concatenate(
            [np.ravel(getattr(obj, p)) for p in properties], dtype=np.float32
        )

        node.paint_buffer(mask, False, include_time=False)

        return feats

    return _feats_callback


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
    image: Optional[ArrayLike] = None,
    properties: Optional[List[str]] = None,
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
    image : Optional[ArrayLike], optional
        Image array for segments properties, channel dimension is optional on last axis, by default None.
    properties : Optional[List[str]], optional
        List of properties to compute for each segment, by default None.
    """
    if config.random_seed == "frame":
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

    node_feats = None
    feats_callback = None

    if properties is not None:
        feats_callback = create_feats_callback(foreground.shape[1:], image, properties)

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

            if feats_callback is not None:
                node_feats = feats_callback(hier_node)

            node = NodeDB(
                id=hier_node.id,
                t_node_id=index,
                t_hier_id=h + 1,
                t=time,
                z=int(z),
                y=int(y),
                x=int(x),
                area=int(hier_node.area),
                frontier=hier_node.frontier,
                height=hier_node.height,
                pickle=pickle.dumps(hier_node),  # pickling to reduce memory usage
                features=node_feats,
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
                raise ValueError(
                    "Duplicated nodes found. Set `overwrite=True` to overwrite existing data."
                ) from e

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


def _get_properties_names(
    shape: ArrayLike,
    image: Optional[ArrayLike],
    properties: Optional[List[str]],
) -> Optional[List[str]]:
    """
    Get properties names from provided properties list.

    Parameters
    ----------
    shape : ArrayLike
        Volume (plane) shape including time.
    image : Optional[ArrayLike]
        Image array for segments properties, could have channel dimension on last axis.
    properties : Optional[List[str]]
        List of properties to compute for each segment, see skimage.measure.regionprops documentation.
    """

    if properties is None:
        return None

    ndim = len(shape) - 1

    if image is None:
        dummy_image = None
    else:
        if image.ndim == len(shape):  # no channel dimension
            dummy_image = np.ones((4,) * ndim, dtype=np.float32)
        else:
            # adding channel dimension
            dummy_image = np.ones((4,) * ndim + (image.shape[-1],), dtype=np.float32)

    dummy_labels = np.zeros((4,) * ndim, dtype=np.uint32)
    dummy_labels[:2, :2] = 1

    data_dict = regionprops_table(dummy_labels, dummy_image, properties=properties)

    return list(data_dict.keys())


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
    image: Optional[ArrayLike] = None,
    properties: Optional[List[str]] = None,
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
    image : Optional[ArrayLike], optional
        Image array of shape (T, (Z), Y, X, (C)) for segments properties, by default None.
        Channel and Z dimensions are optional.
    properties : Optional[List[str]], optional
        List of properties to compute for each segment, see skimage.measure.regionprops documentation.
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

    batch_start_index = int(os.getenv("ULTRACK_BATCH_INDEX_START", "0"))

    if batch_index is None or batch_index == batch_start_index:
        engine = sqla.create_engine(config.data_config.database_path)

        if overwrite:
            clear_all_data(config.data_config.database_path)

        Base.metadata.create_all(engine)
        config.data_config.metadata_add(
            {
                "shape": foreground.shape,
                "properties": _get_properties_names(
                    foreground.shape, image, properties=properties
                ),
            }
        )

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
            image=image,
            properties=properties,
        )

        multiprocessing_apply(
            process,
            time_points,
            config.segmentation_config.n_workers,
            desc="Adding nodes to database",
        )


def get_nodes_features(
    config: MainConfig,
    indices: Optional[ArrayLike] = None,
    include_persistence: bool = False,
) -> pd.DataFrame:
    """
    Creates a pandas dataframe from nodes features defined during segmentation
    plus area and coordinates.

    Parameters
    ----------
    config : MainConfig
        Configuration parameters.
    indices : Optional[ArrayLike], optional
        List of node indices, by default
    include_persistence : bool, optional
        Include persistence features, by default False

    Returns
    -------
    pd.DataFrame
        Dataframe with nodes features
    """
    feats_cols = [NodeDB.t, NodeDB.z, NodeDB.y, NodeDB.x, NodeDB.area, NodeDB.features]
    if include_persistence:
        feats_cols += [NodeDB.hier_parent_id, NodeDB.height]

    df: pd.DataFrame = get_node_values(
        config.data_config, indices=indices, values=feats_cols
    )

    if "features" in df.columns:
        feat_columns = config.data_config.metadata["properties"]
        feat_mat = np.asarray(df["features"].tolist())
        df.loc[:, feat_columns] = feat_mat
        df.drop(columns=["features"], inplace=True)

    df["id"] = df.index

    if include_persistence:

        df.rename(columns={"height": "node_death"}, inplace=True)

        min_height = df["node_death"].min()
        max_height = df["node_death"].max()
        eps = 1e-5

        df.loc[(df["node_death"] < 0) & df["node_death"].isna(), "node_death"] = (
            max_height + eps
        )

        children_df = df[df["hier_parent_id"] > 0]

        df.loc[
            children_df["hier_parent_id"].to_numpy(),
            "node_birth",
        ] = children_df["node_death"].to_numpy()

        df.loc[df["node_birth"].isna(), "node_birth"] = min_height - eps

        df.drop(columns="hier_parent_id", inplace=True)

    return df
