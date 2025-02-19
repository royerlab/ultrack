import logging
from typing import List, Tuple, Union

import numpy as np
import sqlalchemy as sqla
from sqlalchemy import Column, func
from sqlalchemy.orm import Session

from ultrack.config import MainConfig
from ultrack.core.database import NodeDB

LOG = logging.getLogger(__name__)
LOG.setLevel(logging.INFO)


class UltrackArray:
    def __init__(
        self,
        config: MainConfig,
        node_attribute: Column = NodeDB.id,
    ):
        """
        Initialize an array for visualizing segments in the ultrack database.

        The array provides direct visualization of segments stored in the database,
        allowing efficient access and transversing the hierarchy of segments.

        Parameters
        ----------
        config : MainConfig
            Configuration object containing Ultrack settings and metadata.
        node_attribute : sqlalchemy.Column, optional
            Node attribute to use for painting the array, by default NodeDB.id.

        Notes
        -----
        The array shape is determined from the configuration metadata.
        """

        self._buffer: Union[np.ndarray, None] = None
        self.config = config
        self.node_attribute = node_attribute

        self.database_path = config.data_config.database_path
        self.minmax = self.find_min_max_volume_entire_dataset()
        self.num_pix_threshold = self.minmax.mean().astype(int)

    @property
    def shape(self) -> Tuple[int, ...]:
        """
        Return the shape of the array.
        If metadata does not contain shape, returns a dummy dimension (1, 10, 10).

        Returns
        -------
        Tuple[int, ...]
            Shape of the array.
        """
        return tuple(self.config.data_config.metadata.get("shape", (1, 10, 10)))

    @property
    def t_max(self) -> int:
        return self.shape[0]

    @property
    def ndim(self) -> int:
        return len(self.shape)

    @property
    def buffer(self) -> np.ndarray:
        new_buffer_shape = self.shape[1:]
        if (
            self._buffer is None
            or new_buffer_shape != self._buffer.shape
            or self.dtype != self._buffer.dtype
        ):
            self._buffer = np.zeros(new_buffer_shape, dtype=self.dtype)
        return self._buffer

    @property
    def dtype(self) -> np.dtype:
        # NOTE: I don't know how this will behave with BigInteger and Enum columns
        return self.node_attribute.type.python_type

    def __getitem__(
        self,
        indexing: Union[Tuple[Union[int, slice]], int, slice],
    ) -> np.ndarray:
        """Access segments data using array-like indexing.

        Parameters
        ----------
        indexing : tuple or int or slice
            Index specification. Can be:
            - A single integer or slice for time dimension
            - A tuple containing time index/slice followed by spatial dimensions

        Returns
        -------
        numpy.ndarray
            Array containing the requested segment data.
        """
        # print('indexing in getitem:',indexing)

        if isinstance(indexing, tuple):
            time, volume_slicing = indexing[0], indexing[1:]
        else:  # if only 1 (time) is provided
            time = indexing
            volume_slicing = tuple()

        if isinstance(time, slice):  # if all time points are requested
            return np.stack(
                [
                    self.__getitem__((t,) + volume_slicing)
                    for t in range(*time.indices(self.shape[0]))
                ]
            )
        else:
            try:
                time = time.item()  # convert from numpy.int to int
            except AttributeError:
                time = time

        # load buffer to avoid recomputing it
        buffer = self.buffer

        self.fill_array(
            buffer=buffer,
            time=time,
        )

        return buffer[volume_slicing]

    def fill_array(
        self,
        buffer: np.ndarray,
        time: int,
    ) -> None:
        """Paint all segments from the specified timepoint whose number of pixels
        is larger than the number of pixels threshold.

        Parameters
        ----------
        buffer : np.ndarray
            Buffer to paint the segments in-place.
        time : int
            Timepoint at which to paint the segments.

        Notes
        -----
        Only segments without painted parents are included to avoid overlapping
        representations.
        """

        engine = sqla.create_engine(self.database_path)
        buffer.fill(0)

        LOG.info("Painting segments at time %d", time)
        LOG.info("num_pix_threshold: %d", self.num_pix_threshold)

        with Session(engine) as session:
            query = list(
                session.query(
                    self.node_attribute, NodeDB.pickle, NodeDB.id, NodeDB.hier_parent_id
                ).where(
                    NodeDB.t == time,
                    NodeDB.area <= int(self.num_pix_threshold),
                )
            )
            if len(query) == 0:
                LOG.warning("No segments to paint at time %d", time)
                return

            attrs, nodes, node_ids, parent_ids = zip(*query)

            node_ids = set(node_ids)  # faster lookup

            count = 0
            for i in range(len(nodes)):
                if parent_ids[i] not in node_ids:
                    print(nodes[i].area)
                    nodes[i].paint_buffer(buffer, value=attrs[i], include_time=False)
                    count += 1

            LOG.info("Painted %d segments", count)

    def get_tp_num_pixels(
        self,
        timeStart: int,
        timeStop: int,
    ) -> List[int]:
        """Get segment pixel counts for a range of timepoints.

        Parameters
        ----------
        timeStart : int
            Starting timepoint (inclusive).
        timeStop : int
            Ending timepoint (inclusive).

        Returns
        -------
        List[int]
            Number of pixels for each segment within the specified time range.
        """
        engine = sqla.create_engine(self.database_path)

        with Session(engine) as session:
            query = (
                session.query(NodeDB.area)
                .where(NodeDB.t.between(timeStart, timeStop))
                .all()
            )
            num_pix_list = [v for v, in query]  # unpacking the query

        return num_pix_list

    def find_min_max_volume_entire_dataset(self) -> np.ndarray:
        """
        Find global minimum and maximum segment number of pixels.

        Queries the database to find the extreme number of pixels across all
        timepoints in the dataset.

        Returns
        -------
        numpy.ndarray
            Array with two integer elements: [minimum_number_of_pixels, maximum_number_of_pixels].
        """
        engine = sqla.create_engine(self.database_path)
        with Session(engine) as session:
            max_vol = (
                session.query(func.max(NodeDB.area))
                .where(NodeDB.t.between(0, self.t_max))
                .scalar()
            )
            min_vol = (
                session.query(func.min(NodeDB.area))
                .where(NodeDB.t.between(0, self.t_max))
                .scalar()
            )

        LOG.info(f"min_vol: {min_vol}, max_vol: {max_vol}")

        return np.asarray([min_vol, max_vol], dtype=int)
