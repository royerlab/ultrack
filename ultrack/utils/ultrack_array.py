from typing import List, Tuple, Union

import numpy as np
import sqlalchemy as sqla
from sqlalchemy import func
from sqlalchemy.orm import Session

from ultrack.config import MainConfig
from ultrack.core.database import NodeDB


class UltrackArray:
    def __init__(
        self,
        config: MainConfig,
        dtype: np.dtype = np.int32,
    ):
        """Initialize an array for visualizing segments in the ultrack database.

        The array provides direct visualization of segments stored in the database,
        allowing efficient access and transversing the hierarchy of segments.

        Parameters
        ----------
        config : MainConfig
            Configuration object containing Ultrack settings and metadata.
        dtype : numpy.dtype, optional
            Data type of the array, by default numpy.int32.

        Notes
        -----
        The array shape is determined from the configuration metadata.
        """

        self.config = config
        self.shape = tuple(config.data_config.metadata["shape"])  # (t,(z),y,x)
        self.dtype = dtype
        self.t_max = self.shape[0]
        self.ndim = len(self.shape)
        self.array = np.zeros(self.shape[1:], dtype=self.dtype)

        self.database_path = config.data_config.database_path
        self.minmax = self.find_min_max_volume_entire_dataset()
        self.volume = self.minmax.mean().astype(int)

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

        self.fill_array(
            time=time,
        )

        return self.array[volume_slicing]

    def fill_array(
        self,
        time: int,
    ) -> None:
        """Paint all segments from the specified timepoint whose number of pixels
        is larger than the number of pixels threshold.

        Parameters
        ----------
        time : int
            Timepoint at which to paint the segments.

        Notes
        -----
        Only segments without painted parents are included to avoid overlapping
        representations.
        """

        engine = sqla.create_engine(self.database_path)
        self.array.fill(0)

        with Session(engine) as session:
            query = list(
                session.query(NodeDB.id, NodeDB.pickle, NodeDB.hier_parent_id).where(
                    NodeDB.t == time
                )
            )

            idx_to_plot = []

            for idx, q in enumerate(query):
                if q[1].area <= self.volume:
                    idx_to_plot.append(idx)

            id_to_plot = [q[0] for idx, q in enumerate(query) if idx in idx_to_plot]
            label_list = np.arange(1, len(query) + 1, dtype=int)

            to_remove = []
            for idx in idx_to_plot:
                if query[idx][2] in id_to_plot:  # if parent is also printed
                    to_remove.append(idx)

            for idx in to_remove:
                idx_to_plot.remove(idx)

            if len(query) == 0:
                print("query is empty!")

            for idx in idx_to_plot:
                query[idx][1].paint_buffer(
                    self.array, value=label_list[idx], include_time=False
                )

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
                .where(NodeDB.t.between(timeStart, timeStop + 1))
                .all()
            )
            num_pix_list = list(query)

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

        return np.asarray([min_vol, max_vol], dtype=int)
