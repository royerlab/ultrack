from typing import Tuple, Union

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
        """Create an array that directly visualizes the segments in the ultrack database.

        Parameters
        ----------
        config : MainConfig
            Configuration file of Ultrack.
        dtype : np.dtype
            Data type of the array.
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
        """Indexing the ultrack-array

        Parameters
        ----------
        indexing : Tuple or Array

        Returns
        -------
        array : numpy array
            array with painted segments
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
        """Paint all segments of specific time point which volume is bigger than self.volume
        Parameters
        ----------
        time : int
            time point to paint the segments
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
    ) -> list:
        """Gets a list of number of pixels of all segments range of time points (timeStart to timeStop)
        Parameters
        ----------
        timeStart : int
        timeStop : int
        Returns
        -------
        num_pix_list : list
            list with all num_pixels for timeStart to timeStop
        """
        engine = sqla.create_engine(self.database_path)
        num_pix_list = []
        with Session(engine) as session:
            query = list(
                session.query(NodeDB.area).where(NodeDB.t.between(timeStart, timeStop))
            )
            for num_pix in query:
                num_pix_list.append(int(np.array(num_pix)))
        return num_pix_list

    def find_min_max_volume_entire_dataset(self):
        """Find minimum and maximum segment volume for ALL time point

        Returns
        -------
        np.array : np.array
            array with two elements: [min_volume, max_volume]
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

        return np.array([min_vol, max_vol], dtype=int)
