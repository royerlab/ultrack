import logging
from contextlib import nullcontext
from typing import Optional

import fasteners
import mip
import mip.exceptions
import numpy as np
import pandas as pd
import sqlalchemy as sqla
from scipy.signal import medfilt
from skimage.feature import peak_local_max
from sqlalchemy.orm import Session

from ultrack.config.config import MainConfig
from ultrack.core.database import GTLinkDB, NodeDB, OverlapDB

LOG = logging.getLogger(__name__)
LOG.setLevel(logging.INFO)


def _find_peaks(
    area: np.ndarray,
    resolution: int = 100,
    num_peaks: int = 5,
    min_distance_ratio: float = 0.25,
) -> np.ndarray:
    """
    Find peaks in the area distribution in log2 space.

    Parameters
    ----------
    area : np.ndarray
        Area of the nodes.
    resolution : int, optional
        Resolution when binning the area.

    Returns
    -------
    np.ndarray
        Areas of the peaks.
    """
    log_area = (np.log2(area) * resolution).astype(int)

    area_bins = np.bincount(log_area)
    area_bins = medfilt(area_bins, kernel_size=5)
    peaks = peak_local_max(
        area_bins,
        min_distance=int(resolution * min_distance_ratio),
        num_peaks=num_peaks,
    )
    if len(peaks) == 0:
        return np.atleast_1d(np.mean(area).astype(int))

    peak_areas = 2 ** (peaks / resolution)

    LOG.info(f"Found {len(peak_areas)} peaks in {resolution} resolution")
    LOG.info(f"Peaks: {peak_areas}")

    return peak_areas.ravel()


class SQLGTMatcher:
    def __init__(
        self,
        config: MainConfig,
        write_lock: Optional[fasteners.InterProcessLock] = None,
        eps: float = 1e-5,
    ) -> None:
        """
        Ground-truth matching solver from SQL database content.

        Parameters
        ----------
        config : MainConfig
            Ultrack's main configuration parameters.
        write_lock : Optional[fasteners.InterProcessLock], optional
            Lock object for SQLite multiprocessing.
        eps : float, optional
            Weight epsilon to prefer not selecting bad matches than bad ones.
        """
        self._data_config = config.data_config
        self._write_lock = write_lock
        self._connect_args = {"timeout": 45} if self._write_lock is not None else {}
        self._eps = eps

        try:
            self._model = mip.Model(
                sense=mip.MAXIMIZE, solver_name=config.tracking_config.solver_name
            )
        except mip.exceptions.InterfacingError as e:
            LOG.warning(e)
            self._model = mip.Model(sense=mip.MAXIMIZE, solver_name="CBC")

    def _add_nodes(self, time: int) -> None:
        """
        Add nodes to the ILP model.

        Parameters
        ----------
        time : int
            Time point to query and match.
        """
        engine = sqla.create_engine(
            self._data_config.database_path, connect_args=self._connect_args
        )

        with Session(engine) as session:
            query = session.query(NodeDB.id, NodeDB.area).where(NodeDB.t == time)
            self._nodes_df = pd.read_sql(query.statement, session.bind, index_col="id")

        size = len(self._nodes_df)
        self._nodes = self._model.add_var_tensor(
            (size,), name="nodes", var_type=mip.BINARY
        )

        # hierarchy overlap constraints
        with Session(engine) as session:
            query = session.query(OverlapDB).join(
                NodeDB, NodeDB.id == OverlapDB.node_id
            )
            overlap_df = pd.read_sql(query.statement, session.bind)

        overlap_df["node_id"] = self._nodes_df.index.get_indexer(overlap_df["node_id"])
        overlap_df["ancestor_id"] = self._nodes_df.index.get_indexer(
            overlap_df["ancestor_id"]
        )

        for node_id, anc_id in zip(overlap_df["node_id"], overlap_df["ancestor_id"]):
            self._model.add_constr(self._nodes[node_id] + self._nodes[anc_id] <= 1)

    def _add_templates(self, time: int) -> None:
        """
        Add templates to the ILP model.

        Parameters
        ----------
        time : int
            Time point to query and match.
        """
        LOG.info(f"Adding templates for time {time}")

        area = self._nodes_df["area"].to_numpy()

        size = len(self._nodes_df)
        self._peaks = _find_peaks(area)

        self._template_edges = self._model.add_var_tensor(
            (size, len(self._peaks)), name="template_edges", var_type=mip.BINARY
        )

        self._not_templates = self._model.add_var_tensor(
            (len(self._peaks),), name="templates", var_type=mip.BINARY
        )

        # each active node must have one template
        for n in range(size):
            self._model.add_constr(
                mip.xsum(self._template_edges[n, :]) == self._nodes[n]
            )

        # only one template must be off
        self._model.add_constr(mip.xsum(self._not_templates) == len(self._peaks) - 1)

        max_area = np.max(area)

        for r in range(len(self._peaks)):

            # not_template is off when all template edges are on
            self._model.add_constr(
                mip.xsum(1 - self._template_edges[:, r]) >= self._not_templates[r]
            )

            # minimize the difference between the template and the area
            self._model.objective += mip.xsum(
                [
                    # (area[s] - abs(self._peaks[r] - area[s])) * self._template_edges[s, r]
                    (max_area - abs(self._peaks[r] - area[s]))
                    * self._template_edges[s, r]
                    for s in range(size)
                ]
            )

        LOG.info(f"Templates added for time {time}")

    def _add_edges(self, time: int) -> None:
        """
        Add edges to the ILP model.

        Parameters
        ----------
        time : int
            Time point to query and match.
        """
        if not hasattr(self, "_nodes"):
            raise ValueError("Nodes must be added before adding edges.")

        engine = sqla.create_engine(
            self._data_config.database_path, connect_args=self._connect_args
        )

        with Session(engine) as session:

            query = (
                session.query(GTLinkDB)
                .join(NodeDB, NodeDB.id == GTLinkDB.source_id)
                .where(NodeDB.t == time)
            )

            self._edges_df = pd.read_sql(query.statement, session.bind)

        self._edges_df["source_id"] = self._nodes_df.index.get_indexer(
            self._edges_df["source_id"]
        )
        self._edges_df.reset_index(drop=True, inplace=True)

        self._edges = self._model.add_var_tensor(
            (len(self._edges_df),),
            name="edges",
            var_type=mip.BINARY,
        )
        # small value to prefer not selecting edges than bad ones
        # setting objective function
        self._model.objective = mip.xsum(
            (self._edges_df["weight"].to_numpy() - self._eps) * self._edges
        )

        # source_id is time point T (hierarchies id)
        # target_id is time point T+1 (ground-truth)
        for source_id, group in self._edges_df.groupby("source_id", as_index=False):
            self._model.add_constr(
                self._nodes[source_id] == mip.xsum(self._edges[group.index.to_numpy()])
            )

        for _, group in self._edges_df.groupby("target_id", as_index=False):
            self._model.add_constr(mip.xsum(self._edges[group.index.to_numpy()]) <= 1)

    def add_solution(self) -> None:
        """
        Add the solution to the database.
        """
        engine = sqla.create_engine(
            self._data_config.database_path, connect_args=self._connect_args
        )

        edges_records = []
        for idx, e_var in zip(self._edges_df["id"], self._edges):
            if e_var.x > 0.5:
                edges_records.append(
                    {
                        "link_id": idx,
                        "selected": True,
                    }
                )

        LOG.info(f"Selected {len(edges_records)} edges to ground-truth")

        with self._write_lock if self._write_lock is not None else nullcontext():
            with Session(engine) as session:
                stmt = (
                    sqla.update(GTLinkDB)
                    .where(GTLinkDB.id == sqla.bindparam("link_id"))
                    .values(selected=sqla.bindparam("selected"))
                )
                session.connection().execute(
                    stmt,
                    edges_records,
                    execution_options={"synchronize_session": False},
                )
                session.commit()

    def __call__(self, time: int, match_templates: bool) -> float:
        """
        Build the ground-truth matching ILP and solve it.

        Parameters
        ----------
        time : int
            Time point to query and match.
        match_templates : bool
            Whether to minimize the difference between the template.

        Returns
        -------
        float
            Objective value.
        """
        LOG.info(f"Computing GT matching for time {time}")

        self._add_nodes(time)
        self._add_edges(time)
        if match_templates:
            self._add_templates(time)
        self._model.optimize()
        self.add_solution()

        n_selected_vars = sum(e_var.x > 0.5 for e_var in self._edges)
        if match_templates:
            for r in range(len(self._not_templates)):
                if self._not_templates[r].x < 0.5:
                    LOG.info(f"Template {self._peaks[r]} was selected")

        return self._model.objective_value + n_selected_vars * self._eps
