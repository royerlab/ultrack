import functools
from enum import Flag, auto
from typing import Dict, Optional, Tuple

import networkx as nx
import pandas as pd
import zarr
from numpy.typing import ArrayLike

from ultrack import export_tracks_by_extension
from ultrack.config import MainConfig
from ultrack.core.export import (
    to_ctc,
    to_tracks_layer,
    tracks_layer_to_networkx,
    tracks_layer_to_trackmate,
    tracks_to_zarr,
)
from ultrack.core.linking.processing import link
from ultrack.core.main import track
from ultrack.core.segmentation.processing import segment
from ultrack.core.solve.processing import solve
from ultrack.imgproc.flow import add_flow
from ultrack.utils.deprecation import rename_argument


class TrackerStatus(Flag):
    NOT_COMPUTED = auto()
    SEGMENTED = auto()
    LINKED = auto()
    SOLVED = auto()


class Tracker:
    """An Ultrack wrapper for its core functionalities.

    Parameters
    ----------
    config : MainConfig
        The configuration parameters.

    Attributes
    ----------
    config : MainConfig
        The configuration parameters.
    status : TrackerStatus
        The status of the tracking process.

    Examples
    --------
    >>> import ultrack
    >>> from ultrack import MainConfig
    >>> config = MainConfig()
    >>> foreground = ...
    >>> contours = ...
    >>> vector_field = ...
    >>> tracker = ultrack.Tracker(config)
    >>> tracker.segment(foreground=foreground, contours=contours)
    >>> tracker.add_flow(vector_field=vector_field)
    >>> tracker.link()
    >>> tracker.solve()
    """

    def __init__(self, config: MainConfig) -> None:
        self.config = config
        self.status = TrackerStatus.NOT_COMPUTED

    @functools.wraps(segment)
    @rename_argument("detection", "foreground")
    @rename_argument("edges", "contours")
    def segment(self, foreground: ArrayLike, contours: ArrayLike, **kwargs) -> None:
        segment(foreground=foreground, contours=contours, config=self.config, **kwargs)
        self.status = TrackerStatus.SEGMENTED

    @functools.wraps(add_flow)
    def add_flow(self, vector_field: ArrayLike) -> None:
        if TrackerStatus.SEGMENTED not in self.status:
            raise ValueError("You must call `segment` before calling `add_flow`.")
        add_flow(config=self.config, vector_field=vector_field)

    @functools.wraps(link)
    def link(self, *args, **kwargs) -> None:
        if TrackerStatus.SEGMENTED not in self.status:
            raise ValueError("You must call `segment` before calling `link`.")
        link(config=self.config, *args, **kwargs)
        self.status = TrackerStatus.LINKED

    @functools.wraps(solve)
    def solve(self, *args, **kwargs) -> None:
        if TrackerStatus.LINKED not in self.status:
            raise ValueError("You must call `segment` & `link` before calling `solve`.")
        solve(config=self.config, *args, **kwargs)
        self.status = TrackerStatus.SOLVED

    @functools.wraps(track)
    def track(self, *args, **kwargs) -> None:
        track(config=self.config, *args, **kwargs)
        self.status = TrackerStatus.SOLVED

    def _assert_solved(self) -> None:
        """Raise an error if the tracking is not solved."""
        if TrackerStatus.SOLVED not in self.status:
            raise ValueError(
                "The tracking is not ready! Please make sure that you "
                "called `segment` &a `link` & `solve` or `track`."
            )

    @functools.wraps(tracks_layer_to_networkx)
    def to_networkx(
        self, *, tracks_df: Optional[pd.DataFrame] = None, **kwargs
    ) -> nx.DiGraph:
        self._assert_solved()
        if tracks_df is None:
            tracks_df, _ = to_tracks_layer(self.config)
        tracks_nx = tracks_layer_to_networkx(tracks_df, **kwargs)
        return tracks_nx

    @functools.wraps(to_tracks_layer)
    def to_pandas(self, *args, **kwargs) -> pd.DataFrame:
        self._assert_solved()
        tracks_df, _ = to_tracks_layer(config=self.config, *args, **kwargs)
        return tracks_df

    @functools.wraps(tracks_to_zarr)
    def to_zarr(self, **kwargs) -> zarr.Array:
        self._assert_solved()
        tracks_df = kwargs.pop("tracks_df", None)
        if tracks_df is None:
            tracks_df, _ = to_tracks_layer(self.config)
        segments = tracks_to_zarr(self.config, tracks_df=tracks_df, **kwargs)
        return segments

    @functools.wraps(tracks_layer_to_trackmate)
    def to_trackmate(self, tracks_df: Optional[pd.DataFrame] = None) -> str:
        self._assert_solved()
        if tracks_df is None:
            tracks_df, _ = to_tracks_layer(config=self.config)
        trackmate_xml = tracks_layer_to_trackmate(tracks_df)
        return trackmate_xml

    @functools.wraps(to_ctc)
    def to_ctc(self, *args, **kwargs) -> None:
        self._assert_solved()
        to_ctc(config=self.config, *args, **kwargs)

    @functools.wraps(to_tracks_layer)
    def to_napari(self, *args, **kwargs) -> Tuple[pd.DataFrame, Dict]:
        self._assert_solved()
        tracks_df, graph = to_tracks_layer(self.config, *args, **kwargs)
        return tracks_df, graph

    @functools.wraps(to_tracks_layer)
    def export_by_extension(self, filename: str, overwrite: bool = False) -> None:
        self._assert_solved()
        export_tracks_by_extension(self.config, filename, overwrite=overwrite)
