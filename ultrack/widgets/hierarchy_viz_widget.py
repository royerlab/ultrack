import logging
from typing import Sequence

import napari
import numpy as np
from magicgui.widgets import ComboBox, Container, FloatSlider, Label
from scipy import interpolate
from sqlalchemy import PickleType

from ultrack.config import MainConfig
from ultrack.core.database import NodeDB
from ultrack.utils.ultrack_array import UltrackArray
from ultrack.widgets.ultrackwidget import UltrackWidget

logging.basicConfig()
logging.getLogger("sqlachemy.engine").setLevel(logging.INFO)

LOG = logging.getLogger(__name__)
# LOG.setLevel(logging.INFO)


class HierarchyVizWidget(Container):
    HIER_LAYER_NAME = "Ultrack Hierarchy"
    NODE_ATTRIBUTES = {
        column.name: column
        for column in NodeDB.__table__.columns
        if not isinstance(column.type, PickleType)
    }

    def __init__(
        self,
        viewer: napari.Viewer,
        config=None,
        **kwargs,
    ) -> None:
        """
        Initialize the HierarchyVizWidget.

        Parameters
        ----------
        viewer : napari.Viewer
            The napari viewer instance.
        config : MainConfig of Ultrack
            if not provided, config will be taken from UltrackWidget
        **kwargs : dict
            Additional keyword arguments to pass to the UltrackArray constructor.
        """
        super().__init__(layout="vertical")

        self._viewer = viewer
        title_label = Label(value="<h2>Hierarchy Viz. Widget</h2>")
        self.append(title_label)

        if config is None:
            self.config = self._get_config()
        else:
            self.config = config

        self._ultrack_array = UltrackArray(self.config, **kwargs)

        # TODO: might need to be updated when the slider is changed
        self._mapping = self._create_mapping()

        # Create area threshold container with fixed width and margins
        self._area_widget_container = Container(layout="horizontal", label="Area")
        self.append(self._area_widget_container)

        # Configure slider label
        self._slider_label = Label(label="0")
        self._slider_label.native.setFixedWidth(25)

        # Configure area threshold slider
        self._area_threshold_w = FloatSlider(
            min=0,
            max=1,
            readout=False,
            value=0.5,
            tooltip="Scroll to change the threshold and transverse the hierarchy",
        )
        self._area_threshold_w.changed.connect(self._slider_update)

        self._area_widget_container.append(self._area_threshold_w)
        self._area_widget_container.append(self._slider_label)

        # Configure node attribute combo box
        self._node_attribute_w = ComboBox(
            label="Node Attribute",
            choices=list(self.NODE_ATTRIBUTES.keys()),
            value="id",
        )
        self._node_attribute_w.changed.connect(self._on_node_attribute_changed)
        self.append(self._node_attribute_w)

        self._add_ultrack_array()

        self._viewer.layers[self.HIER_LAYER_NAME].refresh()

    def _on_config_changed(self) -> None:
        self._ndim = len(self._shape)

    def _add_ultrack_array(self) -> None:
        if self.HIER_LAYER_NAME in self._viewer.layers:
            self._viewer.layers.remove(self.HIER_LAYER_NAME)

        scale = self.config.data_config.metadata.get("scale", [1, 1])

        try:
            self._viewer.add_labels(
                self._ultrack_array,
                name=self.HIER_LAYER_NAME,
                scale=scale,
            )
        except TypeError:
            self._viewer.add_image(
                self._ultrack_array,
                name=self.HIER_LAYER_NAME,
                scale=scale,
                colormap="magma",
            )

    def _on_node_attribute_changed(self, value: str) -> None:
        self._ultrack_array.node_attribute = self.NODE_ATTRIBUTES.get(value, NodeDB.id)
        self._add_ultrack_array()

    @property
    def _shape(self) -> Sequence[int]:
        return self.config.metadata.get("shape", [])

    def _slider_update(self, value: float) -> None:
        mapped_value = self._mapping(value)
        LOG.info("value %d mapped_value: %d", value, mapped_value)
        self._ultrack_array.num_pix_threshold = mapped_value
        self._slider_label.label = str(int(mapped_value))
        self._viewer.layers[self.HIER_LAYER_NAME].refresh()

    def _create_mapping(self) -> interpolate.interp1d:
        """
        Creates a pseudo-linear mapping from U[0,1] to full range of number of pixels
            num_pixels = mapping([0,1])
        """
        current_time = int(
            self._viewer.dims.point[0]
        )  # we assume that time is the first dim
        num_pixels_list = self._ultrack_array.get_tp_num_pixels(
            timeStart=current_time, timeStop=current_time
        )
        num_pixels_list.append(self._ultrack_array.minmax[0])
        num_pixels_list.append(self._ultrack_array.minmax[1])
        num_pixels_list.sort()

        x_vec = np.linspace(0, 1, len(num_pixels_list))
        y_vec = np.array(num_pixels_list)
        mapping = interpolate.interp1d(x_vec, y_vec)
        return mapping

    def _get_config(self) -> MainConfig:
        """
        Gets config from the Ultrack widget
        """
        ultrack_widget = UltrackWidget.find_ultrack_widget(self._viewer)
        if ultrack_widget is None:
            raise TypeError(
                "config not provided and was not found within ultrack widget"
            )

        return ultrack_widget._data_forms.get_config()
