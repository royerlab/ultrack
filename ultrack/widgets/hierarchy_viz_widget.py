import logging
from typing import Dict, List, Optional, Sequence
# from warnings import warn

import napari
import numpy as np
# import pandas as pd
from scipy import interpolate
# import sqlalchemy as sqla
from magicgui.widgets import FloatSlider, Container, Label
# from qtpy.QtWidgets import QWidget, QVBoxLayout, QLabel, QPushButton, QSlider
# from qtpy.QtCore import Qt

from ultrack.utils.array import UltrackArray

logging.basicConfig()
logging.getLogger("sqlachemy.engine").setLevel(logging.INFO)

LOG = logging.getLogger(__name__)


class HierarchyVizWidget(Container):
    def __init__(self, 
                 viewer: napari.Viewer,
                 new_config = None, 
        ) -> None:
        super().__init__(layout='horizontal')
        
        self._viewer = viewer

        if new_config is None:
            print('ULTRACK WIDGET NOT OPEN!!!')
            #load the config from Ultrack widget
        else:
            self.new_config = new_config

        print('Check if hierarchy doesnt exist already!')
        self.ultrack_layer = UltrackArray(self.new_config)
        self._viewer.add_labels(self.ultrack_layer, name='hierarchy')

        self.mapping = self._create_mapping()

        self._area_threshold_w = FloatSlider(label="Area", min=0, max=1, readout=False)
        self._area_threshold_w.value    = 0.5
        self._area_threshold_w.changed.connect(self._slider_update)

        self.slider_label = Label(label=str(self.mapping(self._area_threshold_w.value)))
        self.slider_label.native.setFixedWidth(100)
        self.append(self._area_threshold_w)
        self.append(self.slider_label)

        # self._area_threshold_w.max      = self.ultrack_layer.minmax[1]+1
        # self._area_threshold_w.min      = self.ultrack_layer.minmax[0]-1
        # self._area_threshold_w.value    = self.ultrack_layer.initial_volume

    def _on_config_changed(self) -> None:
        self._ndim = len(self._shape)

    @property
    def _shape(self) -> Sequence[int]:
        return self.config.metadata.get("shape", [])

    def _slider_update(self, value: float) -> None:
        self.ultrack_layer.volume = self.mapping(value)
        self.slider_label.label = str(int(self.mapping(value)))
        # print(len(self._area_threshold_w.label))
        self._viewer.layers['hierarchy'].refresh()

    def _create_mapping(self):
        volume_list = self.ultrack_layer.get_volume_list(timeLimit=5)
        volume_list.append(self.ultrack_layer.minmax[0])
        volume_list.append(self.ultrack_layer.minmax[1])
        volume_list.sort()

        x_vec = np.linspace(0,1,len(volume_list))
        y_vec = np.array(volume_list)
        mapping = interpolate.interp1d(x_vec,y_vec)
        return mapping