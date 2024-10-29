import logging
from typing import List, Optional, Sequence

import napari
import numpy as np
from scipy import interpolate
from magicgui.widgets import FloatSlider, Container, Label

from ultrack.utils.ultrackArray import UltrackArray
from ultrack.config import MainConfig
from ultrack.widgets.ultrackwidget import UltrackWidget

logging.basicConfig()
logging.getLogger("sqlachemy.engine").setLevel(logging.INFO)

LOG = logging.getLogger(__name__)


class HierarchyVizWidget(Container):
    def __init__(self, 
                 viewer: napari.Viewer,
                 config = None, 
        ) -> None:
        """
        Initialize the HierarchyVizWidget.

        Parameters
        ----------
        viewer : napari.Viewer
            The napari viewer instance.
        config : MainConfig of Ultrack
            if not provided, config will be taken from UltrackWidget
        """
            
        super().__init__(layout='horizontal')
        
        self._viewer = viewer

        if config is None:
            self.config = self._get_config()
        else:
            self.config = config

        self.ultrack_array = UltrackArray(self.config)

        self.mapping = self._create_mapping()

        self._area_threshold_w = FloatSlider(label="Area", min=0, max=1, readout=False)
        self._area_threshold_w.value = 0.5
        self.ultrack_array.volume = self.mapping(0.5)
        self._area_threshold_w.changed.connect(self._slider_update)

        self.slider_label = Label(label=str(int(self.mapping(self._area_threshold_w.value))))
        self.slider_label.native.setFixedWidth(25)

        self.append(self._area_threshold_w)
        self.append(self.slider_label)

        #THERE SHOULD BE CHECK HERE IF THERE EXISTS A LAYER WITH THE NAME 'HIERARCHY'
        self._viewer.add_labels(self.ultrack_array, name='hierarchy')
        self._viewer.layers['hierarchy'].refresh()

    def _on_config_changed(self) -> None:
        self._ndim = len(self._shape)

    @property
    def _shape(self) -> Sequence[int]:
        return self.config.metadata.get("shape", [])

    def _slider_update(self, value: float) -> None:
        self.ultrack_array.volume = self.mapping(value)
        self.slider_label.label = str(int(self.mapping(value)))
        self._viewer.layers['hierarchy'].refresh()

    def _create_mapping(self):
        """
        Creates a pseudo-linear mapping from U[0,1] to full range of number of pixels
            num_pixels = mapping([0,1])
        """
        num_pixels_list = self.ultrack_array.get_tp_num_pixels(timeStart=5,timeStop=5)
        num_pixels_list.append(self.ultrack_array.minmax[0])
        num_pixels_list.append(self.ultrack_array.minmax[1])
        num_pixels_list.sort()

        x_vec = np.linspace(0,1,len(num_pixels_list))
        y_vec = np.array(num_pixels_list)
        mapping = interpolate.interp1d(x_vec,y_vec)
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