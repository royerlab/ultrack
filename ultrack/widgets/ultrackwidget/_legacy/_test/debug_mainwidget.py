"""
Auxiliary file to log and debug main widget usage.
"""

import logging

logging.basicConfig(filename="ultrackwidget.log", filemode="w", level=logging.INFO)

import napari

from ultrack.widgets.ultrackwidget import UltrackWidget

if __name__ == "__main__":
    viewer = napari.Viewer()
    widget = UltrackWidget(viewer)
    viewer.window.add_dock_widget(widget)
    napari.run()
