import napari
from magicgui.widgets import Container, FloatSpinBox, PushButton, SpinBox
from napari.layers import Labels

from ultrack.utils.edge import labels_to_contours


class LabelsToContoursWidget(Container):
    def __init__(self, viewer: napari.Viewer) -> None:
        super().__init__()
        self._viewer = viewer

        self._sigma_w = FloatSpinBox(min=0, max=25, value=0, label="Smoothing")
        self.append(self._sigma_w)

        self._selection_w = SpinBox(label="Num. selected layers", enabled=False)
        self.append(self._selection_w)
        self._update_count()

        self._viewer.layers.selection.events.changed.connect(self._update_count)

        self._run_btn = PushButton(text="Convert")
        self._run_btn.clicked.connect(self._on_run)
        self.append(self._run_btn)

    def _update_count(self) -> None:
        self._selection_w.value = len(self._viewer.layers.selection)

    def _on_run(self) -> None:
        for layer in self._viewer.layers.selection:
            if not isinstance(layer, Labels):
                raise ValueError(f"Layer named `{layer.name}` isn't of `Labels` type.")

        selection = list(layer.data for layer in self._viewer.layers.selection)
        if len(selection) == 0:
            raise ValueError("Select labels for conversion.")

        sigma = self._sigma_w.value if self._sigma_w.value > 0 else None

        foreground, contours = labels_to_contours(selection, sigma)

        self._viewer.add_image(foreground, name="foreground")
        self._viewer.add_image(contours, name="contours")
