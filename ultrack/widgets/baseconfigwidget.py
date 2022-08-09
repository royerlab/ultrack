from abc import abstractmethod
from typing import Dict

from magicgui.widgets import Container, Label
from pydantic import BaseModel
from toolz import curry


class BaseConfigWidget(Container):
    def __init__(self, label: str, config: BaseModel):
        super().__init__()
        self.append(Label(label=label))
        self._attr_to_widget: Dict[str, Container] = {}
        self._setup_widgets()
        self.config = config

    @abstractmethod
    def _setup_widgets(self) -> None:
        pass

    @property
    def config(self) -> BaseModel:
        return self._config

    @config.setter
    def config(self, value: BaseModel) -> None:
        """Sets config and updates the sub widgets values"""
        self._config = value
        for k, v in self._config:
            # some parameters might not be exposed in the UI
            if k in self._attr_to_widget:
                widget = self._attr_to_widget[k]
                widget.changed.disconnect()
                widget.value = v
                widget.changed.connect(curry(setattr, self._config, k))
