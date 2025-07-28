import types
import sys
from typing import Dict

import numpy as np
import pytest
from click.testing import CliRunner


@pytest.fixture(autouse=True)
def _stub_external_modules(monkeypatch):
    """Stub heavy optional dependencies that are not required for CLI parsing.

    The `trackmate` CLI relies on *napari* and *zarr*. Importing these heavy
    libraries is unnecessary for the unit-tests and might fail on CI where they
    are not installed.  Instead we register lightweight stub modules **before**
    the CLI implementation is imported so the import succeeds without the real
    dependencies.
    """
    # ------------------------------------------------------------------
    # stub napari (nested modules: napari.plugins, napari.viewer)
    # ------------------------------------------------------------------
    napari_mod = types.ModuleType("napari")

    # sub-module: napari.plugins
    plugins_mod = types.ModuleType("napari.plugins")

    def _initialize_plugins():  # dummy implementation
        return None

    plugins_mod._initialize_plugins = _initialize_plugins
    napari_mod.plugins = plugins_mod

    # sub-module: napari.viewer with a minimal ViewerModel replacement
    viewer_mod = types.ModuleType("napari.viewer")

    class _DummyLayer:  # minimal stand-in for napari Layer
        def __init__(self, data):
            self.data = data
            self.multiscale = False
            self.name = "layer"

    class _DummyViewerModel:
        """Very small subset of napari.viewer.ViewerModel API"""
        def __init__(self):
            self.layers = {"layer": _DummyLayer(np.zeros((1, 1)))}

        def open(self, *args, **kwargs):  # noqa: D401 – no-op
            return None

    viewer_mod.ViewerModel = _DummyViewerModel
    napari_mod.viewer = viewer_mod

    # Register stub hierarchy before importing the CLI module
    sys.modules.update(
        {
            "napari": napari_mod,
            "napari.plugins": plugins_mod,
            "napari.viewer": viewer_mod,
        }
    )

    # ------------------------------------------------------------------
    # stub zarr – only the ``open`` function is used in _is_zarr_directory
    # ------------------------------------------------------------------
    zarr_mod = types.ModuleType("zarr")

    def _fake_open(path, mode="r", *args, **kwargs):
        return None  # pretend success

    zarr_mod.open = _fake_open
    sys.modules["zarr"] = zarr_mod


@pytest.fixture
def _import_trackmate():
    """Import the CLI module (with stubs already installed)."""
    import importlib

    module = importlib.import_module("ultrack.cli.trackmate")
    module = importlib.reload(module)  # make sure we load fresh each time
    return module


def _patch_pipeline(monkeypatch, trackmate_module):
    """Patch the heavy computational functions with lightweight fakes.

    Returns a dict capturing the call arguments so the tests can inspect them.
    """
    call_info: Dict[str, Dict] = {}

    # Replace _get_data so the test runs without files or napari
    def _fake_get_data(paths, data_type, sigma, reader_plugin):
        call_info["get_data"] = {
            "paths": paths,
            "data_type": data_type,
            "sigma": sigma,
            "reader_plugin": reader_plugin,
        }
        return np.zeros((1, 1)), np.zeros((1, 1))  # foreground, contours

    monkeypatch.setattr(trackmate_module, "_get_data", _fake_get_data)

    # Replace heavy steps
    def _fake_segment(fg, ct, config, overwrite):
        call_info["segment"] = {
            "foreground_shape": fg.shape,
            "contours_shape": ct.shape,
            "config": config,
            "overwrite": overwrite,
        }

    def _fake_link(config, overwrite):
        call_info["link"] = {"config": config, "overwrite": overwrite}

    def _fake_solve(config, overwrite):
        call_info["solve"] = {"config": config, "overwrite": overwrite}

    monkeypatch.setattr(trackmate_module, "segment", _fake_segment)
    monkeypatch.setattr(trackmate_module, "link", _fake_link)
    monkeypatch.setattr(trackmate_module, "solve", _fake_solve)

    # Stub exporters (avoid IO)
    monkeypatch.setattr(trackmate_module, "to_geff", lambda *a, **kw: None)
    monkeypatch.setattr(trackmate_module, "to_trackmate", lambda *a, **kw: None)

    return call_info


# -----------------------------------------------------------------------------
#                                    TESTS
# -----------------------------------------------------------------------------


def test_trackmate_cli_default(tmp_path, monkeypatch, _import_trackmate):
    """CLI runs end-to-end with default parameters (no overrides)."""
    trackmate = _import_trackmate
    call_info = _patch_pipeline(monkeypatch, trackmate)

    runner = CliRunner()
    output_path = tmp_path / "out.geff"

    result = runner.invoke(
        trackmate.trackmate_cli,
        [str(tmp_path), "-o", str(output_path), "--overwrite"],
    )

    assert result.exit_code == 0, result.output
    assert {"segment", "link", "solve"}.issubset(call_info)
    assert call_info["segment"]["overwrite"] is True


def test_trackmate_cli_parameter_override(tmp_path, monkeypatch, _import_trackmate):
    """Override parsing: ensure CLI arguments update the config."""
    trackmate = _import_trackmate
    call_info = _patch_pipeline(monkeypatch, trackmate)

    runner = CliRunner()
    output_path = tmp_path / "result.xml"

    override_str = "segmentation.threshold=0.75"

    result = runner.invoke(
        trackmate.trackmate_cli,
        [str(tmp_path), "-o", str(output_path), "--overwrite", override_str],
    )

    assert result.exit_code == 0, result.output

    seg_cfg = call_info["segment"]["config"].segmentation_config
    assert pytest.approx(seg_cfg.threshold) == 0.75