from npe2 import PluginManifest


def test_napari_plugin_manifest() -> None:
    PluginManifest.from_distribution("ultrack")
