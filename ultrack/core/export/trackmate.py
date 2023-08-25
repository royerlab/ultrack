import xml.dom.minidom as minidom
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional, Union

import pandas as pd

from ultrack.config.config import MainConfig
from ultrack.core.database import NO_PARENT
from ultrack.core.export.tracks_layer import to_tracks_layer


def _set_filter_elem(elem: ET.Element) -> None:
    elem.set("feature", "QUALITY")
    elem.set("value", "0.0")
    elem.set("isabove", "true")


def tracks_layer_to_trackmate(
    tracks_df: pd.DataFrame,
) -> str:
    """
    Convert a pandas DataFrame representation of Napari track layer to TrackMate XML format.

    Parameters
    ----------
    tracks_df : pd.DataFrame
        A DataFrame with columns `track_id, id, parent_id, t, z, y, x`. Cells that belong to the same track
        have the same `track_id`.

    Returns
    -------
    str
        A string representation of the XML in the TrackMate format.
    """

    # Create XML root and child elements
    root = ET.Element("TrackMate")
    model_elem = ET.SubElement(root, "Model")
    tracks_elem = ET.SubElement(model_elem, "TrackCollection")
    all_spots_elem = ET.SubElement(model_elem, "AllSpots")
    features_elem = ET.SubElement(model_elem, "FeatureDeclarations")

    settings_elem = ET.SubElement(root, "Settings")
    _set_filter_elem(ET.SubElement(settings_elem, "InitialSpotFilter"))
    _set_filter_elem(ET.SubElement(settings_elem, "SpotFilterCollection"))

    has_z = "z" in tracks_df.columns

    # Create spot features
    spot_features_elem = ET.SubElement(features_elem, "SpotFeatures")
    spot_features = [
        ("QUALITY", "Quality", "Quality", "QUALITY", "false"),
        ("POSITION_X", "X", "X", "POSITION", "false"),
        ("POSITION_Y", "Y", "Y", "POSITION", "false"),
        ("POSITION_Z", "Z", "Z", "POSITION", "false"),
        ("POSITION_T", "T", "T", "TIME", "false"),
        ("FRAME", "Frame", "Frame", "NONE", "true"),
        ("RADIUS", "Radius", "R", "LENGTH", "false"),
        ("VISIBILITY", "Visibility", "Visibility", "NONE", "true"),
        ("MANUAL_COLOR", "Manual spot color", "Spot color", "NONE", "true"),
        ("MEAN_INTENSITY", "Mean intensity", "Mean", "INTENSITY", "false"),
        ("MEDIAN_INTENSITY", "Median intensity", "Median", "INTENSITY", "false"),
        ("MIN_INTENSITY", "Minimal intensity", "Min", "INTENSITY", "false"),
        ("MAX_INTENSITY", "Maximal intensity", "Max", "INTENSITY", "false"),
        ("TOTAL_INTENSITY", "Total intensity", "Total int.", "INTENSITY", "false"),
        ("STANDARD_DEVIATION", "Standard deviation", "Stdev.", "INTENSITY", "false"),
        ("ESTIMATED_DIAMETER", "Estimated diameter", "Diam.", "LENGTH", "false"),
        ("CONTRAST", "Contrast", "Constrast", "NONE", "false"),
        ("SNR", "Signal/Noise ratio", "SNR", "NONE", "false"),
    ]
    for feature, name, shortname, dimension, isint in spot_features:
        feature_elem = ET.SubElement(spot_features_elem, "Feature")
        feature_elem.set("feature", feature)
        feature_elem.set("name", name)
        feature_elem.set("shortname", shortname)
        feature_elem.set("dimension", dimension)
        feature_elem.set("isint", isint)

    # Create edge features
    # Create edge features
    edge_features_elem = ET.SubElement(features_elem, "EdgeFeatures")
    edge_features = [
        ("SPOT_SOURCE_ID", "Source spot ID", "Source ID", "NONE", "true"),
        ("SPOT_TARGET_ID", "Target spot ID", "Target ID", "NONE", "true"),
        # ... add other edge features if needed
    ]
    for feature, name, shortname, dimension, isint in edge_features:
        feature_elem = ET.SubElement(edge_features_elem, "Feature")
        feature_elem.set("feature", feature)
        feature_elem.set("name", name)
        feature_elem.set("shortname", shortname)
        feature_elem.set("dimension", dimension)
        feature_elem.set("isint", isint)

    # Create spots
    for frame, group in tracks_df.groupby("t"):
        frame_elem = ET.SubElement(all_spots_elem, "SpotsInFrame")
        frame_elem.set("frame", str(frame))
        for spot_id, entry in group.iterrows():
            spot_elem = ET.SubElement(frame_elem, "Spot")
            spot_elem.set("ID", str(spot_id))
            spot_elem.set("QUALITY", "1.0")
            spot_elem.set("NAME", f"spot_{spot_id}")
            spot_elem.set("FRAME", str(entry["t"]))
            spot_elem.set("POSITION_X", str(entry["x"]))
            spot_elem.set("POSITION_Y", str(entry["y"]))
            if has_z:
                spot_elem.set("POSITION_Z", str(entry["z"]))

    # Create tracks using lineage
    for track_id, group in tracks_df.groupby("track_id"):
        track_elem = ET.SubElement(tracks_elem, "Track")
        track_elem.set("TRACK_ID", str(track_id))
        track_elem.set("name", f"Track_{track_id}")

        for spot_id, entry in group.iterrows():
            parent_id = entry["parent_id"]
            if parent_id == NO_PARENT:
                continue
            edge_elem = ET.SubElement(track_elem, "Edge")
            edge_elem.set("SPOT_SOURCE_ID", str(spot_id))
            edge_elem.set("SPOT_TARGET_ID", str(parent_id))

    # Convert to XML string
    xml_str = ET.tostring(root, encoding="unicode")
    xml_str = minidom.parseString(xml_str).toprettyxml()

    return xml_str


def to_trackmate(
    config: MainConfig,
    output_path: Optional[Union[Path, str]] = None,
    overwrite: bool = False,
) -> str:
    """
    Exports tracking results to TrackMate XML format.

    Parameters
    ----------
    config : MainConfig
        ULTrack configuration parameters.
    output_path : Optional[Path], optional
        Output file path, by default None
    overwrite : bool, optional
        Whether to overwrite the output file if it already exists, by default False

    Returns
    -------
    str
        A string representation of the XML in the TrackMate format.
    """
    tracks_df, _ = to_tracks_layer(config)
    xml_str = tracks_layer_to_trackmate(tracks_df)

    # Save to file if output_path is provided
    if output_path is not None:
        if isinstance(output_path, str):
            output_path = Path(output_path)

        if output_path.exists() and not overwrite:
            raise FileExistsError(
                f"File {output_path} already exists. Set overwrite=True to overwrite."
            )

        with output_path.open("w") as f:
            f.write(xml_str)

    return xml_str
