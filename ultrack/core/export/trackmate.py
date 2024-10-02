import xml.dom.minidom as minidom
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional, Union

import pandas as pd

from ultrack.config.config import MainConfig
from ultrack.core.export.tracks_layer import to_tracks_layer
from ultrack.utils.constants import NO_PARENT


def _set_filter_elem(elem: ET.Element) -> None:
    elem.set("feature", "QUALITY")
    elem.set("value", "0.0")
    elem.set("isabove", "true")


def tracks_layer_to_trackmate(
    tracks_df: pd.DataFrame,
) -> str:
    """
    Convert a pandas DataFrame representation of Napari track layer to TrackMate XML format.
    `<ImageData/>` need to be set manually in the output XML.

    Parameters
    ----------
    tracks_df : pd.DataFrame
        A DataFrame with columns `track_id, id, parent_id, t, z, y, x`. Cells that belong to the same track
        have the same `track_id`.

    Returns
    -------
    str
        A string representation of the XML in the TrackMate format.

    Examples
    --------
    >>> tracks_df = pd.DataFrame(
    ...     [[1,0,12.0,49.0,49.0,1000001,-1,-1],
    ...     [1,1,12.0,49.0,32.0,2000001,-1,1000001],
    ...     [2,1,12.0,49.0,66.0,2000002,-1,1000001]],
    ...     columns=['track_id','t','z','y','x','id','parent_track_id','parent_id']
    ... )
    >>> print(tracks_df)
       track_id  t     z     y     x       id  parent_track_id  parent_id
    0         1  0  12.0  49.0  49.0  1000001               -1         -1
    1         1  1  12.0  49.0  32.0  2000001               -1    1000001
    2         2  1  12.0  49.0  66.0  2000002               -1    1000001
    >>> tracks_layer_to_trackmate(tracks_df)
    <?xml version="1.0" ?>
    <TrackMate version="7.11.1">
        <Model spatialunits="pixels" timeunits="frames">
            <AllTracks>
                <Track TRACK_ID="1" NUMBER_SPOTS="2" NUMBER_GAPS="0" TRACK_START="0" TRACK_STOP="1" name="Track_1">
                    <Edge SPOT_SOURCE_ID="1000001" SPOT_TARGET_ID="2000001" EDGE_TIME="0.5"/>
                </Track>
                <Track TRACK_ID="2" NUMBER_SPOTS="1" NUMBER_GAPS="0" TRACK_START="1" TRACK_STOP="1" name="Track_2">
                    <Edge SPOT_SOURCE_ID="1000001" SPOT_TARGET_ID="2000002" EDGE_TIME="0.5"/>
                </Track>
            </AllTracks>
            <FilteredTracks>
                <TrackID TRACK_ID="1"/>
                <TrackID TRACK_ID="2"/>
            </FilteredTracks>
            <AllSpots>
                <SpotsInFrame frame="0">
                    <Spot ID="1000001" QUALITY="1.0" VISIBILITY="1" name="1000001" FRAME="0" RADIUS="5.0" POSITION_X="49.0" POSITION_Y="49.0" POSITION_Z="12.0"/>
                </SpotsInFrame>
                <SpotsInFrame frame="1">
                    <Spot ID="2000001" QUALITY="1.0" VISIBILITY="1" name="2000001" FRAME="1" RADIUS="5.0" POSITION_X="32.0" POSITION_Y="49.0" POSITION_Z="12.0"/>
                    <Spot ID="2000002" QUALITY="1.0" VISIBILITY="1" name="2000002" FRAME="1" RADIUS="5.0" POSITION_X="66.0" POSITION_Y="49.0" POSITION_Z="12.0"/>
                </SpotsInFrame>
            </AllSpots>
            <FeatureDeclarations>
                ...
            </FeatureDeclarations>
        </Model>
        <Settings>
            <InitialSpotFilter feature="QUALITY" value="0.0" isabove="true"/>
            <SpotFilterCollection/>
            <TrackFilterCollection/>
            <ImageData filename="None" folder="None" width="0" height="0" depth="0" nslices="1" nframes="2" pixelwidth="1.0" pixelheight="1.0" voxeldepth="1.0" timeinterval="1.0"/>
        </Settings>
    </TrackMate>
    """  # noqa: E501
    tracks_df["id"] = tracks_df["id"].astype(int)
    if not tracks_df["id"].is_unique:
        raise ValueError("The 'id' column must be unique.")
    tracks_df.set_index("id", inplace=True)
    tracks_df["parent_id"] = tracks_df["parent_id"].astype(int)
    tracks_df["track_id"] = tracks_df["track_id"].astype(int)

    # Create XML root and child elements
    root = ET.Element("TrackMate")
    root.set("version", "7.11.1")  # required by TrackMate, not significant

    model_elem = ET.SubElement(root, "Model")
    model_elem.set("spatialunits", "pixels")
    model_elem.set("timeunits", "frames")

    all_tracks_elem = ET.SubElement(model_elem, "AllTracks")
    filtered_tracks_elem = ET.SubElement(model_elem, "FilteredTracks")
    all_spots_elem = ET.SubElement(model_elem, "AllSpots")
    features_elem = ET.SubElement(model_elem, "FeatureDeclarations")

    settings_elem = ET.SubElement(root, "Settings")

    _set_filter_elem(ET.SubElement(settings_elem, "InitialSpotFilter"))
    ET.SubElement(settings_elem, "SpotFilterCollection")
    ET.SubElement(settings_elem, "TrackFilterCollection")

    image_elem = ET.SubElement(settings_elem, "ImageData")
    image_elem.set("filename", "None")
    image_elem.set("folder", "None")
    image_elem.set("width", "0")
    image_elem.set("height", "0")
    image_elem.set("depth", "0")
    image_elem.set("nslices", "1")
    image_elem.set("nframes", str(tracks_df["t"].max() + 1))
    image_elem.set("pixelwidth", "1.0")
    image_elem.set("pixelheight", "1.0")
    image_elem.set("voxeldepth", "1.0")
    image_elem.set("timeinterval", "1.0")

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
        elem = ET.SubElement(spot_features_elem, "Feature")
        elem.set("feature", feature)
        elem.set("name", name)
        elem.set("shortname", shortname)
        elem.set("dimension", dimension)
        elem.set("isint", isint)

    # Create edge features
    edge_features_elem = ET.SubElement(features_elem, "EdgeFeatures")
    edge_features = [
        ("SPOT_SOURCE_ID", "Source spot ID", "Source ID", "NONE", "true"),
        ("SPOT_TARGET_ID", "Target spot ID", "Target ID", "NONE", "true"),
        # ... add other edge features if needed
    ]
    for feature, name, shortname, dimension, isint in edge_features:
        elem = ET.SubElement(edge_features_elem, "Feature")
        elem.set("feature", feature)
        elem.set("name", name)
        elem.set("shortname", shortname)
        elem.set("dimension", dimension)
        elem.set("isint", isint)

    track_features_elem = ET.SubElement(features_elem, "TrackFeatures")
    track_features = [
        ("NUMBER_SPOTS", "Number of spots in track", "N spots", "NONE", "true"),
        ("NUMBER_GAPS", "Number of gaps", "Gaps", "NONE", "true"),
        ("LONGEST_GAP", "Longest gap", "Longest gap", "NONE", "true"),
        ("NUMBER_SPLITS", "Number of split events", "Splits", "NONE", "true"),
        ("NUMBER_MERGES", "Number of merge events", "Merges", "NONE", "true"),
        ("NUMBER_COMPLEX", "Complex points", "Complex", "NONE", "true"),
        ("TRACK_DURATION", "Duration of track", "Duration", "TIME", "false"),
        ("TRACK_START", "Track start", "T start", "TIME", "false"),
        ("TRACK_STOP", "Track stop", "T stop", "TIME", "false"),
        ("TRACK_DISPLACEMENT", "Track displacement", "Displacement", "LENGTH", "false"),
        ("TRACK_INDEX", "Track index", "Index", "NONE", "true"),
        ("TRACK_ID", "Track ID", "ID", "NONE", "true"),
    ]

    for feature, name, shortname, dimension, isint in track_features:
        elem = ET.SubElement(track_features_elem, "Feature")
        elem.set("feature", feature)
        elem.set("name", name)
        elem.set("shortname", shortname)
        elem.set("dimension", dimension)
        elem.set("isint", isint)

    # Create spots
    for frame, group in tracks_df.groupby("t"):
        frame_elem = ET.SubElement(all_spots_elem, "SpotsInFrame")
        frame_elem.set("frame", str(frame))
        for spot_id, entry in group.iterrows():
            spot_elem = ET.SubElement(frame_elem, "Spot")
            spot_elem.set("ID", str(spot_id))
            spot_elem.set("QUALITY", "1.0")
            spot_elem.set("VISIBILITY", "1")
            spot_elem.set("name", str(spot_id))
            spot_elem.set("FRAME", str(int(entry["t"])))
            spot_elem.set("RADIUS", "5.0")
            spot_elem.set("POSITION_X", str(entry["x"]))
            spot_elem.set("POSITION_Y", str(entry["y"]))
            if has_z:
                spot_elem.set("POSITION_Z", str(entry["z"]))
            else:
                spot_elem.set("POSITION_Z", "0.0")

    # Create tracks using lineage
    for track_id, group in tracks_df.groupby("track_id"):
        track_elem = ET.SubElement(all_tracks_elem, "Track")
        track_elem.set("TRACK_ID", str(track_id))
        track_elem.set("NUMBER_SPOTS", str(len(group)))
        track_elem.set("NUMBER_GAPS", "0")
        track_elem.set("TRACK_START", str(group["t"].min()))
        track_elem.set("TRACK_STOP", str(group["t"].max()))
        track_elem.set("name", f"Track_{track_id}")

        ET.SubElement(filtered_tracks_elem, "TrackID").set("TRACK_ID", str(track_id))

        for spot_id, entry in group.iterrows():  # spot_id is the DataFrame row index
            parent_id = int(entry["parent_id"])
            if parent_id == NO_PARENT:
                continue
            edge_elem = ET.SubElement(track_elem, "Edge")
            edge_elem.set("SPOT_SOURCE_ID", str(parent_id))
            edge_elem.set("SPOT_TARGET_ID", str(spot_id))
            edge_elem.set("EDGE_TIME", str(entry["t"] - 0.5))

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
