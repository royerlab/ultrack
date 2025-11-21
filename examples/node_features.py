"""
This example how request for each segmentation hypotheses features and use them
to compute a custom edge weight between nodes, in this case, the cosine distance.

For this we consider a 3D image as a 2D video, it's more a didactic example than a real use case.
"""

import napari
import numpy as np
from scipy.spatial.distance import cdist
from skimage import morphology as morph
from skimage.data import cells3d

from ultrack import MainConfig, Tracker


def main() -> None:

    config = MainConfig()
    config.data_config.working_dir = "/tmp/ultrack/."

    # removing small segments
    config.segmentation_config.min_area = 1_000
    # disable division
    config.tracking_config.division_weight = -1_000_000

    tracker = Tracker(config)

    # mocking a 3D image as 2D video
    image = cells3d()[:, 1]  # nuclei

    # simple foreground extraction
    foreground = image > image.mean()
    foreground = morph.opening(foreground, morph.disk(3)[None, :])
    foreground = morph.closing(foreground, morph.disk(3)[None, :])

    # contour as inverse of the image
    contour = 1 - image / image.max()

    tracker.segment(
        foreground=foreground,
        contours=contour,
        image=image,
        properties=["equivalent_diameter_area", "intensity_mean", "inertia_tensor"],
        overwrite=True,
    )

    df = tracker.get_nodes_features()

    # extending properties, some include -0-0, -0-1, -1-0, -1-1
    cols = [
        "y",
        "x",
        "area",
        "intensity_mean",
        "inertia_tensor-0-0",
        "inertia_tensor-0-1",
        "inertia_tensor-1-0",
        "inertia_tensor-1-1",
        "equivalent_diameter_area",
    ]

    # normalizing features
    df[cols] -= df[cols].mean()
    df[cols] /= df[cols].std()

    df_by_t = df.groupby("t")
    t_max = df["t"].max()

    # iterating over time and querying pair of frames
    for t in range(t_max + 1):
        try:
            # some frames might be without nodes
            source_df = df_by_t.get_group(t)
            target_df = df_by_t.get_group(t + 1)
        except KeyError:
            continue

        # the higher the weights the more likely the link
        weights = 1 - cdist(source_df[cols], target_df[cols], "cosine").ravel()

        source_ids = np.repeat(source_df.index.to_numpy(), len(target_df))
        target_ids = np.tile(target_df.index.to_numpy(), len(source_df))

        # for very dense graph this not recommended because the ILP problem will be huge
        tracker.add_links(sources=source_ids, targets=target_ids, weights=weights)

    tracker.solve()

    tracks, graph = tracker.to_tracks_layer()
    segments = tracker.to_zarr()

    viewer = napari.Viewer()
    viewer.add_image(image, name="cells")
    viewer.add_tracks(tracks[["track_id", "t", "y", "x"]], name="tracks", graph=graph)
    viewer.add_labels(segments, name="segments")

    napari.run()


if __name__ == "__main__":
    main()
