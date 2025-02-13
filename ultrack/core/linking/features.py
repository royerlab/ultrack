import numpy as np
import pandas as pd
import sqlalchemy as sqla
from sqlalchemy.orm import Session

from ultrack.config import MainConfig
from ultrack.core.database import LinkDB
from ultrack.core.segmentation.processing import get_nodes_features


def get_links_features(
    config: MainConfig,
    **kwargs,
) -> pd.DataFrame:
    """
    Creates a pandas dataframe from links features defined during linking.

    Parameters
    ----------
    config : MainConfig
        Configuration parameters.
    **kwargs : dict
        Keyword arguments passed to `get_nodes_features`.

    Returns
    -------
    pd.DataFrame:
        Links dataframe with the features.
        MultiIndex columns:
        * target_id: target node id
        * source_id: source node id
    """
    nodes_features = get_nodes_features(config, **kwargs)

    engine = sqla.create_engine(config.data_config.database_path)

    with Session(engine) as session:
        query = session.query(
            LinkDB.source_id,
            LinkDB.target_id,
        )
        links_df = pd.read_sql(query.statement, session.bind)

    source_df = nodes_features.loc[links_df["source_id"]]
    target_df = nodes_features.loc[links_df["target_id"]]

    if "x" in source_df.columns:
        # compute distance between source and target
        links_df["dist"] = np.sqrt(
            np.sum(
                np.square(source_df[c].to_numpy() - target_df[c].to_numpy())
                for c in ["x", "y", "z"]
            )
        )

    cols = source_df.columns.drop(["id", "t", "z", "y", "x"], errors="ignore")

    links_df[cols] = source_df[cols].to_numpy() - target_df[cols].to_numpy()

    links_df.set_index(["target_id", "source_id"], inplace=True)

    return links_df
