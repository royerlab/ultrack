from typing import List, Optional, Tuple

import numpy as np
from numpy.typing import ArrayLike
from sqlalchemy import create_engine, func
from sqlalchemy.orm import Session

from ultrack.config.config import LinkingConfig, MainConfig
from ultrack.core.database import LinkDB, NodeDB, OverlapDB
from ultrack.core.segmentation.node import Node


def _nearest_neighbors(
    data_arr: ArrayLike,
    node: Node,
    n_neighbors: int,
    max_distance: Optional[float],
    scale: Optional[ArrayLike],
) -> np.ndarray:
    """
    Returns the indices of the `n_neighbors` nearest neighbors to `node` in `data_arr`.

    Parameters
    ----------
    data_arr : ArrayLike
        Array of (id, z, y, x) coordinates of the nodes (or centroids).
    node : Node
        Node to find neighbors for.
    n_neighbors : int
        Number of neighbors to be considered.
    max_distance : float
        Maximum distance to be considered.
    scale : Optional[ArrayLike], optional
        Scaling factor for the distance, by default None.

    Returns
    -------
    np.ndarray
        Indices of the nearest neighbors.
    """
    if scale is None:
        scale = np.ones(len(node.centroid))

    differences = data_arr[:, -len(node.centroid) :] - node.centroid
    differences *= scale
    sqdist = np.square(differences).sum(axis=1)

    if max_distance is not None:
        valid = sqdist <= (max_distance * max_distance)
        data_arr = data_arr[valid]
        sqdist = sqdist[valid]

    indices = np.argsort(sqdist)

    return data_arr[indices[:n_neighbors], 0]


def _find_links(
    session: Session,
    node: Node,
    adj_time: int,
    scale: Optional[ArrayLike],
    link_config: LinkingConfig,
) -> List[Tuple[NodeDB, float]]:
    """
    Finds links for a given node and time.

    Parameters
    ----------
    session : Session
        SQLAlchemy session.
    node : Node
        Node to search for neighbors.
    adj_time : int
        Adjacent time point.
    scale : Optional[ArrayLike], optional
        Scaling factor for the distance, by default None.
    link_config : LinkingConfig
        Linking configuration parameters.

    Returns
    -------
    List[NodeDB, float]
        List of nodes and their weights.
    """
    data = np.asarray(
        session.query(
            NodeDB.id,
            NodeDB.z,
            NodeDB.y,
            NodeDB.x,
        )
        .where(NodeDB.t == adj_time)
        .all()
    )
    if len(data) == 0:
        return []

    ind = _nearest_neighbors(
        data,
        node,
        2 * link_config.max_neighbors,
        link_config.max_distance,
        scale=scale,
    )

    neigh_nodes = session.query(NodeDB.pickle).where(NodeDB.id.in_(ind)).all()

    if scale is None:
        scale = np.ones(len(node.centroid))

    neigh_nodes_with_dist = []
    for (n,) in neigh_nodes:
        dist = np.linalg.norm((n.centroid - node.centroid) * scale)
        w = node.IoU(n) - link_config.distance_weight * dist
        neigh_nodes_with_dist.append((n, w))

    neigh_nodes_with_dist.sort(key=lambda x: x[1], reverse=True)

    return neigh_nodes_with_dist[: link_config.max_neighbors]


def _add_overlaps(
    session: Session,
    node: Node,
    n_neighbors: int = 10,
    scale: Optional[ArrayLike] = None,
) -> None:
    """
    Adds overlaps to the database.

    Parameters
    ----------
    session : Session
        SQLAlchemy session.
    node : Node
        Node to find overlaps with.
    n_neighbors : int, optional
        Number of neighbors to be considered, by default 10.
    scale : Optional[ArrayLike], optional
        Scaling factor for the distance, by default None.
    """
    data = np.asarray(
        session.query(
            NodeDB.id,
            NodeDB.z,
            NodeDB.y,
            NodeDB.x,
        )
        .where(NodeDB.t == node.time, NodeDB.id != node.id)
        .all()
    )
    ind = _nearest_neighbors(data, node, n_neighbors, max_distance=None, scale=scale)

    overlaps = []

    for (neigh_node,) in session.query(NodeDB.pickle).where(NodeDB.id.in_(ind)).all():
        if node.IoU(neigh_node) > 0.0:
            overlaps.append(
                OverlapDB(
                    node_id=node.id,
                    ancestor_id=neigh_node.id,
                )
            )

    session.add_all(overlaps)


def _add_links(
    session: Session,
    node: Node,
    link_config: LinkingConfig,
    scale: Optional[ArrayLike] = None,
) -> None:
    """
    Adds T - 1 and T + 1 links to the database.

    NOTE: this is not taking node shifts into account.

    Parameters
    ----------
    session : Session
        SQLAlchemy session.
    node : Node
        Node to search for neighbors.
    link_config : LinkingConfig
        Linking configuration parameters.
    scale : Optional[ArrayLike], optional
        Scaling factor for the distance, by default None.
    """
    links = []

    before_links = _find_links(
        session=session,
        node=node,
        adj_time=node.time - 1,
        scale=scale,
        link_config=link_config,
    )
    for before_node, w in before_links:
        links.append(
            LinkDB(
                source_id=before_node.id,
                target_id=node.id,
                weight=w,
            )
        )

    after_links = _find_links(
        session=session,
        node=node,
        adj_time=node.time + 1,
        scale=scale,
        link_config=link_config,
    )
    for after_node, w in after_links:
        links.append(
            LinkDB(
                source_id=node.id,
                target_id=after_node.id,
                weight=w,
            )
        )

    session.add_all(links)


def add_new_node(
    config: MainConfig,
    time: int,
    mask: ArrayLike,
    bbox: Optional[ArrayLike] = None,
    index: Optional[int] = None,
    include_overlaps: bool = True,
) -> int:
    """
    Adds a new node to the database.

    NOTE: this is not taking node shifts or image features (color) into account.

    Parameters
    ----------
    config : MainConfig
        Ultrack configuration parameters.
    time : int
        Time point of the node.
    mask : ArrayLike
        Binary mask of the node.
    bbox : Optional[ArrayLike], optional
        Bounding box of the node, (min_0, min_1, ..., max_0, max_1, ...).
        When provided it assumes the mask is a crop of the original image, by default None
    index : Optional[int], optional
        Node index, otherwise it is automatically generated, and returned.
    include_overlaps : bool, optional
        Include overlaps in the database, by default True
        When False it will allow oclusions between new node and existing nodes.

    Returns
    -------
    int
        New node index.
    """

    node = Node.from_mask(
        time=time,
        mask=mask,
        bbox=np.asarray(bbox),
    )
    if node.area == 0:
        raise ValueError("Node area is zero. Something went wrong.")

    scale = config.data_config.metadata.get("scale")

    engine = create_engine(config.data_config.database_path)
    with Session(engine) as session:

        # querying required data
        if index is None:
            node.id = (
                int(session.query(func.max(NodeDB.id)).where(NodeDB.t == time).scalar())
                + 1
            )
        else:
            node.id = index

        # adding node
        if len(node.centroid) == 2:
            y, x = node.centroid
            z = 0
        else:
            z, y, x = node.centroid

        node_db_obj = NodeDB(
            id=node.id,
            t=node.time,
            z=z,
            y=y,
            x=x,
            area=node.area,
            pickle=node,
        )
        session.add(node_db_obj)

        if include_overlaps:
            _add_overlaps(session=session, node=node, scale=scale)

        _add_links(
            session=session,
            node=node,
            link_config=config.linking_config,
            scale=scale,
        )

        session.commit()

    return node.id
