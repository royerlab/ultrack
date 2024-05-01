from typing import List

import napari
from sqlalchemy import create_engine, func
from sqlalchemy.orm import Session

from ultrack.core.database import (
    NodeDB,
    NodeSegmAnnotation,
    get_node_values,
    set_node_values,
)
from ultrack.core.segmentation.node import Node
from ultrack.widgets._generic_annotation_widget import GenericAnnotationWidget


class NodeAnnotationWidget(GenericAnnotationWidget):
    def __init__(self, viewer: napari.Viewer) -> None:
        # before init due to config initialization
        super().__init__(
            viewer, "Node Annotation", NodeSegmAnnotation, " ~ node annot."
        )

    def _query_samples(self) -> List[Node]:
        engine = create_engine(self.config.database_path)
        with Session(engine) as session:
            nodes = (
                session.query(NodeDB.pickle)
                .where(NodeDB.id.not_in([node.id for node in self._nodes]))
                .order_by(func.random())
                .limit(self._sample_size)
                .all()
            )
        return [node for node, in nodes]

    def get_annotation(self, index: int) -> NodeSegmAnnotation:
        return get_node_values(self.config, index, NodeDB.segm_annotation)

    def set_annotation(self, index: int, annot: NodeSegmAnnotation) -> None:
        set_node_values(self.config, index, annotation=annot)
