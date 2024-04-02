from typing import List

import napari
from sqlalchemy import create_engine, func
from sqlalchemy.orm import Session

from ultrack.core.database import (
    NO_PARENT,
    NodeDB,
    VarAnnotation,
    get_node_values,
    set_node_values,
)
from ultrack.core.segmentation.node import Node
from ultrack.widgets._generic_annotation_widget import GenericAnnotationWidget


class DivisionAnnotationWidget(GenericAnnotationWidget):
    def __init__(self, viewer: napari.Viewer) -> None:
        # before init due to config initialization
        super().__init__(
            viewer, "Division Annotation", VarAnnotation, " ~ division annot."
        )

    def _query_samples(self) -> List[Node]:
        engine = create_engine(self.config.database_path)
        with Session(engine) as session:
            parent_ids = (
                session.query(NodeDB.parent_id)
                .where(
                    NodeDB.parent_id != NO_PARENT,
                    NodeDB.parent_id.not_in([node.id for node in self._nodes]),
                )
                .group_by(NodeDB.parent_id)
                .having(func.count(NodeDB.id) > 1)
                .order_by(func.random())
                .limit(self._sample_size)
            )
            nodes = session.query(NodeDB.pickle).where(NodeDB.id.in_(parent_ids))

        return [node for node, in nodes]

    def get_annotation(self, index: int) -> VarAnnotation:
        return get_node_values(self.config, index, NodeDB.division)

    def set_annotation(self, index: int, annot: VarAnnotation) -> None:
        set_node_values(self.config, index, division=annot)
