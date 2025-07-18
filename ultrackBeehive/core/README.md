# Database Tables

Ultrack has three SQL tables used to store the tracking data, `nodes` is the main table, it stores the cell information and the trajectories by saving the cells' parent, `overlaps` stores the overlap constraints of the hierarchies, and `links` the association between nodes.

Their columns are described below:

#### nodes

- t: Node time point;
- id: Node id;
- parent_id: Node id of parent from t - 1, added after tracking;
- t_node_id: Node id at the time point, for internal use;
- t_hier_id: Hierarchy id of node at the time point, for internal use;
- z: z coordinate of cell centroid;
- y: y coordinate of cell centroid;
- x: x coordinate of cell centroid;
- z_shift: estimated z displacement of cell centroid;
- y_shift: estimated y displacement of cell centroid;
- x_shift: estimated x displacement of cell centroid;
- area: segment size;
- selected: boolean indicating if the node is in solution;
- pickle: pickled data structure of the segment;
- annotation: segmentation curation label (UNKOWN, CORRECT, UNDERSEGMENTED & OVERSEGMENTED);
- division: division curation label (UNKOWN, TRUE, FALSE);


#### overlaps

- id: Overlap id;
- node_id: Node id, matches to `node.id`;
- ancestor_id: Ancestor of `overlaps.node_id` node, matches to `node.id`;


#### links

- id: Link id;
- source_id: Source node from t, matches to `node.id`;
- target_id: Target node from t + 1, matchs to `node.id`;
- weight: Link association score, the higher the better.
