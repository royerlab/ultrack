# Configuration Schema Description

Configurations have default values, therefore you don't need to set them all from the beginning.

 - main_config:

    - data_config:
        - working_dir: Working directory for auxiliary files;
        - database: Database type `sqlite` and `postgresql` supported;
        - address: Postgresql database path, for example, `postgres@localhost:12345/example`;
        - n_workers: Number of worker threads;

    - segmentation_config:
        - anisotropy_penalization: Image graph z-axis penalization, positive values will prioritize segmenting the xy-plane first, negative will do the opposite;
        - n_workers: Number of worker threads;
        - min_area: Minimum segments area, regions smaller than this value are merged or removed when there is no neighboring region;
        - max_area: Maximum segments area, regions larger than this value are removed;
        - min_frontier: Minimum average contour value, neighboring regions with values below this are merged;
        - max_noise: Upper limit of uniform distribution for additive noise on contour map;
        - threshold: Threshold used to binary the cell detection map;
        - ws_hierarchy: Watershed hierarchy function from [higra](https://higra.readthedocs.io/en/stable/python/watershed_hierarchy.html) used to construct the hierarchy;

    - linking_config:
        - distance_weight: Penalization weight $\gamma$ for distance between segment centroids, $w_{pq} - \gamma \|c_p - c_q\|_2$, where $c_p$ is region $p$ center of mass;
        - n_workers: Number of worker threads;
        - max_neighbors: Maximum number of neighbors per candidate segment;
        - max_distance: Maximum distance between neighboring segments;

    - tracking_config:
        - appear_weight: Penalization weight for appearing cell, should be negative;
        - disappear_weight: Penalization for disappearing cell, should be negative;
        - division_weight: Penalization for dividing cell, should be negative;
        - dismiss_weight_guess: Threshold (<=) used to provide 0 valued hint to solver;
        - include_weight_guess: Threshold (>=) used to provide 1 valued hint to solver;
        - window_size: Time window size for partially solving the tracking ILP;
        - overlap_size: Number of frames used to pad each size when partially solving the tracking ILP;
        - solution_gap: solver solution gap;
        - time_limit: solver execution time limit in seconds;
        - method: solver method, (reference)[https://docs.python-mip.com/en/latest/classes.html#lp-method];
        - n_threads: Number of worker threads;
        - link_function: Function used to transform the edge weights, `identity` or `power`;
        - power: Expoent $\eta$ of power transform, $w_{pq}^\eta$;
        - bias: Edge weights bias $b$, $w_{pq} + b$, should be negative;
