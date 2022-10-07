import time
from typing import Tuple

import gurobipy as gp
import mip
import numpy as np
import pandas as pd
from gurobipy import GRB
from rich import print


def mip_model(n_nodes: int, n_neighbors: int) -> Tuple[float, float]:

    # creating dummy data #
    rng = np.random.default_rng(42)
    n_edges = n_nodes * n_neighbors

    edge_weights = rng.uniform(size=n_nodes * n_neighbors)
    edge_ids = rng.integers(n_nodes, size=(len(edge_weights), 2))

    # removing self links
    self_link = edge_ids[:, 0] == edge_ids[:, 1]
    edge_ids[self_link, 1] = (edge_ids[self_link, 1] + 1) % len(edge_weights)

    # compressing repeated edges
    edge_df = pd.DataFrame(edge_ids, columns=["source", "target"])
    edge_df["weights"] = edge_weights
    edge_df = edge_df.groupby(["source", "target"]).sum()
    n_edges = len(edge_df)

    # building model #
    build_start = time.time()

    model = mip.Model(sense=mip.MAXIMIZE, solver_name="CBC")

    node_vars = model.add_var_tensor((n_nodes,), name="nodes", var_type=mip.BINARY)
    source_vars = model.add_var_tensor((n_nodes,), name="sources", var_type=mip.BINARY)
    aux_vars = model.add_var_tensor((n_nodes,), name="auxs", var_type=mip.BINARY)
    dest_vars = model.add_var_tensor((n_nodes,), name="dests", var_type=mip.BINARY)
    edge_vars = model.add_var_tensor((n_edges,), name="edges", var_type=mip.BINARY)

    edge_target = edge_df.groupby("target")
    edge_source = edge_df.groupby("source")

    for i in range(n_nodes):
        # yes, it's flipped
        try:
            i_sources = edge_target.get_group(i)["source"].values
        except KeyError:
            i_sources = []

        try:
            i_targets = edge_source.get_group(i)["target"].values
        except KeyError:
            i_targets = []

        model.add_constr(
            mip.xsum(edge_vars[i_sources]) + source_vars[i] == node_vars[i]
        )
        model.add_constr(
            node_vars[i] + aux_vars[i] == mip.xsum(edge_vars[i_targets]) + dest_vars[i]
        )
        model.add_constr(node_vars[i] >= aux_vars[i])

    model.objective = (
        mip.xsum(source_vars)
        + mip.xsum(aux_vars)
        + mip.xsum(dest_vars)
        + mip.xsum(edge_vars * edge_df["weights"].values)
    )

    build_end = time.time()

    # solving #
    model.optimize()

    return build_end - build_start, model.objective_value


def gurobi_model(n_nodes: int, n_neighbors: int) -> Tuple[float, float]:
    # creating dummy data #
    rng = np.random.default_rng(42)
    n_edges = n_nodes * n_neighbors

    edge_weights = rng.uniform(size=n_edges)
    edge_ids = rng.integers(n_edges, size=(n_edges, 2))

    # removing self links
    self_link = edge_ids[:, 0] == edge_ids[:, 1]
    edge_ids[self_link, 1] = (edge_ids[self_link, 1] + 1) % n_edges

    node_ids = range(n_nodes)
    edges = {(s, t): w for s, t, w in zip(edge_ids[:, 0], edge_ids[:, 1], edge_weights)}

    # building model #
    build_start = time.time()

    model = gp.Model()

    node_vars = model.addVars(node_ids, vtype=GRB.BINARY)
    source_vars = model.addVars(node_ids, obj=1.0, vtype=GRB.BINARY)
    aux_vars = model.addVars(node_ids, obj=1.0, vtype=GRB.BINARY)
    dest_vars = model.addVars(node_ids, obj=1.0, vtype=GRB.BINARY)
    edge_vars = model.addVars(edges.keys(), obj=edges.values(), vtype=GRB.BINARY)

    model.addConstrs(
        edge_vars.sum("*", i) + source_vars[i] == node_vars[i] for i in node_ids
    )

    model.addConstrs(
        node_vars[i] + aux_vars[i] == edge_vars.sum(i, "*") + dest_vars[i]
        for i in node_ids
    )

    model.addConstrs(node_vars[i] >= aux_vars[i] for i in node_ids)

    build_end = time.time()

    # solving #
    model.ModelSense = GRB.MAXIMIZE
    model.optimize()

    return build_end - build_start, model.getObjective().getValue()


if __name__ == "__main__":
    n_nodes = 1_000_000
    n_neighbors = 5

    gurobi_build_time, gurobi_obj = gurobi_model(n_nodes, n_neighbors)
    mip_build_time, mip_obj = mip_model(n_nodes, n_neighbors)

    print(f"\nGUROBI obj {gurobi_obj}")
    print(f"{gurobi_build_time} secs to build model using GUROBI")

    print(f"\nMIP obj {mip_obj}")
    print(f"{mip_build_time} secs to build model using MIP")
