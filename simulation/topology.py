from __future__ import annotations

import networkx as nx

NODE_NAMES = list("ABCDEFGHIJKLMNO")


def build_topology(kind: str = "mesh", nodes: int = 15, seed: int = 7) -> nx.Graph:
    if kind == "grid":
        side = int(round(nodes ** 0.5)); g = nx.grid_2d_graph(side, side)
        mapping = {n: i for i, n in enumerate(g.nodes())}; g = nx.relabel_nodes(g, mapping)
        return g
    if kind == "scale_free":
        return nx.barabasi_albert_graph(nodes, max(1, min(3, nodes - 1)), seed=seed)
    if kind == "random":
        return nx.gnp_random_graph(nodes, min(0.45, 4 / max(5, nodes)), seed=seed)
    g = nx.Graph()
    g.add_nodes_from(range(nodes))
    for i in range(nodes - 1):
        g.add_edge(i, i + 1)
    extra = [(0,2),(0,4),(1,3),(1,5),(2,5),(2,6),(3,6),(3,7),(4,6),(4,8),(5,8),(5,9),(6,9),(7,10),(8,10),(8,11),(9,11),(9,12),(10,13),(11,13),(11,14),(12,14)]
    for e in extra:
        if max(e) < nodes: g.add_edge(*e)
    return g


def node_name(node: int) -> str:
    return NODE_NAMES[node] if node < len(NODE_NAMES) else f"N{node}"


def route_names(route: list[int]) -> list[str]:
    return [node_name(x) for x in route]
