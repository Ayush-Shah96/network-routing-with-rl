from __future__ import annotations
import random
import networkx as nx


def shortest_path(graph, cost_fn, source, destination):
    weight=lambda u,v,d: cost_fn(u,v)
    return nx.shortest_path(graph, source=source, target=destination, weight=weight)


def random_path(graph, source, destination, max_steps=25, seed=7):
    rng=random.Random(seed); cur=source; route=[cur]; visited={cur}
    for _ in range(max_steps):
        if cur==destination:return route
        nbrs=[n for n in graph.neighbors(cur) if n not in visited] or list(graph.neighbors(cur))
        if not nbrs:break
        cur=rng.choice(nbrs); route.append(cur); visited.add(cur)
    return route


def ecmp_path(graph,cost_fn,source,destination,seed=7):
    rng=random.Random(seed); cur=source; route=[cur]
    weight=lambda u,v,d: cost_fn(u,v)
    for _ in range(len(graph.nodes)+3):
        if cur==destination:return route
        scored=[]
        for n in graph.neighbors(cur):
            try:
                d=nx.shortest_path_length(graph,n,destination,weight=weight)+cost_fn(cur,n)
            except nx.NetworkXNoPath:
                continue
            scored.append((round(d,1),n))
        if not scored: break
        best=min(x[0] for x in scored); choices=[n for d,n in scored if d<=best+2]
        cur=rng.choice(choices); route.append(cur)
    return route
