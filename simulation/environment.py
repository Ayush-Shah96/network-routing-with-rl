from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import random
import numpy as np
import networkx as nx

from simulation.topology import build_topology


@dataclass
class LinkState:
    latency_ms: float
    bandwidth_mbps: float
    congestion: float
    packet_loss: float
    availability: float = 1.0
    queue_ms: float = 0.0
    utilization: float = 0.0

    def as_dict(self):
        return {k: float(getattr(self, k)) for k in ("latency_ms","bandwidth_mbps","congestion","packet_loss","availability","queue_ms","utilization")}


class DynamicRoutingEnv:
    """Dynamic multi-condition routing simulator with faults, traffic and explainable state."""
    MAX_STEPS = 20
    FEATURES = 7

    def __init__(self, seed: int | None = None, topology: str = "mesh", nodes: int = 15):
        self.rng = random.Random(seed); self.seed = seed
        self.graph = build_topology(topology, nodes, seed or 7)
        self.NUM_NODES = nodes; self.STATE_SIZE = 2 + nodes * self.FEATURES
        self.links: dict[tuple[int,int], LinkState] = {}
        self.current = 0; self.destination = nodes - 1; self.steps = 0; self.visited=[]
        self.events: list[dict[str,Any]] = []; self.traffic_load = 0.35

    def _key(self,a,b): return (a,b) if a<b else (b,a)
    def get_link(self,u,v): return self.links.get(self._key(u,v))

    def reset(self, source=None, destination=None, scenario: str = "normal"):
        self._randomize_links(); self.current = self.rng.randrange(self.NUM_NODES) if source is None else source
        self.destination = self.rng.randrange(self.NUM_NODES) if destination is None else destination
        while self.destination == self.current: self.destination = self.rng.randrange(self.NUM_NODES)
        self.steps=0; self.visited=[self.current]; self.events=[]
        self.apply_scenario(scenario)
        return self._state()

    def _randomize_links(self):
        self.links.clear()
        for u,v in self.graph.edges():
            self.links[self._key(u,v)] = LinkState(
                latency_ms=self.rng.uniform(6,45), bandwidth_mbps=self.rng.uniform(50,1000),
                congestion=self.rng.uniform(.02,.28), packet_loss=self.rng.uniform(0,.025), availability=1.0,
                queue_ms=self.rng.uniform(0,4), utilization=self.rng.uniform(.1,.4))

    def apply_scenario(self, scenario: str):
        if scenario == "rush_hour":
            for l in self.links.values(): l.congestion=min(1,l.congestion+self.rng.uniform(.25,.45)); l.utilization=min(1,l.utilization+.3); l.queue_ms += self.rng.uniform(5,25)
        elif scenario == "degraded":
            for l in self.links.values(): l.packet_loss=min(.3,l.packet_loss+.04); l.latency_ms*=1.6
        elif scenario == "failure":
            edges=list(self.links); self.set_link_failure(*edges[self.rng.randrange(len(edges))], down=True)
        elif scenario == "storm":
            for _ in range(max(1,len(self.links)//5)):
                e=self.rng.choice(list(self.links)); self.set_link_failure(*e, down=True)

    def set_link_failure(self,u,v,down=True):
        l=self.get_link(u,v)
        if l: l.availability = 0.0 if down else 1.0; l.packet_loss = .30 if down else min(l.packet_loss,.02)
        self.events.append({"event":"link_down" if down else "link_up","u":u,"v":v})

    def _state(self):
        values=[self.current/max(1,self.NUM_NODES-1), self.destination/max(1,self.NUM_NODES-1)]
        for n in range(self.NUM_NODES):
            if self.graph.has_edge(self.current,n):
                l=self.get_link(self.current,n)
                values.extend([l.latency_ms/200.0,l.bandwidth_mbps/1000.0,l.congestion,l.packet_loss,l.availability,l.queue_ms/50.0,l.utilization])
            else: values.extend([1.0,0.0,1.0,1.0,0.0,1.0,1.0])
        return np.asarray(values,dtype=np.float32)

    def valid_actions(self):
        return [n for n in self.graph.neighbors(self.current) if (self.get_link(self.current,n) and self.get_link(self.current,n).availability > 0)]

    def edge_cost(self,u,v):
        l=self.get_link(u,v)
        if l is None or l.availability<=0: return float('inf')
        return l.latency_ms + l.queue_ms + l.congestion*100 + l.packet_loss*300 + (1-l.availability)*10000

    def step(self, action):
        self.steps += 1; info={"valid":False,"event":""}
        if action not in self.valid_actions():
            return self._state(), -50.0, self.steps>=self.MAX_STEPS, {"valid":False,"event":"invalid_action"}
        link=self.get_link(self.current,action)
        loop=action in self.visited
        cost=link.latency_ms/8 + link.queue_ms/5 + link.congestion*9 + link.packet_loss*100 + (1-link.availability)*100
        reward=-cost-(18 if loop else 0)
        self.current=action
        if not loop:self.visited.append(action)
        info={"valid":True,"event":"forwarded","link":link.as_dict(),"estimated_cost":self.edge_cost(self.visited[-2],action) if len(self.visited)>1 else None}
        if self.current==self.destination:
            reward += 120; done=True; info["event"]="delivered"
        else: done=self.steps>=self.MAX_STEPS; reward += -35 if done else 0; info["event"]="timeout" if done else "forwarded"
        # Traffic drift: utilization and congestion evolve every hop.
        for l in self.links.values():
            l.utilization=float(np.clip(l.utilization+self.rng.uniform(-.06,.10)*self.traffic_load,0,1)); l.congestion=float(np.clip(.6*l.congestion+.4*l.utilization,0,1)); l.queue_ms=float(np.clip(l.queue_ms+self.rng.uniform(-2,5)*l.congestion,0,80)); l.latency_ms=float(np.clip(l.latency_ms+self.rng.uniform(-3,6)*l.congestion,3,220)); l.packet_loss=float(np.clip(l.packet_loss+self.rng.uniform(-.005,.012)*l.congestion,0,.3))
        return self._state(), float(reward), done, info

    def inject_congestion(self, route=None, factor=3.2):
        edges=list(zip(route,route[1:])) if route and len(route)>1 else self.rng.sample(list(self.graph.edges()), min(3,self.graph.number_of_edges()))
        for u,v in edges:
            l=self.get_link(u,v)
            if l: l.congestion=min(1,l.congestion*factor+.2); l.utilization=min(1,l.utilization+.4); l.queue_ms=min(80,l.queue_ms*factor+10); l.latency_ms=min(220,l.latency_ms*factor); l.packet_loss=min(.3,l.packet_loss*1.8+.03)
            self.events.append({"event":"congestion_injected","u":u,"v":v})

    def snapshot(self):
        return {"source":self.current,"destination":self.destination,"edges":[{"u":u,"v":v,**self.get_link(u,v).as_dict(),"cost":self.edge_cost(u,v)} for u,v in self.graph.edges()],"visited":list(self.visited),"events":list(self.events)}
