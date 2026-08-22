# Project Report — Adaptive Network Intelligence

## Objective
Build a local routing decision platform that evaluates whether reinforcement learning can adapt next-hop selection under changing network conditions.

## Learning algorithm
The primary model is a **Double-DQN** with a **Dueling Q-network** and **Prioritized Experience Replay**. The agent sees the current node, destination and per-neighbor link telemetry and predicts an action value for each possible next hop.

### Safety layer
The optional **Safe DQN** policy adds a constrained decision layer. It excludes unavailable links, avoids loops when possible, verifies that the selected neighbor can still reach the destination, and rejects an action when its link cost is clearly pathological relative to another safe action.

This turns raw model inference into a **model + safety-shield** architecture appropriate for experimentation.

## Environment
The simulator models:

- latency
- bandwidth
- congestion / utilization
- queueing delay
- packet loss
- link availability / failures
- dynamic temporal drift
- traffic scenarios

Scenarios include normal, rush-hour, degraded, failure and network-storm conditions.

## Baselines
The benchmark contains:

- Dijkstra shortest path using current link cost
- ECMP-style multi-path choice among near-equal shortest candidates
- Random routing
- Raw DQN
- Safety-Shielded DQN

## Important result interpretation
The bundled checkpoint was trained for a practical demo-sized run rather than a publication-grade hyperparameter sweep. The benchmark demonstrates that the safety layer can substantially reduce the raw DQN's pathological behavior, especially in failure and degraded scenarios, but **it does not prove that RL universally beats Dijkstra or ECMP**.

That distinction is intentional. For a research claim of RL superiority, train longer and across more seeds, perform hyperparameter search, and report confidence intervals over repeated topology/traffic draws.

## Path to a real network
The safe next stage is integration with a controlled emulator such as Mininet or ns-3, or with an SDN controller in a lab environment. The decision layer should remain separated from production routing changes.
