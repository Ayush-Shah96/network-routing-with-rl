# Adaptive Network Intelligence
## Dynamic Network Routing with Reinforcement Learning — Local Tool Guide

Adaptive Network Intelligence is a local research and demonstration platform for **dynamic computer-network routing with reinforcement learning**.

It simulates a changing network, trains/loads a Double-DQN routing policy, compares that policy with traditional routing baselines, injects congestion and link failures, and visualizes the routing decision process through a Streamlit web interface.

> **Important:** This project is a network simulator and decision-making research tool. It does **not** change your computer's real routing table, operating-system network routes, or production network interfaces.

---

## 1. What this tool does

The tool models a network as a graph:

```text
A ---- B ---- D
|      |      |
|      |      |
C ---- E ---- F
```

Each link can have changing conditions such as:

- latency
- bandwidth
- congestion
- queueing delay
- packet loss
- link availability/failure

The RL agent observes the current network state and chooses the next hop for a packet. It can therefore change its route when network conditions change.

Example:

```text
Normal network:
A -> B -> D -> F

After congestion on B -> D:
A -> C -> E -> F
```

---

## 2. Main capabilities

### Reinforcement learning

The included routing agent uses an advanced DQN design:

- Double DQN
- Dueling network architecture
- Prioritized experience replay
- Epsilon-greedy exploration
- Valid-action masking
- Model checkpoint save/load

### Dynamic network simulation

The simulator supports:

- multiple topology types
- changing latency
- changing bandwidth
- congestion
- queueing delay
- packet loss
- link failures
- traffic stress scenarios

Built-in scenarios include:

```text
normal
rush_hour
degraded
failure
storm
```

### Routing baselines

The benchmark can compare:

- Safe DQN
- DQN
- Dijkstra shortest path
- ECMP-style routing
- Random routing

### Explainability

The dashboard can show the decision information used by the learned router, including:

- current state
- candidate next hops
- learned Q-values
- latency/cost conditions
- selected next hop

### Benchmarking

The tool can run repeated experiments across multiple scenarios and save:

```text
artifacts/benchmark_results.csv
artifacts/benchmark_summary.csv
```

### Local API

A FastAPI server is also included for programmatic access to the routing engine.

---

# 3. Install on Windows

## Easiest method

Double-click:

```text
setup_windows.bat
```

That script:

1. creates a Python virtual environment
2. activates it
3. upgrades pip
4. installs the required packages
5. runs the automated test suite

When it finishes, start the tool by double-clicking:

```text
start_windows.bat
```

The Streamlit application will print a local address, usually similar to:

```text
http://localhost:8501
```

Open that address in Chrome, Edge, Firefox, or another browser.

## Manual Windows installation

Open Command Prompt or PowerShell in the project folder:

```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pytest -q
```

Then run:

```powershell
python -m streamlit run app\streamlit_app.py
```

---

# 4. Install on Linux/macOS

From the project directory:

```bash
chmod +x setup_unix.sh start_unix.sh start_api_unix.sh
./setup_unix.sh
```

Then start the dashboard:

```bash
./start_unix.sh
```

Or manually:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pytest -q
python -m streamlit run app/streamlit_app.py
```

---

# 5. First use: run the public demo locally

After starting Streamlit, open the URL it prints.

You will see the **Adaptive Network Intelligence** dashboard.

Use this recommended first demo:

### Step 1 — Choose a topology

In the sidebar select:

```text
Topology: mesh
```

### Step 2 — Select source/destination

Example:

```text
Source: A
Destination: O
```

The exact node names available depend on the selected network size.

### Step 3 — Select the policy

Start with:

```text
Safe DQN
```

### Step 4 — Select a scenario

Start with:

```text
normal
```

### Step 5 — Run

Click:

```text
Run adaptive routing
```

The dashboard shows:

- selected route
- network topology
- latency
- packet loss
- throughput
- hop count
- delivery status

---

# 6. Best live demonstration

The most useful demonstration is to show that the route changes when the network becomes unhealthy.

### Demonstration sequence

Start with:

```text
Policy: Safe DQN
Scenario: normal
```

Run the route and note the path.

Then use:

```text
Inject targeted congestion
```

Run the routing decision again.

The simulator increases the stress on the selected route or its surrounding links. The agent can then choose a different path.

You can also use:

```text
Inject link failure
```

to simulate a failed connection.

For a stronger stress demo, select:

```text
rush_hour
failure
storm
```

The important idea to demonstrate is:

```text
Network changes
      ↓
RL observes the new state
      ↓
Candidate route costs change
      ↓
RL selects a different next hop
      ↓
Traffic is routed through another path
```

---

# 7. Use Dijkstra vs DQN vs Safe DQN

The project is designed for comparison rather than assuming RL is automatically better.

### Dijkstra

A conventional graph-search baseline. It is strong when the cost model is accurate and current.

### DQN

The learned policy chooses actions using its neural network value estimates.

### Safe DQN

The learned DQN policy is combined with a routing safety layer. Invalid or clearly unsafe actions can be blocked before they are applied.

This is useful when the model has not yet learned every edge case.

### ECMP

A multipath-style baseline used for comparison.

### Random

A deliberately weak baseline that helps show how much routing quality is gained over arbitrary decisions.

---

# 8. Retrain the model

A pre-trained checkpoint is included:

```text
models/dqn_routing.pt
```

You can retrain it with the supplied script.

## Basic training

Activate the virtual environment first.

Windows:

```powershell
.venv\Scripts\activate
```

Linux/macOS:

```bash
source .venv/bin/activate
```

Then:

```bash
python scripts/train_model.py
```

The default configuration trains for the number of episodes defined by the script/configuration.

## Custom training

Example:

```bash
python scripts/train_model.py --episodes 5000 --seed 42 --topology mesh --nodes 15
```

Model output:

```text
models/dqn_routing.pt
```

Training history:

```text
artifacts/training_history.json
```

### Important

More training does not automatically guarantee better results. Use the benchmark after training to verify whether the learned policy actually improved.

---

# 9. Local REST API

A FastAPI service is included for applications that want to call the routing engine programmatically.

### Windows

Double-click:

```text
start_api_windows.bat
```

### Linux/macOS

Run:

```bash
./start_api_unix.sh
```

Or manually:

```bash
python run_api.py
```

The API is intended for local development and integration with other software.

The interactive Streamlit dashboard is the recommended starting point for normal users.

---

# 10. Project structure

```text
dynamic-network-routing-rl/
│
├── app/
│   └── streamlit_app.py          # Interactive local dashboard
│
├── rl/
│   ├── agent.py                  # Double-DQN / dueling / replay logic
│   ├── network.py                # Neural-network components
│   └── replay_buffer.py          # Prioritized replay implementation
│
├── routing/
│   ├── baselines.py              # Dijkstra / ECMP / Random
│   └── rl_router.py              # Learned and safety-shielded routing
│
├── simulation/
│   ├── environment.py            # Dynamic RL environment
│   └── topology.py               # Network generation/topologies
│
├── scripts/
│   ├── train_model.py            # Train/retrain checkpoint
│   ├── evaluate.py               # Multi-scenario benchmark
│   ├── experiment.py             # Experiment workflow
│   └── run_demo.py               # CLI demonstration
│
├── models/
│   └── dqn_routing.pt            # Pre-trained checkpoint
│
├── artifacts/
│   ├── training_history.json
│   ├── benchmark_results.csv
│   └── benchmark_summary.csv
│
├── tests/                        # Automated tests
│
├── config.yaml                   # Main experiment configuration
├── requirements.txt              # Python dependencies
├── api.py                        # FastAPI routes
├── run_api.py                    # API launcher
├── setup_windows.bat             # Windows setup
├── start_windows.bat             # Windows dashboard launcher
├── setup_unix.sh                 # Linux/macOS setup
├── start_unix.sh                 # Linux/macOS dashboard launcher
└── PROJECT_REPORT.md             # Technical project report
```

---

# 11. What is "Safety-Shielded DQN"?

A learned model can make poor decisions, especially when it has not seen a particular network condition during training.

The safety layer therefore checks candidate actions before allowing them to be used.

It can prevent decisions involving conditions such as:

- failed links
- invalid neighbors
- previously visited nodes in a route
- obviously unsafe routing choices

This creates a hybrid system:

```text
Learned policy
      ↓
Safety validation
      ↓
Final routing action
```

This is an important research direction because practical network automation should not assume a neural network is always correct.

---

# 12. Recommended learning path

For someone new to the project:

```text
1. Start Streamlit
        ↓
2. Run normal DQN routing
        ↓
3. Inject congestion
        ↓
4. Inject link failure
        ↓
5. Compare Dijkstra / DQN / Safe DQN
        ↓
6. Open Explainable AI
        ↓
7. Run Benchmark Lab
        ↓
8. Run training
        ↓
9. Re-run benchmark
        ↓
10. Modify the simulator or reward function
```

---

# 13. Quick command reference

## Windows

```powershell
# First-time setup
setup_windows.bat

# Start dashboard
start_windows.bat

# Start API
start_api_windows.bat

# Tests
.venv\Scripts\activate
pytest -q

# Train
python scripts/train_model.py --episodes 2400

# Benchmark
python scripts/evaluate.py --runs 200
```

## Linux/macOS

```bash
# First-time setup
./setup_unix.sh

# Start dashboard
./start_unix.sh

# Start API
./start_api_unix.sh

# Tests
source .venv/bin/activate
pytest -q

# Train
python scripts/train_model.py --episodes 2400

# Benchmark
python scripts/evaluate.py --runs 200
```
