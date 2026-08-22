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

# 3. System requirements

Recommended:

- Windows 10/11, Linux, or macOS
- Python 3.11+ recommended
- 4 GB RAM minimum
- 8 GB+ RAM recommended for larger experiments
- Internet connection for the first installation only, to download Python packages

A GPU is **not required** for the normal demo. Training can run on CPU, although longer training experiments will take more time.

---

# 4. Get the project

Extract the project ZIP to a folder such as:

```text
C:\Projects\dynamic-network-routing-rl
```

or:

```text
~/projects/dynamic-network-routing-rl
```

Open a terminal in the extracted project directory.

You should see files similar to:

```text
README.md
README_LOCAL_TOOL.md
requirements.txt
config.yaml
app/
rl/
routing/
simulation/
scripts/
models/
tests/
```

---

# 5. Install on Windows

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

# 6. Install on Linux/macOS

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

# 7. First use: run the public demo locally

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

# 8. Best live demonstration

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

# 9. Understand the dashboard

## Live network / topology state

The graph displays the simulated network.

Links can appear healthy, congested, or failed depending on their current state.

The selected route is highlighted.

## Routing outcome

This panel contains:

```text
Policy
Route
Latency
Packet loss
Throughput
Hops
Delivery status
```

## Explainable AI

Run the DQN or Safe DQN policy and open the **Explainable AI** tab.

This shows the learned action values and the network conditions considered at each routing decision.

The purpose is to answer:

> Why did the agent choose this next hop?

## Benchmark Lab

Use this tab to compare routing strategies across multiple dynamic scenarios.

The benchmark evaluates:

```text
Dijkstra
ECMP
Random
DQN
Safe DQN
```

Choose the number of runs and click:

```text
Run full benchmark
```

You can download the detailed CSV from the dashboard.

## Training Lab

Shows the stored training history when available.

The tool plots:

- smoothed episode reward
- smoothed learning loss

## Network Telemetry

Shows current edge-level telemetry including:

- latency
- congestion
- packet loss
- availability
- queue-related conditions

You can also download a JSON snapshot.

---

# 10. Use Dijkstra vs DQN vs Safe DQN

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

# 11. Retrain the model

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

# 12. Run the benchmark from the command line

Activate the virtual environment, then run:

```bash
python scripts/evaluate.py --runs 200 --topology mesh --nodes 15
```

The benchmark prints a summary and writes:

```text
artifacts/benchmark_results.csv
artifacts/benchmark_summary.csv
```

For a larger experiment:

```bash
python scripts/evaluate.py --runs 1000 --topology mesh --nodes 15
```

The CSV files can be opened in Excel, LibreOffice, pandas, or another analysis tool.

---

# 13. Run the experiment workflow

The project also contains:

```text
scripts/experiment.py
scripts/run_demo.py
```

These are useful for repeatable experiments and non-UI demonstrations.

Example:

```bash
python scripts/run_demo.py
```

---

# 14. Run automated tests

Run:

```bash
pytest -q
```

Tests cover the simulator and advanced routing components.

Run the tests whenever you change:

- reward logic
- topology generation
- action selection
- model loading
- routing behaviour
- environment transitions

---

# 15. Local REST API

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

# 16. Project structure

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

# 17. Configuration

The main configuration file is:

```text
config.yaml
```

Example parameters include:

```yaml
seed: 7
topology: mesh
nodes: 15
training:
  episodes: 2400
  max_steps: 20
  epsilon_start: 1.0
  epsilon_end: 0.03
  epsilon_decay: 0.996
scenarios:
  - normal
  - rush_hour
  - degraded
  - failure
  - storm
benchmark_runs: 200
```

You can change the defaults for experiments, but changing topology size or model dimensions may require retraining the checkpoint.

---

# 18. Reproducibility

For repeatable experiments:

1. Set a fixed seed.
2. Keep the topology and node count unchanged.
3. Keep the model checkpoint name/versioned.
4. Record the training episode count.
5. Record benchmark run count.
6. Save the resulting CSVs.

Example:

```bash
python scripts/train_model.py --episodes 5000 --seed 123 --topology mesh --nodes 15
python scripts/evaluate.py --runs 500 --seed 123 --topology mesh --nodes 15
```

---

# 19. How the RL routing works

At each routing decision the agent observes a state containing information about the current routing position and the neighboring links.

Conceptually:

```text
Current node
Destination
Neighbor latency
Neighbor bandwidth
Neighbor congestion
Neighbor packet loss
Queue conditions
Link availability
        ↓
Neural network
        ↓
Q-value for each candidate next hop
        ↓
Selected next hop
```

The reward encourages useful routing behaviour such as:

- reaching the destination
- lower latency
- lower congestion
- lower packet loss
- avoiding invalid moves
- avoiding routing loops

The exact reward function is implemented in the simulation environment and should be treated as part of the experiment design.

---

# 20. What is "Safety-Shielded DQN"?

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

# 21. How to present this project publicly

A strong 3–5 minute demonstration is:

### 1. Show the normal network

Run Safe DQN and show the initial path.

### 2. Show metrics

Point out:

```text
Latency
Packet loss
Throughput
Hops
```

### 3. Inject congestion

Use:

```text
Inject targeted congestion
```

### 4. Run routing again

Show that the policy can select a different path.

### 5. Inject a failure

Use:

```text
Inject link failure
```

### 6. Compare policies

Open **Benchmark Lab** and compare:

```text
Dijkstra vs ECMP vs Random vs DQN vs Safe DQN
```

### 7. Explain the AI decision

Open **Explainable AI** and show why the selected next hop was attractive relative to alternatives.

This demonstrates the entire idea without requiring the audience to read source code.

---

# 22. Important interpretation of benchmark results

Do not assume that RL will always beat Dijkstra.

A routing model can perform worse when:

- it has insufficient training
- the reward is poorly tuned
- the state representation is incomplete
- the network distribution changes significantly
- the baseline is already very strong
- the model has not seen enough failure/congestion patterns

That is why this project includes benchmark and evaluation tooling.

A credible result is:

```text
Train
  ↓
Evaluate
  ↓
Compare with baselines
  ↓
Inspect failure cases
  ↓
Adjust model/reward/environment
  ↓
Retrain
  ↓
Evaluate again
```

This turns the project into an experimental platform rather than a hard-coded demo.

---

# 23. Troubleshooting

## Python is not recognized

Install Python and make sure it is available from the terminal.

Windows check:

```powershell
python --version
```

Linux/macOS check:

```bash
python3 --version
```

## Streamlit command is not found

Activate the virtual environment first:

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
python -m streamlit run app/streamlit_app.py
```

## PyTorch installation problem

Upgrade pip and reinstall the requirements:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If your machine has a special CUDA/GPU requirement, consult the official PyTorch installation instructions for your platform rather than changing the model code.

## Model file missing

Train a checkpoint:

```bash
python scripts/train_model.py --episodes 2400
```

Then start Streamlit again.

## Tests fail after making code changes

Run:

```bash
pytest -q
```

Read the first failing test before changing multiple components at once.

## Port 8501 is already in use

Start Streamlit on another local port:

```bash
python -m streamlit run app/streamlit_app.py --server.port 8502
```

Then open:

```text
http://localhost:8502
```

---

# 24. Security and safety notes

This tool is intended for local experimentation.

It does not automatically:

- modify OS routing tables
- reconfigure network interfaces
- inject traffic into external networks
- control a production router
- connect to the public internet for packet forwarding

The simulator is intentionally isolated so that routing experiments can be performed safely.

Before connecting the decision layer to real network infrastructure, add authentication, authorization, action validation, rollback controls, rate limits, audit logging, and a controller such as an SDN testbed or network emulator.

---

# 25. Recommended learning path

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

# 26. Suggested advanced extensions

The current platform is deliberately structured so it can be extended further.

The next major research-level integrations would be:

- graph neural network routing policies
- multi-agent reinforcement learning
- multi-commodity-flow optimization
- offline RL from historical telemetry
- distributional RL
- constrained RL with formal safety budgets
- curriculum learning for increasingly difficult topologies
- Mininet integration
- ns-3 integration
- Software Defined Networking controller integration
- real telemetry ingestion through a controlled testbed
- model versioning and experiment tracking
- larger-scale distributed training

A real-network integration should be done in a dedicated lab/testbed, not directly against a production network.

---

# 27. Quick command reference

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

---

# 28. Project status

This project is intended as an **advanced simulation/research tool and public demonstration**, not as a production-grade router.

The most important principle is to evaluate the learned policy honestly against strong baselines and under multiple network conditions rather than relying only on training reward.

For the technical design, implementation details, and research discussion, see:

```text
PROJECT_REPORT.md
```

---

## License / usage

This repository is suitable for local educational, experimental, and demonstration use. Add the license and any organization-specific usage terms before distributing it publicly.
