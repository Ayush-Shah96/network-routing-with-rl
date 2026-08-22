# Installation

## Windows

Install Python 3.11+ and run:

```bat
setup_windows.bat
start_windows.bat
```

The setup script creates a virtual environment and installs the dependencies. The start script launches Streamlit.

## Linux / macOS

```bash
bash setup_unix.sh
bash start_unix.sh
```

## CLI

Train:

```bash
python scripts/train_model.py --episodes 2400
```

Evaluate:

```bash
python scripts/evaluate.py --runs 200
```

Test:

```bash
pytest -q
```

## No internet after setup

Once Python packages are installed and the bundled model exists, the dashboard itself runs locally and does not require an external API.

## Local API mode

The same routing engine can be consumed as a local HTTP service:

### Windows

```bat
start_api_windows.bat
```

### Linux/macOS

```bash
bash start_api_unix.sh
```

The service listens on `127.0.0.1:8080`.

`GET /health` checks the service/model. `POST /route` accepts source, destination, scenario, seed and a safe/unsafe policy flag.
