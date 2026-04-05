# DTNet — Tech Stack & Environment

## Python Version
- Python 3.10+ (use 3.11 if available)
- Use virtual environment: `python -m venv .venv`

## Core Dependencies

### Graph & Network
- **NetworkX** (`networkx`) — Build and manipulate the directed graph G = (V, E)
  - All graph operations go through NetworkX
  - Use `nx.DiGraph()` (directed), never `nx.Graph()` (undirected)

### Agent-Based Simulation
- **Mesa** (`mesa`) — Agent-based modeling framework
  - Each digital twin is a Mesa Agent
  - The supply chain network is a Mesa Model
  - Mesa handles scheduling, data collection, and step-by-step simulation

### Graph Neural Network
- **PyTorch** (`torch`) — Deep learning framework
- **PyTorch Geometric** (`torch_geometric`) — GNN library built on PyTorch
  - Use `torch_geometric.nn.GCNConv` or `torch_geometric.nn.GATConv`
  - Convert NetworkX graphs to PyG format with `torch_geometric.utils.from_networkx()`

### Data & Analysis
- **Pandas** (`pandas`) — Data loading and manipulation
- **NumPy** (`numpy`) — Numerical operations
- **Scikit-learn** (`scikit-learn`) — Metrics, train/test split, baselines

### Visualization
- **Matplotlib** (`matplotlib`) — Static plots and graph visualization
- **Plotly** (`plotly`) — Interactive visualizations (optional, for dashboard)
- **Seaborn** (`seaborn`) — Statistical plots

### Optional / Dashboard
- **Streamlit** (`streamlit`) — Quick interactive dashboard for thesis demo
  - Only if time permits, NOT a priority

## Installation

```bash
pip install networkx mesa torch torch_geometric pandas numpy scikit-learn matplotlib plotly seaborn
```

For PyTorch Geometric, follow official install guide (depends on PyTorch + CUDA version):
https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html

## Project Structure

```
dtnet/
├── instructions/          # This folder — AI guidance docs
│   ├── PROJECT.md
│   ├── TECH_STACK.md
│   ├── CODING_PATTERNS.md
│   ├── COMMON_MISTAKES.md
│   └── WORKFLOW.md
├── src/
│   ├── agents/            # Digital twin agent definitions
│   │   ├── base_agent.py        # Base DigitalTwinAgent class
│   │   ├── machine_agent.py     # Machine-level twin (with sensors)
│   │   ├── supplier_agent.py    # Supplier twin
│   │   ├── logistics_agent.py   # Logistics hub twin
│   │   ├── plant_agent.py       # Plant-level twin
│   │   └── distribution_agent.py
│   ├── graph/             # Graph construction and manipulation
│   │   ├── builder.py           # Build G = (V, E) from data
│   │   ├── topology.py          # Topology inference from dataset
│   │   └── metrics.py           # Graph centrality, vulnerability metrics
│   ├── simulation/        # Mesa-based cascading failure simulation
│   │   ├── model.py             # DTNetModel (Mesa Model)
│   │   ├── scheduler.py         # Custom activation scheduler
│   │   └── scenarios.py         # Pre-defined disruption scenarios
│   ├── gnn/               # Graph Neural Network
│   │   ├── dataset.py           # Convert simulations to PyG dataset
│   │   ├── model.py             # GNN architecture (GCN or GAT)
│   │   ├── train.py             # Training loop
│   │   └── evaluate.py          # Evaluation and comparison with baseline
│   ├── data/              # Data loading and preprocessing
│   │   ├── loader.py            # Load Kaggle dataset
│   │   └── preprocess.py        # Clean, normalize, feature engineering
│   └── viz/               # Visualization
│       ├── graph_viz.py         # Network visualizations
│       ├── simulation_viz.py    # Cascading failure animations
│       └── comparison_viz.py    # Networked vs isolated plots
├── notebooks/             # Jupyter notebooks for exploration
│   ├── 01_data_exploration.ipynb
│   ├── 02_graph_construction.ipynb
│   ├── 03_simulation_runs.ipynb
│   ├── 04_gnn_training.ipynb
│   └── 05_results_comparison.ipynb
├── data/
│   ├── raw/               # Original Kaggle dataset
│   └── processed/         # Cleaned data, generated simulations
├── results/               # Output figures, tables, metrics
├── tests/                 # Unit tests
├── requirements.txt
└── README.md
```

## Environment Notes

- **GPU:** Use Google Colab (free GPU) for GNN training if local machine is slow
- **Git:** Commit after every major feature. Use meaningful commit messages.
- **Notebooks vs Scripts:** Use notebooks for exploration, scripts for the actual pipeline. The final prototype should run as scripts, not just notebooks.
