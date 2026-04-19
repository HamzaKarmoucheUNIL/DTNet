# DTNet — Digital Twin Network for Supply Chain Disruption Prediction

DTNet is a research prototype that models industrial supply chains as networks of interconnected digital twins. It combines Graph Neural Networks (GNN) with agent-based simulation to detect and predict cascading disruptions across multi-layered supply chain graphs. The system was designed to evaluate how structural vulnerabilities propagate from individual machine failures to plant-level and network-level disruptions.

> **Context:** Master's thesis — HEC Lausanne, Master in Data Science, 2025–2026.

---

## Project Status

| Phase | Description | Status |
|-------|-------------|--------|
| 1 | Data Loading & Preprocessing | ✅ Complete |
| 2 | Graph Construction | ✅ Complete |
| 3 | Agent-Based Simulation | ✅ Complete |
| 4 | GNN Training | ✅ Complete |
| 5 | Results & Comparison | 🔄 In progress |

---

## Project Structure

```
Prototype/
├── data/
│   ├── raw/                        # Raw synthetic industrial machine data
│   └── processed/                  # Cleaned and feature-engineered datasets
├── notebooks/
│   ├── 01_data_exploration.ipynb   # EDA and preprocessing
│   ├── 02_graph_construction.ipynb # Build the digital twin network
│   ├── 03_simulation_runs.ipynb    # Agent-based disruption scenarios
│   ├── 04_gnn_training.ipynb       # GAT model training and evaluation
│   └── 05_results_comparison.ipynb # Benchmarking and result analysis
├── src/
│   ├── data/        # Data loading and entity mapping
│   ├── graph/       # Graph builder, topology, and vulnerability metrics
│   ├── simulation/  # Mesa-based agents, scenarios, and scheduler
│   ├── gnn/         # GAT model, dataset, training loop, evaluation
│   └── viz/         # Visualization utilities (dark theme)
├── results/         # Saved figures, trained model checkpoints (.pt)
├── tests/
├── requirements.txt
└── CLAUDE.md
```

---

## Tech Stack

| Component | Library |
|-----------|---------|
| Graph modeling | NetworkX (`DiGraph`) |
| Agent-based simulation | Mesa |
| GNN (Graph Attention Network) | PyTorch Geometric |
| Deep learning framework | PyTorch |
| Data manipulation | Pandas, NumPy |
| Machine learning utilities | scikit-learn |
| Visualization | Matplotlib (dark theme), Seaborn, Plotly |

---

## Installation

```bash
pip install -r requirements.txt
```

> **Note:** PyTorch Geometric requires a compatible version of PyTorch. Refer to the [official installation guide](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html) if you encounter issues. GPU training was performed on Google Colab.

---

## Running the Project

Execute the notebooks **in order**:

1. **`01_data_exploration.ipynb`** — Load raw data, perform EDA, and export cleaned features to `data/processed/`.
2. **`02_graph_construction.ipynb`** — Build the multi-layered directed graph (suppliers → plants → machines → parts) and compute structural metrics.
3. **`03_simulation_runs.ipynb`** — Run five disruption scenarios using the Mesa agent-based model and export simulation traces.
4. **`04_gnn_training.ipynb`** — Train the Graph Attention Network on simulation-labelled graphs; best checkpoint saved to `results/dtnet_gnn_best.pt`.
5. **`05_results_comparison.ipynb`** — Compare DTNet predictions against an isolated (non-graph) baseline; generate final figures.

All outputs (figures, model weights) are written to the `results/` directory.

---

## Disruption Scenarios

Five scenarios are evaluated in Phase 3:

| ID | Scenario | Description |
|----|----------|-------------|
| S1 | Single supplier failure | One critical supplier goes offline |
| S2 | Logistics bottleneck | A transport node becomes saturated |
| S3 | Multi-supplier disruption | Simultaneous failure of several suppliers |
| S4 | Targeted attack | High-centrality nodes are selectively removed |
| S5 | Random disruption | Uniformly random node failures |

---

## Reproducibility

All experiments use fixed random seeds (`np.random.seed(42)`, `torch.manual_seed(42)`). Results are fully reproducible given the same environment and data files.
