# DTNet — Networked Digital Twins for Supply Chain Disruption Prediction

DTNet is a research prototype that models industrial supply chains as a directed graph of
interconnected digital twins and uses a Graph Attention Network (GNN) combined with
agent-based simulation (ABS) to predict how disruptions cascade through the network.
Rather than monitoring each factory node in isolation — the current industry norm — DTNet
connects all digital twins into a single graph so that structural dependencies and
propagation paths are first-class citizens of the prediction model. The system is evaluated
against an isolated-twin baseline to quantify the benefit of the networked approach.

> **Context:** Master's thesis — HEC Lausanne, Master in Data Science, 2025–2026.

---

## Research Questions

| | Question |
|---|---|
| **RQ1** | How can a supply chain be represented as a directed graph of digital twins, and what node/edge attributes best capture entity state and inter-dependency? |
| **RQ2** | How can a GNN combined with agent-based simulation predict the propagation of cascading failures across this graph? |
| **RQ3** | Does the networked digital-twin approach outperform isolated (non-graph) twins for disruption prediction? |

---

## Architecture

![DTNet Architecture](results/fig_architecture.png)

| Layer | Component | Role |
|---|---|---|
| **Node** | Digital Twin Agents (Mesa) | Each supply-chain entity is a stateful agent with its own attributes and health score |
| **Graph** | Directed Graph G = (V, E) (NetworkX) | Nodes are digital twins; edges encode flow type and criticality weight |
| **Intelligence** | GNN + ABS | GATConv learns disruption propagation; Mesa simulation generates training data |

---

## Prerequisites

- Python **3.10** or higher
- `pip` (comes with Python)
- GPU optional (CPU training is supported; GPU training was done on Google Colab)

---

## Setup

```bash
git clone <repo-url>
cd DTNet/Prototype

python -m venv .venv

# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

> **PyTorch Geometric note:** `torch-geometric` requires a matching PyTorch version.
> If you need GPU support, install the correct CUDA wheel for your system first:
> <https://pytorch.org/get-started/locally/>
> then install `torch-geometric` separately following
> <https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html>.

---

## Data

The dataset (**"Machine Demand & Failure Prediction"**) is included in the repository at:

```
data/raw/updated_data.csv
```

No download required — the file is ready to use after cloning.

---

## Run the Full Pipeline

```bash
python run_all.py
```

This runs all 7 steps in order and saves all results to `results/`.
Expected runtime: 15–60 min depending on hardware (steps 3–4 are longest).

---

## Run Individual Components

```bash
# 1. Preprocess raw CSV
python -m src.data.preprocess updated_data.csv processed.csv

# 2. Generate simulation training data (5 000 runs)
python -m src.simulation.generate_data

# 3. Train GNN and baseline
python -m src.gnn.train

# 4. Evaluate on test set
python -m src.gnn.evaluate

# 5. Robustness evaluation across 5 seeds
python -m src.gnn.evaluate_robust

# 6. Generate thesis figures
python -m src.viz.architecture_viz
python -m src.viz.full_graph_viz
python -m src.viz.scenario_analysis_viz
python -m src.viz.robustness_viz
```

---

## Interactive Dashboard

```bash
streamlit run dashboard/app.py
```

---

## Tests

```bash
python -m pytest tests/
```

---

## Project Structure

```
Prototype/
├── data/
│   ├── raw/                    # Kaggle CSV (not in git — download manually)
│   └── processed/              # Generated .pkl and cleaned CSVs
├── src/
│   ├── agents/                 # Digital twin agent classes (one per node type)
│   ├── data/                   # CSV loader, preprocessor, entity mapping
│   ├── graph/                  # Graph builder, topology inference, metrics
│   ├── simulation/             # Mesa DTNetModel, scenarios, data generator
│   ├── gnn/                    # GATConv model, dataset, training, evaluation
│   └── viz/                    # Thesis figures and dashboard visualisations
├── dashboard/
│   └── app.py                  # Streamlit single-page demo
├── notebooks/                  # Jupyter notebooks (exploratory, phases 1–5)
├── results/                    # Saved figures (300 DPI) and JSON summaries
├── tests/                      # Unit tests
├── run_all.py                  # End-to-end reproducibility script
└── requirements.txt
```

---

## Key Results

| Metric | DTNetGNN | Isolated Baseline | Δ |
|---|---|---|---|
| R² | **0.68** | 0.41 | +0.27 |
| F1 | **0.70** | 0.54 | +0.16 |
| AUC | **0.86** | 0.71 | +0.15 |

- **Networked outperforms isolated** across all 5 random seeds on all metrics — confirming RQ3.
- **Critical-hub disruption** (highest betweenness node) cascaded to 71 / 82 nodes within 1 step.
- **Supplier cascade** (all 10 suppliers) produced a sustained health degradation to ~66%.
- Main thesis figures: `results/fig_robustness_comparison.png`, `results/fig_scenario_comparison.png`, `results/fig_full_graph.png`.

---

## Graph Topology

![DTNet Graph Topology](results/fig_full_graph.png)

82 nodes · 219 edges · 5 node types · 4 edge types

---

## Author

**Hamza Karmouche** — hamza.karmouche@unil.ch
Master in Data Science, HEC Lausanne

## Supervisor

**Prof. Yash Raj Shrestha** — HEC Lausanne, Department of Information Systems
