# DTNet — Networked Digital Twins for Supply Chain Disruption Prediction

DTNet is a research prototype that models industrial supply chains as a directed graph of
interconnected digital twins and uses a Graph Attention Network (GNN) combined with
agent-based simulation (ABS) to predict how disruptions cascade through the network.
Rather than monitoring each factory node in isolation — the current industry norm — DTNet
connects all digital twins into a single graph so that structural dependencies and propagation
paths are first-class citizens of the prediction model. The system is evaluated against an
isolated-twin baseline to quantify the benefit of the networked approach.

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

The framework has three layers:

| Layer | Component | Role |
|---|---|---|
| **Node** | Digital Twin Agents (Mesa) | Each supply-chain entity (supplier, logistics, plant, machine, distribution) is a stateful agent with its own attributes and health score |
| **Graph** | Directed Graph G = (V, E) (NetworkX) | Nodes are digital twins; edges encode flow type (`material_flow`, `operational`, `process_chain`, `shared_part_dependency`) and criticality weight |
| **Intelligence** | GNN + ABS | A GATConv-based model learns disruption propagation from graph structure; Mesa simulation generates training data and runs scenarios |

---

## Installation

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

> **PyTorch Geometric** requires a matching PyTorch + CUDA version.
> See the [official installation guide](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html).
> GPU training was performed on Google Colab.

---

## Usage

### 1 — Generate simulation training data
```bash
python -m src.simulation.generate_data
```

### 2 — Train the GNN
```bash
python -m src.gnn.train
```

### 3 — Evaluate robustness (5 seeds, networked vs isolated)
```bash
python -m src.gnn.evaluate_robust
```

### 4 — Run scenario analysis visualisation
```bash
python -m src.viz.scenario_analysis_viz
```

### Interactive dashboard (thesis demo)
```bash
streamlit run dashboard/app.py
```

### Tests
```bash
python -m pytest tests/
```

---

## Project Structure

```
Prototype/
├── data/
│   ├── raw/                    # Original Kaggle dataset (updated_data.csv)
│   └── processed/              # Simulation runs (.pkl), cleaned DataFrames
├── src/
│   ├── agents/                 # Digital twin agent classes (one per node type)
│   ├── data/                   # CSV loader, preprocessor, entity mapping
│   ├── graph/                  # Graph builder, topology inference, metrics
│   ├── simulation/             # Mesa DTNetModel, scenarios, data generator
│   ├── gnn/                    # GATConv model, dataset, training, evaluation
│   └── viz/                    # Thesis figures and dashboard visualisations
├── dashboard/
│   └── app.py                  # Streamlit single-page demo
├── notebooks/                  # Exploratory Jupyter notebooks (phases 1–5)
├── results/                    # Saved figures (300 DPI) and model checkpoints
├── tests/                      # Unit tests
├── instructions/               # Project guidance (CLAUDE.md companion docs)
├── requirements.txt
└── CLAUDE.md
```

---

## Key Results

All output figures are saved to `results/` at 300 DPI. Key findings:

- **Networked outperforms isolated** across all 5 random seeds on MAE, RMSE, F1, Precision, and Recall — confirming RQ3.
- **Critical-hub disruption** (highest betweenness node) caused the widest cascade: 71 / 82 nodes disrupted within 1 simulation step.
- **Supplier cascade** (all 10 suppliers simultaneously) produced a slower but sustained health degradation, reaching a final network health of ~66%.
- See `results/fig_robustness_comparison.png`, `results/fig_scenario_comparison.png`, and `results/fig_full_graph.png` for the main thesis figures.

---

## Graph Topology

![DTNet Graph Topology](results/fig_full_graph.png)

82 nodes · 219 edges · 5 node types · 4 edge types

---

## Author

[Your Name] — [your.email@unil.ch]

## Supervisor

**Prof. Yash Raj Shrestha** — HEC Lausanne, Department of Information Systems
