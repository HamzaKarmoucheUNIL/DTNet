# DTNet — Networked Digital Twins for Supply Chain Disruption Prediction

DTNet is a research prototype that models an industrial supply chain as a directed graph of
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
| **Graph** | Directed Graph G = (V, E) (NetworkX) | 82 nodes · 219 edges · 5 node types · 4 edge types |
| **Intelligence** | Dual-head GAT + ABS | GATConv learns disruption propagation; Mesa simulation generates 5 000 training graphs |

**Model:** Dual-head GAT (`DTNetGNN`) — 16-dim node features (10 twin + 6 structural),
`hidden=128`, `heads=(4, 1)`, `edge_dim=3`, 78 466 trainable parameters.
Outputs: regression severity score + binary disrupted/not classification.

---

## Key Results

Results sourced from `results/results_summary.json` (test set) and
`results/robustness_results.json` (mean over 5 seeds: 42, 123, 456, 789, 1024).

| Metric | DTNetGNN | Isolated Baseline | Δ |
|---|---|---|---|
| R² | **0.68** | 0.28 | +0.40 |
| F1 (5-seed mean) | **0.703** | 0.255 | +0.448 |
| AUC (5-seed mean) | **0.856** | 0.656 | +0.200 |
| MAE | **0.078** | 0.124 | −0.046 |

- DTNetGNN outperforms the isolated baseline on all metrics across all 5 seeds — confirming RQ3.
- Critical-hub scenario (highest-betweenness node) cascaded to 71 / 82 nodes within 1 step.
- Supplier-cascade scenario (all 10 suppliers simultaneously) produced sustained health degradation to ~66%.

Main thesis figures: `results/fig_robustness_comparison.png`, `results/fig_seeds_stability.png`,
`results/fig_scenario_comparison.png`, `results/fig_full_graph.png`, `results/fig_loss_curves.png`.

---

## Graph Topology

![DTNet Graph Topology](results/fig_full_graph.png)

82 nodes · 219 edges · 5 node types · 4 edge types

---

## Prerequisites

- Python **3.10** or higher
- `pip` (comes with Python)
- GPU optional — CPU training is supported; GPU training was done on Google Colab

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

> **PyTorch Geometric note:** `torch-geometric` requires a matching PyTorch build.
> If you need GPU support, install the correct CUDA wheel for your system from
> <https://pytorch.org/get-started/locally/> first, then install `torch-geometric`
> following <https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html>.

> **Windows encoding note:** Some preprocessing output contains Unicode characters.
> Run scripts with `$env:PYTHONUTF8="1"` in PowerShell to avoid `charmap` codec errors.

---

## Data

The raw dataset (**"Machine Demand & Failure Prediction"**, 219 200 rows × 21 columns)
is included in the repository at:

```
data/raw/updated_data.csv
```

No download required — the file is ready to use after cloning.

---

## Run the Full Pipeline

```bash
python run_all.py
```

Runs all 7 steps in order and saves all results to `results/`.
Expected runtime: 15–60 min depending on hardware (steps 3–4 are the longest).

The 7 steps are:
1. Preprocess raw CSV → `data/processed/processed.csv`
2. Build the DTNet graph (82 nodes, 219 edges)
3. Generate simulation training data → `data/processed/simulation_runs.pkl` (5 000 runs)
4. Train DTNetGNN + IsolatedBaseline → `results/dtnet_gnn_best.pt`
5. Evaluate on test set → `results/results_summary.json`
6. Robustness evaluation across 5 seeds → `results/robustness_results.json`
7. Generate thesis figures (architecture, full graph, scenario analysis, robustness)

---

## Run Individual Components

```bash
# Preprocess raw CSV
python -m src.data.preprocess

# Generate simulation training data (5 000 runs)
python -m src.simulation.generate_data

# Train GNN and baseline
python -m src.gnn.train

# Evaluate on test set
python -m src.gnn.evaluate

# Robustness evaluation across 5 seeds
python -m src.gnn.evaluate_robust

# Generate thesis figures
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

The dashboard runs the **agent-based cascade simulation** interactively — it does not
perform GNN inference. Controls: propagation decay, threshold, disruption severity, and
4 scenarios (random node, critical hub, all suppliers, bottleneck plant). Visualises
the before/after network state, summary metrics, and a step-by-step cascade timeline slider.

---

## Project Structure

```
Prototype/
├── data/
│   ├── raw/                    # Raw CSV (included in repo — 219 200 rows)
│   └── processed/              # Generated simulation_runs.pkl and processed CSVs
├── src/
│   ├── agents/                 # Digital twin agent classes (one per node type)
│   │                           #   base_agent, machine_agent, supplier_agent,
│   │                           #   plant_agent, logistics_agent, distribution_agent
│   ├── data/                   # loader, preprocess, entity_mapping
│   ├── graph/                  # topology, builder, metrics, vulnerability metrics
│   ├── simulation/             # Mesa DTNetModel, scenarios, data generator, scheduler
│   ├── gnn/                    # model (dual-head GAT), dataset, train, tune,
│   │                           #   evaluate, evaluate_robust
│   └── viz/                    # Thesis figures and visualisation helpers
├── dashboard/
│   └── app.py                  # Streamlit single-page demo (simulation only)
├── notebooks/                  # Jupyter notebooks (exploratory, phases 1–5)
│   ├── 01_data_exploration.ipynb
│   ├── 02_graph_construction.ipynb
│   ├── 03_simulation_runs.ipynb
│   ├── 04_gnn_training.ipynb
│   └── 05_results_comparison.ipynb
├── scripts/                    # One-off thesis appendix generators
│   ├── gen_appendix_D.py       # LaTeX: node centralities + edge structure
│   ├── gen_appendix_E.py       # LaTeX: training constants, param counts, loss curve
│   └── out/                    # Generated .tex snippets
├── results/                    # Saved figures (300 DPI), JSON summaries, checkpoints
├── run_all.py                  # End-to-end reproducibility script (7 steps)
└── requirements.txt
```

---

## Tech Stack

| Component | Library | Version |
|---|---|---|
| Graph | NetworkX | 3.6.1 |
| Agent simulation | Mesa | 3.5.1 |
| GNN | PyTorch Geometric (GATConv) | torch 2.11.0 / pyg 2.7.0 |
| Data | Pandas / NumPy / Scikit-learn | 3.0.2 / 2.4.4 / 1.8.0 |
| Visualisation | Matplotlib / Seaborn / Plotly | 3.10.8 / 0.13.2 / 6.6.0 |
| Dashboard | Streamlit | 1.56.0 |

---

## Author

**Hamza Karmouche** — hamza.karmouche@unil.ch
Master in Data Science, HEC Lausanne

## Supervisor

**Prof. Yash Raj Shrestha** — HEC Lausanne, Department of Information Systems

---

## Discrepancies Corrected from Previous README

The following claims in the old README were incorrect and have been fixed:

| Field | Old README | Correct value | Source |
|---|---|---|---|
| Baseline R² | 0.41 | **0.28** | `results/results_summary.json` |
| Baseline F1 | 0.54 | **0.255** | `results/robustness_results.json` |
| Baseline AUC | 0.71 | **0.656** | `results/robustness_results.json` |
| Δ R² | +0.27 | **+0.40** | derived from corrected baseline |
| Δ F1 | +0.16 | **+0.448** | derived from corrected baseline |
| Δ AUC | +0.15 | **+0.200** | derived from corrected baseline |
| `data/raw/` | "not in git — download manually" | **included in repo** | `data/raw/updated_data.csv` exists |
| `tests/` | listed as a directory | **does not exist** | no tests/ directory in repo |
| `python -m pytest tests/` | listed under "Tests" | **removed** (no tests exist) | |
