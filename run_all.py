"""run_all.py — Reproduce all DTNet results end-to-end with a single command.

Pipeline
--------
1. Data Preprocessing      src/data/preprocess.py
2. Graph Construction      src/graph/builder.py  (inline)
3. Simulation Data         src/simulation/generate_data.py
4. GNN Training            src/gnn/train.py
5. GNN Evaluation          src/gnn/evaluate.py
6. Robustness Evaluation   src/gnn/evaluate_robust.py
7. Thesis Figures          src/viz/

Usage
-----
    python run_all.py

Prerequisites: data/raw/updated_data.csv must exist (see README).
"""

from __future__ import annotations

import random
import subprocess
import sys
import time

import numpy as np
import torch

np.random.seed(42)
torch.manual_seed(42)
random.seed(42)

PYTHON: str = sys.executable
RAW_CSV: str = "updated_data.csv"
TOTAL: int = 7

# Inline graph-construction script (builder.py has no __main__)
_GRAPH_SCRIPT: str = (
    "from src.data.loader import load_csv; "
    "from src.data.preprocess import preprocess; "
    "from src.data.entity_mapping import build_entity_mappings; "
    "from src.graph.topology import infer_topology; "
    "from src.graph.builder import build_graph, print_graph_summary; "
    "df_raw=load_csv('updated_data.csv'); "
    "df_c,_=preprocess(df_raw); "
    "em=build_entity_mappings(df_raw); "
    "ns,es=infer_topology(em); "
    "G=build_graph(ns,es,df_c); "
    "print_graph_summary(G)"
)

_VIZ_MODULES: list[str] = [
    "src.viz.architecture_viz",
    "src.viz.full_graph_viz",
    "src.viz.scenario_analysis_viz",
    "src.viz.robustness_viz",
]


def _run(step: int, label: str, cmd: list[str]) -> None:
    """Run a subprocess command for one pipeline step; exit on failure."""
    print(f"\n{'=' * 60}")
    print(f"=== STEP {step}/{TOTAL}: {label} ===")
    print(f"{'=' * 60}")
    t0: float = time.time()
    result: subprocess.CompletedProcess = subprocess.run(cmd)
    elapsed: float = time.time() - t0
    if result.returncode != 0:
        print(
            f"\n[run_all] FAILED: step {step} exited with code {result.returncode}. "
            "Fix the error above and re-run."
        )
        sys.exit(result.returncode)
    print(f"[run_all] Step {step} done in {elapsed:.1f}s")


def main() -> None:
    """Execute the full DTNet reproducibility pipeline."""
    total_t0: float = time.time()

    _run(1, "Data Preprocessing",
         [PYTHON, "-m", "src.data.preprocess", RAW_CSV, "processed.csv"])

    _run(2, "Graph Construction",
         [PYTHON, "-c", _GRAPH_SCRIPT])

    _run(3, "Simulation Data Generation",
         [PYTHON, "-m", "src.simulation.generate_data"])

    _run(4, "GNN Training",
         [PYTHON, "-m", "src.gnn.train"])

    _run(5, "GNN Evaluation",
         [PYTHON, "-m", "src.gnn.evaluate"])

    _run(6, "Robustness Evaluation (5 seeds)",
         [PYTHON, "-m", "src.gnn.evaluate_robust"])

    print(f"\n{'=' * 60}")
    print(f"=== STEP 7/{TOTAL}: Thesis Figures ===")
    print(f"{'=' * 60}")
    t0: float = time.time()
    for module in _VIZ_MODULES:
        result = subprocess.run([PYTHON, "-m", module])
        if result.returncode != 0:
            print(f"\n[run_all] FAILED: {module} exited with code {result.returncode}.")
            sys.exit(result.returncode)
    print(f"[run_all] Step 7 done in {time.time() - t0:.1f}s")

    total_elapsed: float = time.time() - total_t0
    print(f"\n{'=' * 60}")
    print(f"[run_all] All {TOTAL} steps completed in {total_elapsed / 60:.1f} min.")
    print(f"[run_all] Results saved to results/")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
