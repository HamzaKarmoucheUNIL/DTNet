# scripts/gen_appendix_D.py
# Generates ready-to-paste LaTeX for Appendix D (node centralities + edge structure).
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import networkx as nx
import pandas as pd
from collections import Counter

# === 1. Obtain your DTNet graph G (a networkx.DiGraph) =======================
# Mirrors the pipeline in run_all.py exactly:
#   load raw CSV → preprocess → entity mappings → topology specs → build graph
import numpy as np
import torch

np.random.seed(42)
torch.manual_seed(42)

from src.data.loader import load_csv
from src.data.preprocess import preprocess
from src.data.entity_mapping import build_entity_mappings
from src.graph.topology import infer_topology
from src.graph.builder import build_graph

df_raw = load_csv("updated_data.csv")
df_c, _ = preprocess(df_raw)
em = build_entity_mappings(df_raw)
ns, es = infer_topology(em)
G = build_graph(ns, es, df_c)
# ===========================================================================

LAYER_ORDER = ["supplier", "logistics", "plant", "machine", "distribution"]

# --- centralities (same definitions as the GNN's structural features) ---
deg   = nx.degree_centrality(G)
betw  = nx.betweenness_centrality(G)
close = nx.closeness_centrality(G)
pr    = nx.pagerank(G)
ind   = dict(G.in_degree())
outd  = dict(G.out_degree())

def esc(s):  # escape special chars for LaTeX
    return (str(s).replace("&", r"\&").replace("_", r"\_")
                  .replace("%", r"\%").replace("#", r"\#"))

# ---------- D.1 : node centrality longtable ----------
rows = []
for layer in LAYER_ORDER:
    nodes = [n for n in G.nodes if G.nodes[n].get("layer") == layer]
    nodes.sort(key=lambda n: betw.get(n, 0), reverse=True)
    for n in nodes:
        rows.append((esc(n), layer.capitalize(), ind[n], outd[n],
                     deg[n], betw[n], close[n], pr[n]))

with open("appD_node_centrality.tex", "w") as f:
    f.write(r"""\footnotesize
\begin{longtable}{l l r r r r r r}
\caption{Structural centrality metrics for all 82 nodes.}\label{tab:appD-centrality}\\
\toprule
\textbf{Node} & \textbf{Layer} & \textbf{In} & \textbf{Out} & \textbf{Degree} & \textbf{Betw.} & \textbf{Close.} & \textbf{PageRank}\\
\midrule
\endfirsthead
\multicolumn{8}{l}{\small\itshape Table~\ref{tab:appD-centrality} (continued)}\\
\toprule
\textbf{Node} & \textbf{Layer} & \textbf{In} & \textbf{Out} & \textbf{Degree} & \textbf{Betw.} & \textbf{Close.} & \textbf{PageRank}\\
\midrule
\endhead
\midrule \multicolumn{8}{r}{\small\itshape continued on next page}\\
\endfoot
\bottomrule
\endlastfoot
""")
    for name, lay, i, o, d, b, c, p in rows:
        f.write(f"{name} & {lay} & {i} & {o} & {d:.3f} & {b:.3f} & {c:.3f} & {p:.3f}\\\\\n")
    f.write(r"\end{longtable}" + "\n")

# ---------- D.2 : edge structure by type ----------
agg = {}
for u, v, a in G.edges(data=True):
    et = a.get("edge_type", "unknown")
    lu, lv = G.nodes[u].get("layer"), G.nodes[v].get("layer")
    d = agg.setdefault(et, {"n": 0, "pairs": Counter(), "crit": 0.0, "flow": 0.0})
    d["n"] += 1
    d["pairs"][f"{lu}$\\to${lv}"] += 1
    d["crit"] += a.get("criticality_weight", 0.0)
    d["flow"] += a.get("flow_capacity", 0.0)

ORDER = ["material_flow", "operational", "process_chain", "shared_part_dependency"]
with open("appD_edge_structure.tex", "w") as f:
    f.write(r"""\small
\begin{tabularx}{\textwidth}{l X r c c}
\toprule
\textbf{Edge type} & \textbf{Connected layers} & \textbf{Count} & \textbf{Mean crit.} & \textbf{Mean flow}\\
\midrule
""")
    for et in ORDER + [k for k in agg if k not in ORDER]:
        if et not in agg: continue
        d = agg[et]
        pairs = ", ".join(p for p, _ in d["pairs"].most_common())
        f.write(f"\\texttt{{{esc(et)}}} & {pairs} & {d['n']} & "
                f"{d['crit']/d['n']:.2f} & {d['flow']/d['n']:.2f}\\\\\n")
    f.write(r"""\bottomrule
\end{tabularx}
""")

print("Wrote appD_node_centrality.tex and appD_edge_structure.tex")
print("Nodes:", G.number_of_nodes(), "| Edges:", G.number_of_edges())
