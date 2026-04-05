# DTNet — Common Mistakes to Avoid

## CRITICAL: Read This Before Every Task

This file lists mistakes that AI coding assistants commonly make on this project.
**Reference this file in every prompt.**

---

## Mistake #1: Giving sensor attributes to non-machine nodes

**Wrong:** A supplier node with `temp_bearing=45.0` and `vibration_h=2.0`
**Right:** A supplier node with `delivery_reliability=0.92` and `lead_time_days=5`

Each node type has its OWN attributes. Only machine nodes have sensor readings from the Kaggle dataset.

---

## Mistake #2: Using undirected graph

**Wrong:** `G = nx.Graph()`
**Right:** `G = nx.DiGraph()`

Supply chain flows have direction. A supplier feeds a plant, not the other way around.

---

## Mistake #3: Forgetting random seeds

**Wrong:** `np.random.uniform(0.8, 1.0)` without setting seed
**Right:** `np.random.seed(42)` at the top of every script/notebook

This is research. Results must be reproducible.

---

## Mistake #4: Modifying things that were not asked for

**DO NOT** change existing functions, class structures, variable names, or file organization unless explicitly asked to. If you think something should be refactored, ASK FIRST. Do ONLY what is requested in the prompt.

---

## Mistake #5: Hardcoding values inside functions

**Wrong:**
```python
def simulate(G):
    decay = 0.65  # hardcoded inside
    threshold = 0.12
```

**Right:**
```python
DEFAULT_DECAY = 0.65
DEFAULT_THRESHOLD = 0.12

def simulate(G, propagation_decay=DEFAULT_DECAY, threshold=DEFAULT_THRESHOLD):
```

All parameters must be configurable with sensible defaults.

---

## Mistake #6: Not collecting simulation data properly

Every simulation step MUST record the full state. The GNN training data comes from these records. If you skip data collection, the entire GNN pipeline breaks.

Always collect: timestep, newly_disrupted, total_disrupted, network_health, total_capacity, per-node states.

---

## Mistake #7: Converting NetworkX to PyG incorrectly

**Wrong:** Converting the graph without first setting `x` (features) and `y` (targets) as node attributes.

**Right:**
```python
# FIRST set features on every node
for n in G.nodes:
    G.nodes[n]['x'] = [feature1, feature2, ...]
    G.nodes[n]['y'] = target_value

# THEN convert
pyg_data = from_networkx(G, group_node_attrs=['x'], group_edge_attrs=[...])
```

---

## Mistake #8: Training GNN on a single simulation

The GNN needs thousands of diverse simulations to learn properly. Generate at least 5,000-10,000 simulation runs with varied:
- Starting disruption nodes (random)
- Severity levels (0.3 to 1.0)
- Multiple simultaneous failures

---

## Mistake #9: Overcomplicated visualization code

Keep visualizations clean and readable. Do NOT:
- Add 3D effects or unnecessary animations
- Use more than 5-6 colors
- Create plots with tiny unreadable text
- Forget axis labels, titles, or legends

DO:
- Use the dark theme consistently (#0a0e17 background)
- Use the established color coding (blue=supplier, purple=logistics, green=plant, amber=machine, red=distribution)
- Save at 200 DPI minimum
- Add clear, descriptive titles

---

## Mistake #10: Mixing simulation state across runs

**Wrong:** Running a second simulation on a graph that was already mutated by the first simulation.

**Right:** Either reset all node states before each run, or rebuild the graph from scratch.

```python
# ALWAYS reset or rebuild before a new simulation
G_fresh = build_dtnet_graph()  # rebuild
history = simulate_cascading_failure(G_fresh, ...)
```

---

## Mistake #11: Ignoring edge attributes in propagation

The cascading failure must use edge `criticality_weight` and `latency_days`. A disruption through a low-criticality edge should propagate less than through a high-criticality one. Do not treat all edges the same.

---

## Mistake #12: Printing instead of logging

**Wrong:** `print("Building graph...")`
**Right:** Use Python's `logging` module or at minimum structured print statements with consistent format:
```python
print(f"[{step}/{total}] Building graph... nodes={G.number_of_nodes()}")
```

---

## Mistake #13: Giant monolithic files

No single Python file should exceed 300 lines. If it does, split it into modules following the project structure in TECH_STACK.md. Each file should have ONE clear responsibility.

---

## Mistake #14: Not handling the isolated baseline properly

The isolated baseline (for RQ3 comparison) must use the EXACT SAME initial disruption as the networked model. The only difference is: isolated twins don't propagate. If you change anything else, the comparison is unfair.

---

## Reminder

**Before writing any code, re-read PROJECT.md to understand the context.**
**Do ONLY what the prompt asks. Nothing more, nothing less.**
