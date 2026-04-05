# DTNet — Development Workflow with Claude Code

## How to Use These Instruction Files

Before starting any coding session with Claude Code, say:

```
Read the files in /instructions/ folder before starting. 
Pay special attention to COMMON_MISTAKES.md.
```

This ensures Claude Code has the full context of the project.

## Development Phases

Build the project in this exact order. Do NOT skip ahead.

### Phase 1: Data Loading & Exploration
**Goal:** Load the Kaggle dataset and understand its structure.

Tasks (do them one at a time, one prompt per task):
1. Write `src/data/loader.py` — load the CSV, basic info (shape, dtypes, null counts)
2. Write `src/data/preprocess.py` — clean data, normalize sensor columns, encode categoricals
3. Create `notebooks/01_data_exploration.ipynb` — EDA: distributions, correlations, breakdown patterns
4. Identify the entity structure: how many plants, machines per plant, parts per machine

**Checkpoint:** You should know exactly how many unique plant_codes, asset_tags, and part_nos exist, and how they relate to each other.

### Phase 2: Graph Construction
**Goal:** Build G = (V, E) from the dataset's entity structure.

Tasks:
1. Write `src/graph/topology.py` — infer graph topology from machine-part-plant relationships
2. Write `src/graph/builder.py` — construct the full NetworkX DiGraph with proper node/edge attributes
3. Write `src/graph/metrics.py` — compute graph metrics (degree centrality, betweenness, vulnerability)
4. Create `notebooks/02_graph_construction.ipynb` — visualize the graph, validate topology

**Checkpoint:** You have a reproducible graph with correct node types, edge types, and attributes.

### Phase 3: Agent-Based Simulation
**Goal:** Simulate cascading failures using Mesa.

Tasks:
1. Write `src/agents/base_agent.py` — base DigitalTwinAgent class
2. Write the 5 specialized agent files (machine, supplier, plant, logistics, distribution)
3. Write `src/simulation/model.py` — DTNetModel that runs on the graph
4. Write `src/simulation/scenarios.py` — predefined disruption scenarios
5. Run simulations and collect data: `notebooks/03_simulation_runs.ipynb`
6. Generate 5,000+ simulation runs for GNN training data

**Checkpoint:** You can run a simulation, it produces correct cascading behavior, and you have thousands of recorded runs.

### Phase 4: GNN Training
**Goal:** Train a GNN to predict disruption propagation.

Tasks:
1. Write `src/gnn/dataset.py` — convert simulation records to PyG dataset
2. Write `src/gnn/model.py` — GATConv-based architecture
3. Write `src/gnn/train.py` — training loop with early stopping
4. Write `src/gnn/evaluate.py` — evaluation metrics + comparison with isolated baseline
5. Train and evaluate: `notebooks/04_gnn_training.ipynb`

**Checkpoint:** GNN achieves better disruption prediction than the isolated baseline.

### Phase 5: Results & Comparison
**Goal:** Produce the final comparison (RQ3) and thesis-ready visualizations.

Tasks:
1. Write `src/viz/comparison_viz.py` — networked vs isolated comparison plots
2. Create `notebooks/05_results_comparison.ipynb` — full results analysis
3. Generate all thesis figures
4. Write a results summary

**Checkpoint:** You have clear, reproducible evidence that networked twins outperform isolated ones.

## Prompting Rules for Claude Code

### Rule 1: One task per prompt
```
# WRONG
"Build the entire simulation engine with Mesa, add all 5 agent types, 
create scenarios, and generate training data"

# RIGHT
"Write src/agents/base_agent.py — the base DigitalTwinAgent dataclass 
with state variables, health_score computation, and disruption logic. 
See CODING_PATTERNS.md for the expected structure."
```

### Rule 2: Always reference instruction files
```
# GOOD PROMPT
"Read instructions/CODING_PATTERNS.md and instructions/COMMON_MISTAKES.md, 
then write src/graph/builder.py following the graph patterns described there."
```

### Rule 3: Provide file context for edits
```
# GOOD PROMPT
"Look at src/agents/base_agent.py and src/graph/builder.py, then write 
src/simulation/model.py that uses both. The model should..."
```

### Rule 4: Specify what NOT to change
```
# GOOD PROMPT
"Add a new method `compute_vulnerability()` to the MachineAgent class 
in src/agents/machine_agent.py. Do NOT modify any existing methods."
```

### Rule 5: When debugging, give the error + context
```
# GOOD PROMPT
"I'm getting this error when running src/gnn/train.py:
[paste error]
The relevant files are src/gnn/model.py and src/gnn/dataset.py.
Find the bug and fix it. Do not change anything else."
```

### Rule 6: Start new chat when context gets long
If you've been in the same Claude Code chat for 10+ prompts, start a new one. Briefly summarize where you are:
```
"I'm building DTNet (see instructions/ folder). I've completed Phase 1-2. 
Currently working on Phase 3 — agent-based simulation. 
I have base_agent.py and machine_agent.py done. 
Next: write supplier_agent.py. See CODING_PATTERNS.md for node attributes."
```

## Git Strategy

- **Commit after each completed task** (not after each prompt)
- **Branch per phase:** `phase-1-data`, `phase-2-graph`, `phase-3-simulation`, `phase-4-gnn`, `phase-5-results`
- **Commit message format:** `[phase-N] brief description`
  - Example: `[phase-2] build graph topology from dataset entity structure`
- **Tag milestones:** `v0.1-graph`, `v0.2-simulation`, `v0.3-gnn`, `v1.0-thesis`

## Quality Checks

After each phase, run these checks:

1. **Does it run without errors?** `python -m pytest tests/`
2. **Is it reproducible?** Run twice with same seed → same results?
3. **Are results logged?** Can you reconstruct what happened from the output?
4. **Is it documented?** Would someone else understand the code without you explaining it?
5. **Is it committed?** `git status` should be clean.
