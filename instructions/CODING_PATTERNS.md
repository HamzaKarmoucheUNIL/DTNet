# DTNet — Coding Patterns & Best Practices

## General Rules

1. **This is a research prototype.** Prioritize correctness and reproducibility over performance optimization.
2. **Every function must have a docstring** explaining what it does, its parameters, and what it returns.
3. **Use type hints everywhere.** This is a data science project — types prevent silent bugs.
4. **Set random seeds** (`np.random.seed(42)`, `torch.manual_seed(42)`) for reproducibility.
5. **Never hardcode magic numbers.** Use named constants or config dicts at the top of the file.

## Agent Design Patterns

### Base Agent Structure
Every digital twin agent must follow this pattern:

```python
@dataclass
class DigitalTwinAgent:
    node_id: str
    node_type: str  # 'supplier' | 'plant' | 'machine' | 'logistics' | 'distribution'

    # State variables (specific to node_type)
    capacity: float          # [0, 1] normalized
    throughput: float        # [0, 1] normalized
    failure_prob: float      # [0, 1]

    # Disruption state
    is_disrupted: bool = False
    disruption_severity: float = 0.0

    def compute_health_score(self) -> float:
        """Must return a float in [0, 1]. Higher = healthier."""
        ...

    def apply_disruption(self, severity: float, timestep: int):
        """Apply disruption. Must update is_disrupted, disruption_severity, and degrade state."""
        ...

    def step(self):
        """One simulation step. Called by Mesa scheduler."""
        ...
```

### Node-Type-Specific Attributes
Each node type has DIFFERENT attributes. Do NOT give sensor readings (temp, vibration) to suppliers or logistics nodes.

- **Machine nodes:** temp_bearing, temp_motor, vibration_h, vibration_v, oil_pressure, load_pct, power_kw, rpm, breakdown_flag
- **Supplier nodes:** delivery_reliability, lead_time_days, capacity, cost_per_unit, defect_rate
- **Plant nodes:** production_rate, quality_rate, overall_capacity, num_active_machines
- **Logistics nodes:** transit_time, warehouse_capacity, route_reliability, backlog
- **Distribution nodes:** demand_variability, fulfillment_rate, stock_level, delivery_delay

## Graph Patterns

### Always use DiGraph
```python
# CORRECT
G = nx.DiGraph()

# WRONG — supply chain flows are directional
G = nx.Graph()
```

### Node data structure
```python
G.add_node(node_id, 
    twin=agent_instance,    # The DigitalTwinAgent object
    layer='machine',        # One of: supplier, logistics, plant, machine, distribution
)
```

### Edge data structure
```python
G.add_edge(source, target,
    edge_type='material_flow',      # material_flow | operational | process_chain | shared_part_dependency
    flow_capacity=0.9,              # [0, 1]
    criticality_weight=0.8,         # [0, 1] — how critical this connection is
    latency_days=2,                 # How long disruption takes to propagate
)
```

### Graph construction must be deterministic
Given the same input data + seed, the graph must always be identical. Never introduce randomness in topology construction without a seed.

## Simulation Patterns

### Propagation formula
When a disrupted node propagates to a successor:
```python
incoming_severity = (
    parent.disruption_severity
    * edge.criticality_weight
    * propagation_decay          # global parameter, ~0.5-0.7
)

# Adjust for successor's own vulnerability
vulnerability = 1.0 - successor.compute_health_score()
adjusted = incoming_severity * (1 + vulnerability * 0.5)

if adjusted > threshold:  # global parameter, ~0.1-0.2
    successor.apply_disruption(adjusted, current_timestep)
```

### Simulation must collect data at every step
```python
# Every timestep, record:
{
    'timestep': t,
    'newly_disrupted': [...],        # nodes disrupted at this step
    'total_disrupted': [...],        # all disrupted nodes so far
    'network_health': float,         # average health score across all nodes
    'total_capacity': float,         # average capacity across all nodes
    'node_states': {node_id: {...}}, # full state snapshot for each node
}
```

## GNN Patterns

### Data conversion (NetworkX → PyG)
```python
from torch_geometric.utils import from_networkx

# Add node features as node attributes BEFORE converting
for node_id in G.nodes:
    twin = G.nodes[node_id]['twin']
    G.nodes[node_id]['x'] = [
        twin.capacity, twin.throughput, twin.failure_prob,
        twin.compute_health_score(), twin.disruption_severity,
        # ... other features
    ]
    G.nodes[node_id]['y'] = twin.disruption_severity  # target

pyg_data = from_networkx(G, group_node_attrs=['x'], group_edge_attrs=['criticality_weight'])
```

### Model architecture
```python
# Prefer GAT (Graph Attention Network) over GCN
# GAT learns WHICH connections matter most — perfect for supply chain
# where not all edges are equally critical

class DTNetGNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=4)
        self.conv2 = GATConv(hidden_channels * 4, hidden_channels, heads=1)
        self.lin = Linear(hidden_channels, out_channels)
    
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = F.dropout(x, p=0.3, training=self.training)
        x = self.conv2(x, edge_index).relu()
        return self.lin(x)
```

### Training data generation
Generate training data by running many simulations with different:
- Initial disruption nodes (random)
- Initial disruption severities (0.3 to 1.0)
- Number of simultaneous disruptions (1 to 3)

Each simulation snapshot becomes a training example:
- **Input (X):** node features at time t
- **Target (Y):** disruption severity at time t+1 (or final state)

## Visualization Patterns

- **Dark theme** for all plots (background `#0a0e17`, white text)
- **Color coding by layer:** supplier=blue, logistics=purple, plant=green, machine=amber, distribution=red
- **Disrupted nodes** always shown in red regardless of layer
- **Save all figures** at 200 DPI minimum
- **Every figure must have a title** that describes what is shown, not just "Figure 1"

## File Naming

- Python files: `snake_case.py`
- Classes: `PascalCase`
- Functions/variables: `snake_case`
- Constants: `UPPER_SNAKE_CASE`
- Output files: `dtnet_{description}_{date}.png`
