# CLAUDE.md — DTNet Project Instructions for Claude Code

## Project Overview
DTNet is a research prototype for a Master's thesis (HEC Lausanne, Data Science).
It models industrial supply chains as networks of interconnected digital twins and uses GNN + agent-based simulation to predict cascading disruptions.

**Read ALL files in the `instructions/` folder before starting any task.**

## Critical Rules

1. **Do ONLY what the prompt asks.** Do not modify, refactor, rename, or reorganize anything unless explicitly told to.
2. **Read `instructions/COMMON_MISTAKES.md` before every task.** It contains known failure patterns specific to this project.
3. **Every function needs a docstring.** Every variable needs a type hint.
4. **Set `np.random.seed(42)` and `torch.manual_seed(42)`** at the top of every script.
5. **No file should exceed 300 lines.** Split into modules if needed.
6. **Use `nx.DiGraph()`**, never `nx.Graph()`.
7. **Each node type has its own attributes.** Machines have sensors. Suppliers have lead times. Do NOT mix them.

## Tech Stack
- Graph: NetworkX
- Simulation: Mesa
- GNN: PyTorch Geometric (GAT preferred)
- Data: Pandas, NumPy
- Viz: Matplotlib (dark theme, `#0a0e17` background)
- Training: Google Colab for GPU if needed

## Project Structure
See `instructions/TECH_STACK.md` for full directory layout.

## Development Order
See `instructions/WORKFLOW.md`. Build in phases: Data → Graph → Simulation → GNN → Results.

## When In Doubt
- Check `instructions/CODING_PATTERNS.md` for code patterns
- Check `instructions/COMMON_MISTAKES.md` for things to avoid
- Check `instructions/PROJECT.md` for the big picture
- Ask for clarification rather than guessing
