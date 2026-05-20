# Decision-Making-Under-Uncertainity

## Requirements
Install dependencies with:
```
pip install -r requirements.txt
```
> **Note:** `gurobipy` requires a valid Gurobi license.

## Assignment 1 – MILP Optimization
Mixed-integer linear programming model for heating and ventilation system.

## Files
- `task1_milp.py` – MILP model implementation
- `SystemCharacteristics.py` – System parameters and data
- `PlotsRestaurant.py` – Visualization scripts
- `PriceData.csv`, `OccupancyRoom1.csv`, `OccupancyRoom2.csv` – Input data

## Part B – Decision-Making Under Uncertainty

Online policies evaluated over 100 days using `v2_PriceData.csv` and occupancy data.

**Policies:**
- `Optimal_in_Hindsight_02.py` — lower bound via full-day MILP with perfect information
- `SP_policy_02.py` — multi-stage stochastic programming (k-medoids scenario tree)
- `EV_policy_02.py` — expected value (deterministic lookahead)
- `two_stage_SP_policy_02.py` — two-stage stochastic programming
- `ADP_policy_02.py` — approximate dynamic programming with linear value function
- `Hybrid_policy_02.py` — hybrid SP + ADP policy

**Run the full comparison:**
```bash
cd PartB
python environment.py
```

**Input data:**
- `v2_PriceData.csv` — electricity prices (column 0: initial `price_previous`, columns 1–10: hourly prices)
- `OccupancyRoom1.csv`, `OccupancyRoom2.csv` — room occupancy
