import gurobipy as gp
from gurobipy import GRB
import numpy as np
import pandas as pd
from SystemCharacteristics import get_fixed_data

# =============================================================================
# Load parameters
# =============================================================================
data = get_fixed_data()

T = range(data['num_timeslots'])   # {0, 1, ..., 9}
R = [0, 1]                         # two rooms

# Scalar parameters
P_bar    = data['heating_max_power']           # max heater power (kW)
P_vent   = data['ventilation_power']           # ventilation power (kW)
T_low    = data['temp_min_comfort_threshold']  # 18°C
T_ok     = data['temp_OK_threshold']           # 22°C
T_high   = data['temp_max_comfort_threshold']  # 26°C
H_high   = data['humidity_threshold']          # 70%
zeta_exch = data['heat_exchange_coeff']        # 0.6
zeta_loss = data['thermal_loss_coeff']         # 0.1
zeta_conv = data['heating_efficiency_coeff']   # 1.0
zeta_cool = data['heat_vent_coeff']            # 0.7
zeta_occ  = data['heat_occupancy_coeff']       # 0.02
eta_occ   = data['humidity_occupancy_coeff']   # 0.18
eta_vent  = data['humidity_vent_coeff']        # 15
T_out     = data['outdoor_temperature']        # list of 10 values
T_init    = data['initial_temperature']        # 21°C
T_init_prev = data['previous_initial_temperature']  # 21°C (used for t=-1 in dynamics)
H_init    = data['initial_humidity']           # 40%
min_up    = data['vent_min_up_time']           # 3

M_pow  = P_bar

# =============================================================================
# Load historical data  (100 days x 10 hours)
# =============================================================================
price_data = pd.read_csv("PriceData.csv",       header=0, index_col=None).values  # (100, 10)
occ_r1     = pd.read_csv("OccupancyRoom1.csv",  header=0, index_col=None).values  # (100, 10)
occ_r2     = pd.read_csv("OccupancyRoom2.csv",  header=0, index_col=None).values  # (100, 10)

num_days = price_data.shape[0]

# =============================================================================
# Function: solve MILP for a single day
# =============================================================================
def solve_day(price, occ):
    """
    price : array of shape (10,)  — electricity price at each hour
    occ   : array of shape (2,10) — occupancy[room, hour]
    Returns the optimal cost, or None if infeasible.
    """
    m = gp.Model("task1_milp")
    m.Params.OutputFlag = 0  # suppress solver output

    # -------------------------------------------------------------------------
    # Decision variables
    # -------------------------------------------------------------------------
    p      = m.addVars(R, T, lb=0, ub=P_bar, vtype=GRB.CONTINUOUS, name="p")
    v      = m.addVars(T, vtype=GRB.BINARY,     name="v")
    Ltemp  = m.addVars(R, T, vtype=GRB.BINARY,  name="Ltemp")
    Htemp  = m.addVars(R, T, vtype=GRB.BINARY,  name="Htemp")
    temp   = m.addVars(R, T, vtype=GRB.CONTINUOUS, name="temp")
    hum    = m.addVars(T, lb=0, vtype=GRB.CONTINUOUS, name="hum")
    z_trig = m.addVars(R, range(1, len(T)), vtype=GRB.BINARY, name="z_trig")
    z_rec  = m.addVars(R, range(1, len(T)), vtype=GRB.BINARY, name="z_rec")

    # -------------------------------------------------------------------------
    # Objective: minimise total electricity cost
    # -------------------------------------------------------------------------
    m.setObjective(
        gp.quicksum(p[r, t] * price[t] for r in R for t in T)
        + gp.quicksum(P_vent * v[t] * price[t] for t in T),
        GRB.MINIMIZE
    )

    # -------------------------------------------------------------------------
    # Temperature dynamics
    # -------------------------------------------------------------------------
    for r in R:
        r_other = 1 - r
        for t in T:
            if t == 0:
                # Initial condition
                m.addConstr(temp[r, 0] == T_init, name=f"temp_init_{r}")
            else:
                m.addConstr(
                    temp[r, t] == temp[r, t-1]
                    + zeta_exch * (temp[r_other, t-1] - temp[r, t-1])
                    + zeta_loss * (T_out[t-1] - temp[r, t-1])
                    + zeta_conv * p[r, t-1]
                    - zeta_cool * v[t-1]
                    + zeta_occ  * occ[r, t-1],
                    name=f"temp_dyn_{r}_{t}"
                )

    # -------------------------------------------------------------------------
    # Humidity dynamics
    # -------------------------------------------------------------------------
    for t in T:
        if t == 0:
            m.addConstr(hum[0] == H_init, name="hum_init")
        else:
            m.addConstr(
                hum[t] == hum[t-1]
                + eta_occ  * gp.quicksum(occ[r, t-1] for r in R)
                - eta_vent * v[t-1],
                name=f"hum_dyn_{t}"
            )

    # -------------------------------------------------------------------------
    # Low temperature overrule controller  (Ltemp=1 ↔ temp <= T_low, off when temp >= T_ok)
    # -------------------------------------------------------------------------
    for r in R:
        for t in T:
            if t == 0:
                # At t=0: activate if temp <= T_low
                m.addGenConstrIndicator(Ltemp[r,0], 1, temp[r,0] <= T_low, name=f"Ltemp_if1_{r}_0")
                m.addGenConstrIndicator(Ltemp[r,0], 0, temp[r,0] >= T_low + 0.01, name=f"Ltemp_if0_{r}_0")
            else:
                # z_trig[r,t] = 1 if temp[r,t] <= T_low
                m.addGenConstrIndicator(z_trig[r,t], 1, temp[r,t] <= T_low,        name=f"ztrig_if1_{r}_{t}")
                m.addGenConstrIndicator(z_trig[r,t], 0, temp[r,t] >= T_low + 0.01, name=f"ztrig_if0_{r}_{t}")

                # z_rec[r,t] = 1 if temp[r,t] >= T_ok
                m.addGenConstrIndicator(z_rec[r,t], 1, temp[r,t] >= T_ok,        name=f"zrec_if1_{r}_{t}")
                m.addGenConstrIndicator(z_rec[r,t], 0, temp[r,t] <= T_ok - 0.01, name=f"zrec_if0_{r}_{t}")

                m.addConstr(Ltemp[r,t] >= z_trig[r,t],                          name=f"Ltemp_trig_{r}_{t}")
                m.addConstr(Ltemp[r,t] >= Ltemp[r,t-1] - z_rec[r,t],           name=f"Ltemp_pers_{r}_{t}")
                m.addConstr(Ltemp[r,t] <= z_trig[r,t] + Ltemp[r,t-1],          name=f"Ltemp_ub1_{r}_{t}")
                m.addConstr(Ltemp[r,t] <= 1 - z_rec[r,t] + z_trig[r,t],        name=f"Ltemp_ub2_{r}_{t}")

            # When active: heater at max power
            m.addGenConstrIndicator(Ltemp[r,t], 1, p[r,t] == P_bar, name=f"Ltemp_p_{r}_{t}")

    # -------------------------------------------------------------------------
    # High temperature overrule (unchanged - single timestep)
    # -------------------------------------------------------------------------
    for r in R:
        for t in T:
            m.addGenConstrIndicator(Htemp[r,t], 1, temp[r,t] >= T_high,        name=f"Htemp_if1_{r}_{t}")
            m.addGenConstrIndicator(Htemp[r,t], 0, temp[r,t] <= T_high - 0.01, name=f"Htemp_if0_{r}_{t}")
            m.addGenConstrIndicator(Htemp[r,t], 1, p[r,t] == 0,                name=f"Htemp_p_{r}_{t}")

    # -------------------------------------------------------------------------
    # Mutual exclusion: can't be too cold AND too hot simultaneously
    # -------------------------------------------------------------------------
    for r in R:
        for t in T:
            m.addConstr(Ltemp[r,t] + Htemp[r,t] <= 1, name=f"excl_{r}_{t}")

    # -------------------------------------------------------------------------
    # Humidity overrule controller  (unchanged)
    # -------------------------------------------------------------------------
    for t in T:
        m.addGenConstrIndicator(v[t], 0, hum[t] <= H_high, name=f"hum_overrule_{t}")

    # -------------------------------------------------------------------------
    # Ventilation minimum up-time (eq. 15)
    # -------------------------------------------------------------------------
    for t in T:
        if t == 0:
            continue  # no t-1 available
        m.addConstr(
            gp.quicksum(v[t2] for t2 in range(t, min(t + min_up, len(T))))
            >= min_up * (v[t] - v[t-1]),
            name=f"min_uptime_{t}"
        )

    # -------------------------------------------------------------------------
    # Solve
    # -------------------------------------------------------------------------
    m.optimize()

    if m.Status == GRB.INFEASIBLE:
        m.computeIIS()
        m.write("infeasible.ilp")

    if m.Status == GRB.OPTIMAL:
        results = {
            'cost':     m.ObjVal,
            'Temp_r1':  [temp[0, t].X for t in T],
            'Temp_r2':  [temp[1, t].X for t in T],
            'h_r1':     [p[0, t].X for t in T],
            'h_r2':     [p[1, t].X for t in T],
            'v':        [v[t].X for t in T],
            'Hum':      [hum[t].X for t in T],
            'Ltemp_r1': [Ltemp[0, t].X for t in T],  
            'Ltemp_r2': [Ltemp[1, t].X for t in T],  
            'Htemp_r1': [Htemp[0, t].X for t in T],  
            'Htemp_r2': [Htemp[1, t].X for t in T],
            'z_trig_r1': [z_trig[0, t].X for t in range(1, len(T))],
            'z_trig_r2': [z_trig[1, t].X for t in range(1, len(T))],
            'z_rec_r1':  [z_rec[0, t].X for t in range(1, len(T))],
            'z_rec_r2':  [z_rec[1, t].X for t in range(1, len(T))],
        }
        return results
    else:
        return None
    
    


# =============================================================================
# Solve for all 100 days
# =============================================================================
costs = []
all_results = {}
for day in range(num_days):
    price = price_data[day]                        # shape (10,)
    occ   = np.array([occ_r1[day], occ_r2[day]])  # shape (2, 10)
    res   = solve_day(price, occ)
    if res is not None:
        costs.append(res['cost'])
        all_results[day] = res
        print(f"Day {day+1:3d}: cost = {res['cost']:.4f}")
    else:
        print(f"Day {day+1:3d}: INFEASIBLE")

print(f"\nAverage daily electricity cost: {np.mean(costs):.4f}")

# =============================================================================
# Plot example days
# =============================================================================
from PlotsRestaurant import plot_HVAC_results

for day in range(5):   # change to any two days you like
    price = price_data[day]
    occ   = np.array([occ_r1[day], occ_r2[day]])
    res   = all_results[day]
    res['price']   = list(price)
    res['Occ_r1']  = list(occ[0])
    res['Occ_r2']  = list(occ[1])
    plot_HVAC_results(res, day=day)
