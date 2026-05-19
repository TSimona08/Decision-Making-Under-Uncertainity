# -*- coding: utf-8 -*-
"""
Created on Mon Nov 17 11:14:31 2025

@author: geots
"""

# The state will be provided by the environment as the following dictionary

# state = {
#     "T1": ..., #Temperature of room 1
#     "T2": ..., #Temperature of room 2
#     "H": ..., #Humidity
#     "Occ1": ..., #Occupancy of room 1
#     "Occ2": ..., #Occupancy of room 2
#     "price_t": ..., #Price
#     "price_previous": ..., #Previous Price
#     "vent_counter": ..., #For how many consecutive hours has the ventilation been on 
#     "low_override_r1": ..., #Is the low-temperature overrule controller of room 1 active 
#     "low_override_r2": ..., #Is the low-temperature overrule controller of room 2 active 
#     "current_time": ... #What is the hour of the day
# }

import numpy as np
import pyomo.environ as pyo

from v2_SystemCharacteristics import get_fixed_data
from PriceProcessRestaurant import price_model
from OccupancyProcessRestaurant import next_occupancy_levels
from pyomo.opt import SolverFactory

_ADP_SOLVER = SolverFactory('gurobi')
_ADP_SOLVER.options['OutputFlag'] = 0
_ADP_SOLVER.options['TimeLimit']  = 5

def select_action(state):

    ### Here goes your code
    # ------------------------------------------------------------------
    # System parameters
    # ------------------------------------------------------------------
    params    = get_fixed_data()
    T_SLOTS   = params['num_timeslots']
    P_MAX     = params['heating_max_power']
    T_LOW     = params['temp_min_comfort_threshold']
    T_HIGH    = params['temp_max_comfort_threshold']
    H_HIGH    = params['humidity_threshold']
    VENT_MIN  = params['vent_min_up_time']
    T_OUT     = params['outdoor_temperature']
    ZETA_CONV = params['heating_efficiency_coeff']
    ZETA_LOSS = params['thermal_loss_coeff']
    ZETA_EXCH = params['heat_exchange_coeff']
    ZETA_COOL = params['heat_vent_coeff']
    ZETA_OCC  = params['heat_occupancy_coeff']
    ETA_OCC   = params['humidity_occupancy_coeff']
    ETA_VENT  = params['humidity_vent_coeff']
    P_VENT    = params['ventilation_power']
    M_BIG     = 200.0
 
    # ------------------------------------------------------------------
    # Pre-trained weights eta[t], shape (T x 8).
    # Row t+1 is used when deciding at hour t.
    # eta[T-1] = zeros (terminal hour: no future value).
    #
    # Feature vector phi(x_{t+1}) = [
    #   [0]  1                      bias
    #   [1]  T1_{t+1}               exact physics
    #   [2]  T2_{t+1}               exact physics
    #   [3]  H_{t+1}                exact physics
    #   [4]  E[lambda_{t+1}]        Monte Carlo from price_model()
    #   [5]  E[Occ1_{t+1}]          Monte Carlo from next_occupancy_levels()
    #   [6]  E[Occ2_{t+1}]          Monte Carlo from next_occupancy_levels()
    #   [7]  c'_{t+1}               min(c_t+1, U^vent)*v — linear in v
    #
    # ------------------------------------------------------------------
    eta = np.array([[ 1.40381742e+02, -4.40269607e+00, -4.36370499e+00,
         8.78759337e-01,  3.53734361e+01,  4.38284981e-01,
        -1.57397279e-01,  1.82022587e+00],
       [ 1.47219200e+02, -4.66917458e+00, -4.74358292e+00,
         8.57355518e-01,  3.26947409e+01,  3.50045201e-01,
        -2.21717608e-01,  1.07915755e-01],
       [ 1.31047062e+02, -4.34678059e+00, -4.19028690e+00,
         8.26636321e-01,  2.86881582e+01,  2.21443559e-01,
        -2.59414146e-01,  6.47143452e-01],
       [ 1.19978324e+02, -3.62157459e+00, -3.91431975e+00,
         7.68491724e-01,  2.38595676e+01,  1.23778761e-01,
        -3.01083216e-01,  2.13179051e-01],
       [ 1.30062069e+02, -4.43840317e+00, -3.11057810e+00,
         6.51174833e-01,  1.77801874e+01,  1.01546815e-01,
        -2.25706816e-01,  1.48994586e+00],
       [ 1.19660821e+02, -2.81275964e+00, -3.72707061e+00,
         5.07062033e-01,  1.14790622e+01,  1.09799743e-01,
        -1.11290553e-01,  1.75213260e+00],
       [ 8.25971293e+01, -2.13576240e+00, -2.51835222e+00,
         3.71077100e-01,  7.18435766e+00,  1.00314722e-01,
         1.49143411e-03,  1.47051554e+00],
       [ 5.86231606e+01, -1.94186589e+00, -1.55353946e+00,
         2.63091636e-01,  4.46267686e+00,  4.47304278e-02,
         1.04848970e-01,  8.11251678e-01],
       [ 4.68762667e+01, -9.66805097e-01, -1.70642036e+00,
         1.38154188e-01,  2.26862849e+00,  3.01543202e-02,
         2.23281222e-02,  6.52422489e-01],
       [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
         0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
         0.00000000e+00,  0.00000000e+00]])
 
    # ------------------------------------------------------------------
    # Extract state variables
    # ------------------------------------------------------------------
    t            = int(state['current_time'])
    T1           = float(state['T1'])
    T2           = float(state['T2'])
    H            = float(state['H'])
    Occ1         = float(state['Occ1'])
    Occ2         = float(state['Occ2'])
    price        = float(state['price_t'])
    lam_prev     = float(state['price_previous'])
    vent_counter = int(state['vent_counter'])
    lo_r1        = int(state['low_override_r1'])
    lo_r2        = int(state['low_override_r2'])
    T_outdoor    = T_OUT[t]
 
    # ------------------------------------------------------------------
    # Conditional expectations for phi(x_{t+1})[4],[5],[6]
    #
    # Average N_EXP draws from the process models to estimate:
    #   E[lambda_{t+1} | lambda_t, lambda_{t-1}]
    #   E[Occ1_{t+1}   | Occ1_t, Occ2_t]
    #   E[Occ2_{t+1}   | Occ1_t, Occ2_t]
    #
    # ------------------------------------------------------------------
    N_EXP  = 50
    prices = [price_model(price, lam_prev)     for _ in range(N_EXP)]
    occs   = [next_occupancy_levels(Occ1, Occ2) for _ in range(N_EXP)]
 
    lambda_exp = float(np.mean(prices))
    occ1_exp   = float(np.mean([o[0] for o in occs]))
    occ2_exp   = float(np.mean([o[1] for o in occs]))
 
    # ------------------------------------------------------------------
    # Ventilation counter for next step — phi element [7]
    # c'_{t+1} = min(c_t + 1, U^vent) * v  — linear in v
    # ------------------------------------------------------------------
    c_next_coeff = float(min(vent_counter + 1, VENT_MIN))
 
    # ------------------------------------------------------------------
    # Build MILP
    # ------------------------------------------------------------------
    m = pyo.ConcreteModel()
 
    # Decision variables
    m.p1 = pyo.Var(within=pyo.NonNegativeReals, bounds=(0, P_MAX))
    m.p2 = pyo.Var(within=pyo.NonNegativeReals, bounds=(0, P_MAX))
    m.v  = pyo.Var(within=pyo.Binary)
 
    # Next-state variables — phi elements [1],[2],[3]
    m.T1_next = pyo.Var(within=pyo.Reals)
    m.T2_next = pyo.Var(within=pyo.Reals)
    m.H_next  = pyo.Var(within=pyo.NonNegativeReals, bounds=(0, 100))
 
    # Binary indicators for overrule controllers
    m.hi_r1     = pyo.Var(within=pyo.Binary)
    m.hi_r2     = pyo.Var(within=pyo.Binary)
    m.lo_ind_r1 = pyo.Var(within=pyo.Binary)
    m.lo_ind_r2 = pyo.Var(within=pyo.Binary)
 
    # ------------------------------------------------------------------
    # Physics constraints — x_{t+1} = f(x_t, u_t)
    # ------------------------------------------------------------------
    m.T1_dyn = pyo.Constraint(expr=
        m.T1_next == T1
                   + ZETA_CONV * m.p1
                   - ZETA_LOSS * (T1 - T_outdoor)
                   + ZETA_EXCH * (T2 - T1)
                   - ZETA_COOL * m.v
                   + ZETA_OCC  * Occ1)
 
    m.T2_dyn = pyo.Constraint(expr=
        m.T2_next == T2
                   + ZETA_CONV * m.p2
                   - ZETA_LOSS * (T2 - T_outdoor)
                   + ZETA_EXCH * (T1 - T2)
                   - ZETA_COOL * m.v
                   + ZETA_OCC  * Occ2)
 
    m.H_dyn = pyo.Constraint(expr=
        m.H_next == H + ETA_OCC * (Occ1 + Occ2) - ETA_VENT * m.v)
 
    # ------------------------------------------------------------------
    # Overrule constraints (big-M linearisation)
    # HIGH temperature: T_r >= T_HIGH  =>  p_r = 0
    # ------------------------------------------------------------------
    # LOW temperature overrule
    if lo_r1:
        m.lo_ind_r1.fix(1)
        m.hi_r1.fix(0)
        m.lo_p1 = pyo.Constraint(expr= m.p1 == P_MAX)
    else:
        m.hi_r1_ub = pyo.Constraint(expr= T1 - T_HIGH <= M_BIG * m.hi_r1)
        m.hi_r1_lb = pyo.Constraint(expr= T1 - T_HIGH >= -M_BIG * (1 - m.hi_r1))
        m.hi_p1    = pyo.Constraint(expr= m.p1 <= P_MAX * (1 - m.hi_r1))
        m.lo_r1_ub = pyo.Constraint(expr= T_LOW - T1 <= M_BIG * m.lo_ind_r1)
        m.lo_r1_lb = pyo.Constraint(expr= T_LOW - T1 >= -M_BIG * (1 - m.lo_ind_r1))
        m.lo_p1    = pyo.Constraint(expr= m.p1 >= P_MAX * m.lo_ind_r1)
        m.excl_r1  = pyo.Constraint(expr= m.lo_ind_r1 + m.hi_r1 <= 1)

    if lo_r2:
        m.lo_ind_r2.fix(1)
        m.hi_r2.fix(0)
        m.lo_p2 = pyo.Constraint(expr= m.p2 == P_MAX)
    else:
        m.hi_r2_ub = pyo.Constraint(expr= T2 - T_HIGH <= M_BIG * m.hi_r2)
        m.hi_r2_lb = pyo.Constraint(expr= T2 - T_HIGH >= -M_BIG * (1 - m.hi_r2))
        m.hi_p2    = pyo.Constraint(expr= m.p2 <= P_MAX * (1 - m.hi_r2))
        m.lo_r2_ub = pyo.Constraint(expr= T_LOW - T2 <= M_BIG * m.lo_ind_r2)
        m.lo_r2_lb = pyo.Constraint(expr= T_LOW - T2 >= -M_BIG * (1 - m.lo_ind_r2))
        m.lo_p2    = pyo.Constraint(expr= m.p2 >= P_MAX * m.lo_ind_r2)
        m.excl_r2  = pyo.Constraint(expr= m.lo_ind_r2 + m.hi_r2 <= 1)
 
    # Ventilation inertia: counter in {1,2} forces v=1
    if vent_counter in (1, 2):
        m.vent_inertia = pyo.Constraint(expr= m.v == 1)
 
    # Humidity overrule: H >= H_HIGH forces v=1
    m.hum_overrule = pyo.Constraint(expr= H - H_HIGH <= M_BIG * m.v)
 
    # ------------------------------------------------------------------
    # Objective
    #
    # min  lambda_t*(p1 + p2 + P_vent*v)     [immediate cost — exact]
    #    + eta[t+1] @ phi(x_{t+1})           [future cost — VFA]
    #
    # phi(x_{t+1}):
    #   [0]  1             constant
    #   [1]  T1_next       Pyomo variable (T1_dyn)
    #   [2]  T2_next       Pyomo variable (T2_dyn)
    #   [3]  H_next        Pyomo variable (H_dyn)
    #   [4]  lambda_exp    constant — E[lambda_{t+1}]
    #   [5]  occ1_exp      constant — E[Occ1_{t+1}]
    #   [6]  occ2_exp      constant — E[Occ2_{t+1}]
    #   [7]  c_next*v      linear in v
    # ------------------------------------------------------------------
    if t < T_SLOTS - 1:
        th = eta[t + 1]
 
        future_cost = (
            th[0]                            # [0] bias
          + th[1] * m.T1_next               # [1] T1_{t+1}
          + th[2] * m.T2_next               # [2] T2_{t+1}
          + th[3] * m.H_next                # [3] H_{t+1}
          + th[4] * lambda_exp              # [4] E[lambda_{t+1}]
          + th[5] * occ1_exp                # [5] E[Occ1_{t+1}]
          + th[6] * occ2_exp                # [6] E[Occ2_{t+1}]
          + th[7] * c_next_coeff * m.v      # [7] c'_{t+1}
        )
    else:
        future_cost = 0.0
 
    m.obj = pyo.Objective(
        expr  = price * (m.p1 + m.p2 + P_VENT * m.v) + future_cost,
        sense = pyo.minimize,
    )
 
    # ------------------------------------------------------------------
    # Solve
    # ------------------------------------------------------------------
    result = _ADP_SOLVER.solve(m, tee=False)
 
    # ------------------------------------------------------------------
    # Extract solution — fallback respects all overrule constraints
    # ------------------------------------------------------------------
    if result.solver.termination_condition == pyo.TerminationCondition.optimal:
        HereAndNowActions = {
            'HeatPowerRoom1': float(pyo.value(m.p1)),
            'HeatPowerRoom2': float(pyo.value(m.p2)),
            'VentilationON':  int(round(pyo.value(m.v))),
        }
    else:
        HereAndNowActions = {
            'HeatPowerRoom1': P_MAX if lo_r1 else 0.0,
            'HeatPowerRoom2': P_MAX if lo_r2 else 0.0,
            'VentilationON':  1 if (vent_counter in (1, 2)
                                    or H >= H_HIGH) else 0,
        }
 
    return HereAndNowActions