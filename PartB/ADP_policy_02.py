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
import os
from v2_SystemCharacteristics import get_fixed_data


def select_action(state):

    ### Here goes your code

    # Get the fixed data from the system characteristics file
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
    M_T       = 100.0   # big-M for temperature constraints
    M_H       = 200.0   # big-M for humidity constraint

    # Pre-trained theta parameters for the future VFA
    theta = np.array([[ 3.20059592e-02,  6.72125143e-01,  6.72125143e-01,
         1.28023837e+00,  4.78038510e+01, -2.59907658e+00,
        -1.39416433e+00,  0.00000000e+00],
       [ 4.82492536e+02, -4.05212597e+00, -2.95522035e+00,
        -6.03199280e+00,  2.89414490e+01,  7.18962013e-02,
         1.81440266e-02, -9.14056369e+01],
       [ 2.56484037e+02, -2.25553320e+00, -3.66755886e+00,
        -1.48737402e+00,  2.15254910e+01,  7.82927826e-02,
         2.55478981e-01, -2.48905405e+01],
       [ 1.77246994e+02, -3.35249713e+00, -1.93319431e+00,
        -4.98741196e-01,  1.77918644e+01,  1.46945106e-01,
         4.15216240e-01, -8.87124115e+00],
       [ 1.47705305e+02, -2.29358459e+00, -3.33940769e+00,
        -2.02709553e-02,  1.53757443e+01, -2.85362942e-02,
         2.55243948e-01, -7.78696639e-01],
       [ 1.23610874e+02, -3.05843860e+00, -1.86062208e+00,
         5.13620172e-02,  1.32545560e+01, -2.56153716e-01,
         3.83021916e-01, -1.38143178e+00],
       [ 8.71818701e+01, -1.91713841e+00, -1.53628197e+00,
         7.95664347e-02,  1.08913101e+01, -2.75490643e-01,
         2.23336255e-01, -1.09028670e+00],
       [ 6.68581210e+01, -1.77231600e+00, -1.18970544e+00,
         9.38827353e-02,  7.89344701e+00, -9.55245152e-02,
         7.38625666e-02, -2.57510346e-01],
       [ 3.95781576e+01, -8.91281928e-01, -9.04039021e-01,
         1.29799589e-02,  4.18961016e+00, -5.58656210e-02,
         1.09248418e-01, -7.26057516e-01],
       [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
         0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
         0.00000000e+00,  0.00000000e+00]])

    # Extract state variables and convert to appropriate types if needed from the state dictionary provided by the environment
    t            = int(state['current_time'])
    T1           = state['T1']
    T2           = state['T2']
    H            = state['H']
    Occ1         = state['Occ1']
    Occ2         = state['Occ2']
    price        = state['price_t']
    vent_counter = int(state['vent_counter'])
    lo_r1        = int(state['low_override_r1'])
    lo_r2        = int(state['low_override_r2'])
    T_outdoor    = T_OUT[t]

    m = pyo.ConcreteModel()

    # ------------------------------------------------------------------
    # Decision variables
    # ------------------------------------------------------------------
    m.p1      = pyo.Var(within=pyo.NonNegativeReals, bounds=(0, P_MAX))
    m.p2      = pyo.Var(within=pyo.NonNegativeReals, bounds=(0, P_MAX))
    m.v       = pyo.Var(within=pyo.Binary)
    m.T1_next = pyo.Var(within=pyo.Reals)
    m.T2_next = pyo.Var(within=pyo.Reals)
    m.H_next  = pyo.Var(within=pyo.NonNegativeReals, bounds=(0, 100))

    # Binary indicators for overrule controllers
    m.Htemp_r1 = pyo.Var(within=pyo.Binary)
    m.Htemp_r2 = pyo.Var(within=pyo.Binary)
    m.Ltemp_r1 = pyo.Var(within=pyo.Binary)
    m.Ltemp_r2 = pyo.Var(within=pyo.Binary)

    # ------------------------------------------------------------------
    # Dynamics constraints
    # ------------------------------------------------------------------
    m.T1_dyn = pyo.Constraint(expr=
        m.T1_next == T1 + ZETA_CONV*m.p1 - ZETA_LOSS*(T1-T_outdoor)
        + ZETA_EXCH*(T2-T1) - ZETA_COOL*m.v + ZETA_OCC*Occ1)

    m.T2_dyn = pyo.Constraint(expr=
        m.T2_next == T2 + ZETA_CONV*m.p2 - ZETA_LOSS*(T2-T_outdoor)
        + ZETA_EXCH*(T1-T2) - ZETA_COOL*m.v + ZETA_OCC*Occ2)

    m.H_dyn = pyo.Constraint(expr=
        m.H_next == H + ETA_OCC*(Occ1+Occ2) - ETA_VENT*m.v)

    # ------------------------------------------------------------------
    # High temperature overrule (big-M): T1 >= T_HIGH → Htemp_r1=1 → p1=0
    # ------------------------------------------------------------------
    m.Htemp_r1_ub = pyo.Constraint(expr= T1 - T_HIGH <= M_T * m.Htemp_r1)
    m.Htemp_r1_lb = pyo.Constraint(expr= T1 - T_HIGH >= -M_T * (1 - m.Htemp_r1))
    m.Htemp_p1    = pyo.Constraint(expr= m.p1 <= P_MAX * (1 - m.Htemp_r1))

    m.Htemp_r2_ub = pyo.Constraint(expr= T2 - T_HIGH <= M_T * m.Htemp_r2)
    m.Htemp_r2_lb = pyo.Constraint(expr= T2 - T_HIGH >= -M_T * (1 - m.Htemp_r2))
    m.Htemp_p2    = pyo.Constraint(expr= m.p2 <= P_MAX * (1 - m.Htemp_r2))

    # ------------------------------------------------------------------
    # Low temperature overrule: lo_r from state encodes hysteresis
    # ------------------------------------------------------------------
    if lo_r1:
        m.Ltemp_r1.fix(1)
        m.Ltemp_p1 = pyo.Constraint(expr= m.p1 == P_MAX)
    else:
        m.Ltemp_r1_ub = pyo.Constraint(expr= T_LOW - T1 <= M_T * m.Ltemp_r1)
        m.Ltemp_r1_lb = pyo.Constraint(expr= T_LOW - T1 >= -M_T * (1 - m.Ltemp_r1))
        m.Ltemp_p1    = pyo.Constraint(expr= m.p1 >= P_MAX * m.Ltemp_r1)

    if lo_r2:
        m.Ltemp_r2.fix(1)
        m.Ltemp_p2 = pyo.Constraint(expr= m.p2 == P_MAX)
    else:
        m.Ltemp_r2_ub = pyo.Constraint(expr= T_LOW - T2 <= M_T * m.Ltemp_r2)
        m.Ltemp_r2_lb = pyo.Constraint(expr= T_LOW - T2 >= -M_T * (1 - m.Ltemp_r2))
        m.Ltemp_p2    = pyo.Constraint(expr= m.p2 >= P_MAX * m.Ltemp_r2)

    # Mutual exclusion
    m.excl_r1 = pyo.Constraint(expr= m.Ltemp_r1 + m.Htemp_r1 <= 1)
    m.excl_r2 = pyo.Constraint(expr= m.Ltemp_r2 + m.Htemp_r2 <= 1)

    # ------------------------------------------------------------------
    # Ventilation inertia
    # ------------------------------------------------------------------
    if vent_counter in (1, 2):
        m.vent_inertia = pyo.Constraint(expr= m.v == 1)

    # ------------------------------------------------------------------
    # Humidity overrule (big-M): H > H_HIGH → v = 1
    # ------------------------------------------------------------------
    m.hum_overrule = pyo.Constraint(expr= H - H_HIGH <= M_H * m.v)

    # ------------------------------------------------------------------
    # Future VFA
    # ------------------------------------------------------------------
    if t < T_SLOTS - 1:
        th = theta[t + 1]
        vc_next = float(min(vent_counter + 1, VENT_MIN)) * m.v
        future_cost = (th[0] + th[1]*m.T1_next + th[2]*m.T2_next
                    + th[3]*m.H_next + th[4]*price
                    + th[5]*Occ1 + th[6]*Occ2 + th[7]*vc_next)
    else:
        future_cost = 0.0

    # ------------------------------------------------------------------
    # Objective
    # ------------------------------------------------------------------
    m.obj = pyo.Objective(
        expr=price*(m.p1 + m.p2 + P_VENT*m.v) + future_cost,
        sense=pyo.minimize)

    solver = pyo.SolverFactory('gurobi')
    result = solver.solve(m, tee=False)

    if result.solver.termination_condition != pyo.TerminationCondition.optimal:
        HereAndNowActions = {
            "HeatPowerRoom1": 0.0,
            "HeatPowerRoom2": 0.0,
            "VentilationON":  0
        }
    else:
        HereAndNowActions = {
            "HeatPowerRoom1": float(pyo.value(m.p1)),
            "HeatPowerRoom2": float(pyo.value(m.p2)),
            "VentilationON":  int(round(pyo.value(m.v))),
        }

    return HereAndNowActions

#print(select_action(1))