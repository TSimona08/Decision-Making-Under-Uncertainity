# -*- coding: utf-8 -*-
"""
Optimal-in-hindsight policy.

Solves the full-day MILP once with perfect knowledge of the entire
price and occupancy sequence, then replays the optimal actions
step-by-step during simulation.

Usage:
    from Optimal_in_Hindsight_02 import HindsightPolicy
    policy = HindsightPolicy()
    # environment calls policy.set_day(...) before each day
"""

import numpy as np
import gurobipy as gp
from gurobipy import GRB
from v2_SystemCharacteristics import get_fixed_data

_data      = get_fixed_data()
_L         = _data['num_timeslots']
_T         = range(_L)
_R         = [0, 1]
_P_BAR     = _data['heating_max_power']
_P_VENT    = _data['ventilation_power']
_T_LOW     = _data['temp_min_comfort_threshold']
_T_OK      = _data['temp_OK_threshold']
_T_HIGH    = _data['temp_max_comfort_threshold']
_H_HIGH    = _data['humidity_threshold']
_U_VENT    = _data['vent_min_up_time']
_ZETA_CONV = _data['heating_efficiency_coeff']
_ZETA_LOSS = _data['thermal_loss_coeff']
_ZETA_EXCH = _data['heat_exchange_coeff']
_ZETA_COOL = _data['heat_vent_coeff']
_ZETA_OCC  = _data['heat_occupancy_coeff']
_ETA_OCC   = _data['humidity_occupancy_coeff']
_ETA_VENT  = _data['humidity_vent_coeff']
_T_OUT     = _data['outdoor_temperature']
_T_INIT    = _data['T1']
_H_INIT    = _data['H']
_M_TEMP    = 100.0
_M_HUM     = 200.0


def _solve_milp(price, occ):
    """
    Solve the full-day MILP with perfect hindsight.

    price : (L,)   array — electricity price at each hour
    occ   : (2, L) array — occ[0]=room1, occ[1]=room2
    Returns dict with 'p1', 'p2', 'v' lists of length L, or None.
    """
    m = gp.Model("hindsight")
    m.Params.OutputFlag = 0

    p      = m.addVars(_R, _T, lb=0, ub=_P_BAR,   vtype=GRB.CONTINUOUS, name="p")
    v      = m.addVars(_T,            vtype=GRB.BINARY,     name="v")
    s      = m.addVars(_T,            vtype=GRB.BINARY,     name="s")
    temp   = m.addVars(_R, _T,        vtype=GRB.CONTINUOUS, name="temp")
    hum    = m.addVars(_T, lb=0, ub=100, vtype=GRB.CONTINUOUS, name="hum")
    y_low  = m.addVars(_R, _T,        vtype=GRB.BINARY,     name="y_low")
    y_ok   = m.addVars(_R, _T,        vtype=GRB.BINARY,     name="y_ok")
    y_high = m.addVars(_R, _T,        vtype=GRB.BINARY,     name="y_high")
    u      = m.addVars(_R, _T,        vtype=GRB.BINARY,     name="u")

    # Objective
    m.setObjective(
        gp.quicksum(price[t] * p[r, t] for r in _R for t in _T)
        + gp.quicksum(price[t] * _P_VENT * v[t] for t in _T),
        GRB.MINIMIZE
    )

    # Initial conditions
    for r in _R:
        m.addConstr(temp[r, 0] == _T_INIT)
    m.addConstr(hum[0] == _H_INIT)

    # Temperature dynamics
    for r in _R:
        r_other = 1 - r
        for t in _T:
            if t < _L - 1:
                m.addConstr(
                    temp[r, t+1] == temp[r, t]
                    + _ZETA_CONV * p[r, t]
                    - _ZETA_LOSS * (temp[r, t] - _T_OUT[t])
                    + _ZETA_EXCH * (temp[r_other, t] - temp[r, t])
                    - _ZETA_COOL * v[t]
                    + _ZETA_OCC  * occ[r, t]
                )

    # Humidity dynamics
    for t in _T:
        if t < _L - 1:
            m.addConstr(
                hum[t+1] == hum[t]
                + _ETA_OCC * gp.quicksum(occ[r, t] for r in _R)
                - _ETA_VENT * v[t]
            )

    # High-temperature detection
    for r in _R:
        for t in _T:
            m.addConstr(temp[r, t] >= _T_HIGH - _M_TEMP * (1 - y_high[r, t]))
            m.addConstr(temp[r, t] <= _T_HIGH + _M_TEMP * y_high[r, t])

    # Low-temperature detection
    for r in _R:
        for t in _T:
            m.addConstr(temp[r, t] <= _T_LOW + _M_TEMP * (1 - y_low[r, t]))
            m.addConstr(temp[r, t] >= _T_LOW - _M_TEMP * y_low[r, t])

    # OK-temperature detection
    for r in _R:
        for t in _T:
            m.addConstr(temp[r, t] >= _T_OK - _M_TEMP * (1 - y_ok[r, t]))
            m.addConstr(temp[r, t] <= _T_OK + _M_TEMP * y_ok[r, t])

    # Overrule controller u (hysteresis)
    for r in _R:
        m.addConstr(u[r, 0] >= y_low[r, 0])
        m.addConstr(u[r, 0] <= y_low[r, 0])
        for t in range(1, _L):
            m.addConstr(u[r, t] >= y_low[r, t])
            m.addConstr(u[r, t] <= u[r, t-1] + y_low[r, t])
            m.addConstr(u[r, t] >= u[r, t-1] - y_ok[r, t])
            m.addConstr(u[r, t] <= 1 - y_ok[r, t])

    # Forcing constraints
    for r in _R:
        for t in _T:
            m.addConstr(p[r, t] <= _P_BAR * (1 - y_high[r, t]))
            m.addConstr(p[r, t] >= _P_BAR * u[r, t])
            m.addConstr(u[r, t] + y_high[r, t] <= 1)

    # Humidity overrule
    for t in _T:
        m.addConstr(hum[t] <= _H_HIGH + _M_HUM * v[t])

    # Startup variable
    for t in _T:
        if t == 0:
            m.addConstr(s[t] >= v[t])
            m.addConstr(s[t] <= v[t])
        else:
            m.addConstr(s[t] >= v[t] - v[t-1])
            m.addConstr(s[t] <= v[t])
            m.addConstr(s[t] <= 1 - v[t-1])

    # Minimum up-time
    for t in _T:
        tau_end   = min(t + _U_VENT - 1, _L - 1)
        min_slots = min(_U_VENT, _L - t)
        m.addConstr(
            gp.quicksum(v[tau] for tau in range(t, tau_end + 1)) >= min_slots * s[t]
        )

    m.optimize()

    if m.Status != GRB.OPTIMAL:
        return None

    return {
        'cost': m.ObjVal,
        'p1':   [p[0, t].X          for t in _T],
        'p2':   [p[1, t].X          for t in _T],
        'v':    [int(round(v[t].X)) for t in _T],
    }


class HindsightPolicy:
    """
    Callable policy that replays the full-day optimal MILP solution.

    The environment must call set_day() before simulating each day.
    """

    def __init__(self):
        self._plan = None

    def set_day(self, price_row, occ1_row, occ2_row):
        """Pre-solve the MILP for one day and cache the plan (including ObjVal)."""
        occ = np.array([occ1_row, occ2_row])
        self._plan = _solve_milp(price_row, occ)

    def optimal_cost(self):
        """Return the MILP's own objective value (true hindsight lower bound)."""
        if self._plan is None:
            return None
        return self._plan['cost']

    def __call__(self, state):
        t = int(state['current_time'])
        if self._plan is None:
            return {'HeatPowerRoom1': 0.0, 'HeatPowerRoom2': 0.0, 'VentilationON': 0}
        return {
            'HeatPowerRoom1': float(self._plan['p1'][t]),
            'HeatPowerRoom2': float(self._plan['p2'][t]),
            'VentilationON':  int(self._plan['v'][t]),
        }
