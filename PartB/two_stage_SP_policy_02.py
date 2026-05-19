# -*- coding: utf-8 -*-
#
#
# Two-Stage SP policy — thin wrapper around the SP infrastructure.
# Reuses _build_tree and _solve from SP_policy.py with L=2.
#
# With L=2 and _BRANCHING[2] = [10]:
#   - Node 0: here-and-now (one shared decision across all scenarios)
#   - 10 scenario nodes at stage 1: sampled via k-medoids from pool of 60
#   - Probability weights: cluster size / 60 (not equal weights)
#
# At the last hour (remaining=1), L=1 gives a single-node greedy solve.

from SP_policy_02 import _build_tree, _solve, _PARAMS, _T_TOTAL


def select_action(state):
    """
    Two-Stage SP policy.

    Here-and-now decision at t + one stochastic recourse stage at t+1.
    Reuses SP infrastructure with lookahead horizon capped at L=2.
    """
    cur_time  = int(state.get('current_time', 0))
    remaining = max(1, _T_TOTAL - cur_time)
    L         = min(remaining, 2)   # force two-stage lookahead

    try:
        nodes     = _build_tree(state, L)
        p1, p2, v = _solve(state, nodes, cur_time)
    except Exception:
        p1, p2, v = 0.0, 0.0, 0

    Pmax = _PARAMS['heating_max_power']
    return {
        'HeatPowerRoom1': float(max(0.0, min(Pmax, p1))),
        'HeatPowerRoom2': float(max(0.0, min(Pmax, p2))),
        'VentilationON':  int(v),
    }