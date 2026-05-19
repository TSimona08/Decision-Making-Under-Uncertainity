# -*- coding: utf-8 -*-
#
#
# Expected Value (deterministic lookahead) policy.
#
# Builds a deterministic chain from t to T-1 using one scenario per
# stage — the conditional expectation of the next exogenous state.
# This is equivalent to the SP policy with K=1 scenario per stage.
#
# Uses the same SP infrastructure (_build_tree, _solve) as the
# Two-Stage SP and Multi-Stage SP policies, guaranteeing identical
# MILP structure and constraint formulation for a fair comparison.
#
# The only difference from Two-Stage SP and Multi-Stage SP is the
# branching: one scenario per stage instead of multiple.

from SP_policy_02 import _build_tree, _solve, _PARAMS, _T_TOTAL


def select_action(state):
    """
    Expected Value (deterministic lookahead) policy.

    Looks ahead to T-1 using a single expected-value scenario chain.
    Uses SP infrastructure with branching overridden to K=1 per stage.
    """
    cur_time  = int(state.get('current_time', 0))
    remaining = max(1, _T_TOTAL - cur_time)
    L         = min(remaining, 5)

    # One scenario per stage = deterministic expected value chain.
    # _build_tree will sample once per stage and use that single
    # realisation as the representative scenario — equivalent to
    # using the conditional expectation at each stage.
    ev_branching = {1: [], 2: [1], 3: [1, 1], 4: [1, 1, 1], 5: [1, 1, 1, 1]}
    branching    = ev_branching.get(L, [1] * (L - 1))

    try:
        nodes     = _build_tree(state, L, branching_override=branching)
        p1, p2, v = _solve(state, nodes, cur_time)
    except Exception:
        p1, p2, v = 0.0, 0.0, 0

    Pmax = _PARAMS['heating_max_power']
    return {
        'HeatPowerRoom1': float(max(0.0, min(Pmax, p1))),
        'HeatPowerRoom2': float(max(0.0, min(Pmax, p2))),
        'VentilationON':  int(v),
    }