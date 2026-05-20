# -*- coding: utf-8 -*-
#
# Simulation environment for evaluating decision-making policies.
#
# Policies included:
#   - Dummy
#   - Optimal in Hindsight  (imported from Optimal_in_Hindsight.py)
#   - Stochastic Programming (imported from SP_policy_02.py)
#   - Expected Value / deterministic lookahead (imported from EV_policy_02.py)
#   - Two-Stage SP (imported from two_stage_SP_policy_02.py)
#   - ADP (imported from ADP_policy_02.py)
#   - Hybrid (imported from Hybrid_policy_02.py)

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from v2_SystemCharacteristics import get_fixed_data

# Load parameters once at module level
params    = get_fixed_data()
T         = params['num_timeslots']
P_MAX     = params['heating_max_power']
P_VENT    = params['ventilation_power']
T_LOW     = params['temp_min_comfort_threshold']
T_OK      = params['temp_OK_threshold']
T_HIGH    = params['temp_max_comfort_threshold']
H_HIGH    = params['humidity_threshold']
VENT_MIN  = params['vent_min_up_time']
ZETA_CONV = params['heating_efficiency_coeff']
ZETA_LOSS = params['thermal_loss_coeff']
ZETA_EXCH = params['heat_exchange_coeff']
ZETA_COOL = params['heat_vent_coeff']
ZETA_OCC  = params['heat_occupancy_coeff']
ETA_OCC   = params['humidity_occupancy_coeff']
ETA_VENT  = params['humidity_vent_coeff']
T_OUT     = params['outdoor_temperature']


# ==================================================================
# load_data
# ==================================================================
def load_data(price_file='v2_PriceData.csv',
              occ1_file='OccupancyRoom1.csv',
              occ2_file='OccupancyRoom2.csv'):
    """Load 100-day time series. Returns arrays of shape (n_days, T).

    v2_PriceData.csv has an extra first column — the initial price_previous
    for each day. This is extracted separately and the remaining 10 columns
    are the hourly prices.
    """
    raw    = pd.read_csv(price_file, header=0).values.astype(float)
    occ1   = pd.read_csv(occ1_file,  header=0).values.astype(float)
    occ2   = pd.read_csv(occ2_file,  header=0).values.astype(float)

    if raw.shape[1] == T + 1:
        price_previous_per_day = raw[:, 0]
        prices = raw[:, 1:]
    else:
        price_previous_per_day = None
        prices = raw

    assert prices.shape == occ1.shape == occ2.shape, \
        "Price and occupancy CSV files must have the same shape."
    assert prices.shape[1] == T, \
        f"Expected {T} columns but found {prices.shape[1]}."

    return prices, occ1, occ2, price_previous_per_day


# ==================================================================
# _apply_overrules(state, action)
# ==================================================================
def _apply_overrules(state, action):
    p1 = float(action['HeatPowerRoom1'])
    p2 = float(action['HeatPowerRoom2'])
    v  = int(action['VentilationON'])

    if state['T1'] >= T_HIGH:
        p1 = 0.0
    if state['T2'] >= T_HIGH:
        p2 = 0.0

    if state['low_override_r1']:
        p1 = P_MAX
    if state['low_override_r2']:
        p2 = P_MAX

    if state['vent_counter'] in (1, 2):
        v = 1

    if state['H'] >= H_HIGH:
        v = 1

    return {'HeatPowerRoom1': p1, 'HeatPowerRoom2': p2, 'VentilationON': v}


# ==================================================================
# _step
# ==================================================================
def _step(state, eff, t, price_next, occ1_next, occ2_next):
    T1   = state['T1']
    T2   = state['T2']
    H    = state['H']
    Occ1 = state['Occ1']
    Occ2 = state['Occ2']
    p1   = eff['HeatPowerRoom1']
    p2   = eff['HeatPowerRoom2']
    v    = eff['VentilationON']

    T1_next = (T1
               + ZETA_CONV * p1
               - ZETA_LOSS * (T1 - T_OUT[t])
               + ZETA_EXCH * (T2 - T1)
               - ZETA_COOL * v
               + ZETA_OCC  * Occ1)

    T2_next = (T2
               + ZETA_CONV * p2
               - ZETA_LOSS * (T2 - T_OUT[t])
               + ZETA_EXCH * (T1 - T2)
               - ZETA_COOL * v
               + ZETA_OCC  * Occ2)

    H_next = float(np.clip(
        H + ETA_OCC * (Occ1 + Occ2) - ETA_VENT * v,
        0.0, 100.0))

    if T1_next < T_LOW:
        lo_r1_next = 1
    elif T1_next >= T_OK:
        lo_r1_next = 0
    else:
        lo_r1_next = state['low_override_r1']

    if T2_next < T_LOW:
        lo_r2_next = 1
    elif T2_next >= T_OK:
        lo_r2_next = 0
    else:
        lo_r2_next = state['low_override_r2']

    if v == 1:
        vc_next = min(state['vent_counter'] + 1, VENT_MIN)
    else:
        vc_next = 0

    return {
        'T1':              T1_next,
        'T2':              T2_next,
        'H':               H_next,
        'Occ1':            occ1_next,
        'Occ2':            occ2_next,
        'price_t':         price_next,
        'price_previous':  state['price_t'],
        'vent_counter':    vc_next,
        'low_override_r1': lo_r1_next,
        'low_override_r2': lo_r2_next,
        'current_time':    t + 1,
    }

# ==================================================================
# evaluate_day
# ==================================================================
def evaluate_day(policy, day_idx, prices, occ1, occ2, price_previous_init=None):
    """Run one policy through one full day using fixed historical data."""

    data  = get_fixed_data()
    if price_previous_init is None:
        price_previous_init = data['price_previous']
    state = {
        'T1':              data['T1'],
        'T2':              data['T2'],
        'H':               data['H'],
        'Occ1':            float(occ1[day_idx, 0]),
        'Occ2':            float(occ2[day_idx, 0]),
        'price_t':         float(prices[day_idx, 0]),
        'price_previous':  price_previous_init,
        'vent_counter':    data['vent_counter'],
        'low_override_r1': data['low_override_r1'],
        'low_override_r2': data['low_override_r2'],
        'current_time':    0,
    }

    daily_cost = 0.0
    history    = []

    for t in range(T):

        try:
            action = policy(state)
        except Exception as e:
            print(f"  [Day {day_idx+1}, t={t}] Policy crashed: {e}. Using dummy.")
            action = {'HeatPowerRoom1': 0.0, 'HeatPowerRoom2': 0.0, 'VentilationON': 0}

        eff = _apply_overrules(state, action)

        cost_t = state['price_t'] * (
            eff['HeatPowerRoom1'] + eff['HeatPowerRoom2'] + P_VENT * eff['VentilationON']
        )
        daily_cost += cost_t

        history.append({
            'hour':  t,
            'T1':    state['T1'],
            'T2':    state['T2'],
            'H':     state['H'],
            'price': state['price_t'],
            'Occ1':  state['Occ1'],
            'Occ2':  state['Occ2'],
            'p1':    eff['HeatPowerRoom1'],
            'p2':    eff['HeatPowerRoom2'],
            'v':     eff['VentilationON'],
            'cost':  cost_t,
        })

        if t < T - 1:
            state = _step(
                state, eff, t,
                price_next = float(prices[day_idx, t + 1]),
                occ1_next  = float(occ1[day_idx,   t + 1]),
                occ2_next  = float(occ2[day_idx,   t + 1]),
            )

    return daily_cost, history


# ==================================================================
# evaluate_day_hindsight
#
# Special path for the Optimal-in-Hindsight policy.
# Solves the full-day MILP with perfect information and reads the
# cost directly from m.ObjVal — this is the true lower bound and
# avoids the simulation mismatch that would inflate the cost when
# replaying actions through the environment's separate physics.
# ==================================================================
def evaluate_day_hindsight(hindsight_policy, day_idx, prices, occ1, occ2,
                           price_previous_init=None):
    """Pre-solve hindsight MILP and return its own objective as cost."""
    hindsight_policy.set_day(
        price_row = prices[day_idx],
        occ1_row  = occ1[day_idx],
        occ2_row  = occ2[day_idx],
    )
    cost = hindsight_policy.optimal_cost()
    if cost is None:
        cost = float('inf')
    # Return a minimal history so the result dict is consistent
    history = [{'hour': t, 'cost': None} for t in range(T)]
    return cost, history


# ==================================================================
# evaluate_policy
# ==================================================================
def evaluate_policy(policy, n_days=100, verbose=True,
                    price_file='v2_PriceData.csv',
                    occ1_file='OccupancyRoom1.csv',
                    occ2_file='OccupancyRoom2.csv',
                    hindsight=False,
                    price_previous_per_day=None):
    prices, occ1, occ2, pp_from_csv = load_data(price_file, occ1_file, occ2_file)
    n_days = min(n_days, prices.shape[0])

    # CSV-provided price_previous takes priority over caller-supplied values
    if pp_from_csv is not None:
        price_previous_per_day = pp_from_csv

    daily_costs = np.zeros(n_days)
    all_history = []

    for d in range(n_days):
        pp_init = price_previous_per_day[d] if price_previous_per_day is not None else None
        if hindsight:
            cost, history = evaluate_day_hindsight(policy, d, prices, occ1, occ2, pp_init)
        else:
            cost, history = evaluate_day(policy, d, prices, occ1, occ2, pp_init)
        daily_costs[d] = cost
        all_history.append(history)
        if verbose and (d + 1) % 10 == 0:
            print(f"  Day {d+1:3d}/{n_days}  |  cost = {cost:.2f}  |  "
                  f"running mean = {daily_costs[:d+1].mean():.2f}")

    results = {
        'daily_costs': daily_costs,
        'mean_cost':   float(daily_costs.mean()),
        'std_cost':    float(daily_costs.std()),
        'pct95_cost':  float(np.percentile(daily_costs, 95)),
        'worst_cost':  float(daily_costs.max()),
        'best_cost':   float(daily_costs.min()),
        'all_history': all_history,
    }

    if verbose:
        print(f"\n  Mean cost:        {results['mean_cost']:.4f}")
        print(f"  Std dev:          {results['std_cost']:.4f}")
        print(f"  95th percentile:  {results['pct95_cost']:.4f}")
        print(f"  Worst day:        {results['worst_cost']:.4f}")
        print(f"  Best day:         {results['best_cost']:.4f}")

    return results


# ==================================================================
# compare_policies
# ==================================================================
def compare_policies(policies_dict, n_days=100, verbose=True, **kwargs):
    """
    policies_dict values are either:
      - a callable  policy(state) -> action
      - a tuple     (callable, {'hindsight': True})  for special evaluation

    price_previous per day is read from v2_PriceData.csv (first column),
    ensuring all policies share identical initial conditions.
    """

    all_results = {}
    for name, entry in policies_dict.items():
        if isinstance(entry, tuple):
            policy, extra_kwargs = entry
        else:
            policy, extra_kwargs = entry, {}

        if verbose:
            print(f"\nEvaluating: {name}")
            print("-" * 50)

        all_results[name] = evaluate_policy(
            policy, n_days=n_days, verbose=verbose,
            **{**kwargs, **extra_kwargs})

    return all_results


# ==================================================================
# plot_results
# ==================================================================
def plot_results(all_results, bins=20, save_path=None):
    names  = list(all_results.keys())
    colors = plt.cm.tab10(np.linspace(0, 0.9, len(names)))

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    ax1   = axes[0]
    means = [all_results[n]['mean_cost'] for n in names]
    stds  = [all_results[n]['std_cost']  for n in names]

    bars = ax1.bar(range(len(names)), means, yerr=stds, capsize=5,
                   color=colors, edgecolor='white',
                   error_kw={'elinewidth': 1.5, 'ecolor': 'black'})
    ax1.set_xticks(range(len(names)))
    ax1.set_xticklabels(names, rotation=30, ha='right', fontsize=10)
    ax1.set_ylabel("Average daily cost", fontsize=11)
    ax1.set_title("Average Daily Cost per Policy\n(error bars = ±1 std dev)",
                  fontsize=12, fontweight='bold')
    ax1.grid(axis='y', linestyle='--', alpha=0.4)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    for bar, mean in zip(bars, means):
        ax1.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + max(stds) * 0.05,
                 f"{mean:.2f}",
                 ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax2 = axes[1]
    for name, color in zip(names, colors):
        ax2.hist(all_results[name]['daily_costs'], bins=bins,
                 alpha=0.5, color=color, label=name,
                 edgecolor='white', linewidth=0.5)
    ax2.set_xlabel("Daily cost", fontsize=11)
    ax2.set_ylabel("Number of days", fontsize=11)
    ax2.set_title("Distribution of Daily Costs\n(100 experiments)",
                  fontsize=12, fontweight='bold')
    ax2.legend(fontsize=9, framealpha=0.8)
    ax2.grid(axis='y', linestyle='--', alpha=0.4)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    plt.show()
    return fig


# ==================================================================
# print_summary_table
# ==================================================================
def print_summary_table(all_results):
    names = list(all_results.keys())
    col_w = max(len(n) for n in names) + 2
    header = (f"{'Policy':<{col_w}}  {'Mean':>8}  {'Std':>8}  "
              f"{'95th %':>8}  {'Worst':>8}  {'Best':>8}")
    print("\n" + "=" * len(header))
    print(header)
    print("=" * len(header))
    for name in names:
        r = all_results[name]
        print(f"{name:<{col_w}}  "
              f"{r['mean_cost']:>8.4f}  "
              f"{r['std_cost']:>8.4f}  "
              f"{r['pct95_cost']:>8.4f}  "
              f"{r['worst_cost']:>8.4f}  "
              f"{r['best_cost']:>8.4f}")
    print("=" * len(header))


# ==================================================================
# Main — run all 7 policies
# ==================================================================
if __name__ == "__main__":

    from ADP_policy_02    import select_action as adp_policy
    from SP_policy_02     import select_action as sp_policy
    from Hybrid_policy_02 import select_action as hybrid_policy
    from Optimal_in_Hindsight_02 import HindsightPolicy
    from two_stage_SP_policy_02 import select_action as two_stage_sp_policy
    from EV_policy_02 import select_action as ev_policy

    def dummy_policy(state):
        return {'HeatPowerRoom1': 0.0, 'HeatPowerRoom2': 0.0, 'VentilationON': 0}

    hindsight_policy = HindsightPolicy()

    policies = {
        'Hindsight':   (hindsight_policy, {'hindsight': True}),
        'Dummy':        dummy_policy,
        'EV':           ev_policy,
        'Two-Stage SP': two_stage_sp_policy,
        'SP':           sp_policy,
        'ADP':          adp_policy,
        'Hybrid':       hybrid_policy,
    }

    print("Starting policy evaluation...")
    all_results = compare_policies(policies, n_days=100, verbose=True)

    print_summary_table(all_results)
    plot_results(all_results, save_path='policy_comparison_all.png')
