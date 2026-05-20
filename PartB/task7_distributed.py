import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import pyomo.environ as pyo
from pyomo.opt import SolverFactory

HERE = Path(__file__).resolve().parent
ROOT = HERE if (HERE / 'DataTask7.py').exists() else HERE.parent
sys.path.insert(0, str(ROOT))
from DataTask7 import fetch_data


_DATA = fetch_data()
T          = int(_DATA['num_timeslots'])
P_MALL     = float(_DATA['P_mall'])
T_REF      = float(_DATA['Temperature_reference'])
T_INIT     = float(_DATA['initial_temperature'])
P_MAX      = float(_DATA['heating_max_power'])
ZX         = float(_DATA['heat_exchange_coeff'])
ZL         = float(_DATA['thermal_loss_coeff'])
ZH         = float(_DATA['heating_efficiency_coeff'])
ZC         = float(_DATA['heat_vent_coeff'])
ZO         = float(_DATA['heat_occupancy_coeff'])
T_OUT      = list(_DATA['outdoor_temperature'])

N = 15
W = {n: n + 1 for n in range(1, N + 1)}


def _load_occupancy(path):
    with open(path) as f:
        lines = f.readlines()
    def parse(line):
        parts = line.strip().split(',')
        out = []
        for p in parts:
            p = p.strip()
            if not p or 'room' in p.lower():
                continue
            out.append(float(p))
        return np.array(out)
    return parse(lines[1]), parse(lines[2])


OCC1, OCC2 = _load_occupancy(ROOT / 'Task7Occupancies.csv')
assert len(OCC1) == T and len(OCC2) == T


_SOLVER = SolverFactory('gurobi')
_SOLVER.options['OutputFlag'] = 0


def _safe_value(x):
    try:
        return float(pyo.value(x))
    except Exception:
        return 0.0


def solve_local(w_n, lam):
    m = pyo.ConcreteModel()
    m.T_idx = pyo.RangeSet(0, T - 1)
    m.R = pyo.Set(initialize=[1, 2])

    m.p = pyo.Var(m.R, m.T_idx, bounds=(0, P_MAX))
    m.T = pyo.Var(m.R, m.T_idx)

    def _init(mm, r):
        return mm.T[r, 0] == T_INIT
    m.c_init = pyo.Constraint(m.R, rule=_init)

    def _occ(r, t):
        return OCC1[t] if r == 1 else OCC2[t]

    def _dyn(mm, r, t):
        if t == 0:
            return pyo.Constraint.Skip
        rp = 2 if r == 1 else 1
        return mm.T[r, t] == (
            mm.T[r, t - 1]
            + ZX * (mm.T[rp, t - 1] - mm.T[r, t - 1])
            - ZL * (mm.T[r, t - 1] - T_OUT[t - 1])
            + ZH * mm.p[r, t - 1]
            - ZC
            + ZO * _occ(r, t - 1)
        )
    m.c_dyn = pyo.Constraint(m.R, m.T_idx, rule=_dyn)

    def _terminal_local(mm, r):
        return mm.p[r, T - 1] == 0
    m.c_terminal = pyo.Constraint(m.R, rule=_terminal_local)

    def _obj(mm):
        sq = sum(w_n * (mm.T[r, t] - T_REF) ** 2 for r in mm.R for t in mm.T_idx)
        lin = sum(lam[t] * (mm.p[1, t] + mm.p[2, t]) for t in mm.T_idx)
        return sq + lin
    m.obj = pyo.Objective(rule=_obj, sense=pyo.minimize)

    _SOLVER.solve(m, tee=False)

    p1 = np.array([_safe_value(m.p[1, t]) for t in range(T)])
    p2 = np.array([_safe_value(m.p[2, t]) for t in range(T)])
    T1 = np.array([_safe_value(m.T[1, t]) for t in range(T)])
    T2 = np.array([_safe_value(m.T[2, t]) for t in range(T)])
    return p1, p2, T1, T2


def solve_centralised():
    m = pyo.ConcreteModel()
    m.N = pyo.RangeSet(1, N)
    m.T_idx = pyo.RangeSet(0, T - 1)
    m.R = pyo.Set(initialize=[1, 2])

    m.p = pyo.Var(m.N, m.R, m.T_idx, bounds=(0, P_MAX))
    m.T = pyo.Var(m.N, m.R, m.T_idx)

    def _init(mm, n, r):
        return mm.T[n, r, 0] == T_INIT
    m.c_init = pyo.Constraint(m.N, m.R, rule=_init)

    def _occ(r, t):
        return OCC1[t] if r == 1 else OCC2[t]

    def _dyn(mm, n, r, t):
        if t == 0:
            return pyo.Constraint.Skip
        rp = 2 if r == 1 else 1
        return mm.T[n, r, t] == (
            mm.T[n, r, t - 1]
            + ZX * (mm.T[n, rp, t - 1] - mm.T[n, r, t - 1])
            - ZL * (mm.T[n, r, t - 1] - T_OUT[t - 1])
            + ZH * mm.p[n, r, t - 1]
            - ZC
            + ZO * _occ(r, t - 1)
        )
    m.c_dyn = pyo.Constraint(m.N, m.R, m.T_idx, rule=_dyn)

    def _terminal_central(mm, n, r):
        return mm.p[n, r, T - 1] == 0
    m.c_terminal = pyo.Constraint(m.N, m.R, rule=_terminal_central)

    def _mall(mm, t):
        return sum(mm.p[n, 1, t] + mm.p[n, 2, t] for n in mm.N) <= P_MALL
    m.c_mall = pyo.Constraint(m.T_idx, rule=_mall)

    def _obj(mm):
        return sum(W[n] * (mm.T[n, r, t] - T_REF) ** 2
                   for n in mm.N for r in mm.R for t in mm.T_idx)
    m.obj = pyo.Objective(rule=_obj, sense=pyo.minimize)

    _SOLVER.solve(m, tee=False)
    obj = float(pyo.value(m.obj))

    p_star = np.zeros((N, 2, T))
    for n in range(1, N + 1):
        for t in range(T):
            p_star[n - 1, 0, t] = _safe_value(m.p[n, 1, t])
            p_star[n - 1, 1, t] = _safe_value(m.p[n, 2, t])
    return obj, p_star


def distributed_run(alpha_schedule, num_iters=100):
    lam = np.zeros(T)
    obj_hist = []
    lam_hist = []
    slack_hist = []
    last_store_p = {}

    for k in range(num_iters):
        total_p = np.zeros(T)
        primal_obj = 0.0
        for n in range(1, N + 1):
            p1, p2, T1, T2 = solve_local(W[n], lam)
            last_store_p[n] = (p1, p2)
            total_p += p1 + p2
            primal_obj += W[n] * (np.sum((T1 - T_REF) ** 2) + np.sum((T2 - T_REF) ** 2))

        obj_hist.append(primal_obj)
        lam_hist.append(lam.copy())
        slack = total_p - P_MALL
        slack_hist.append(slack.copy())

        a = alpha_schedule(k)
        lam = np.maximum(0.0, lam + a * slack)

    return {
        'obj':    np.array(obj_hist),
        'lam':    np.array(lam_hist),
        'slack':  np.array(slack_hist),
        'last_p': last_store_p,
    }


CASES = [
    ('alpha_0p001', r'$\alpha=0.001$',     lambda k: 0.001),
    ('alpha_0p01',  r'$\alpha=0.01$',      lambda k: 0.01),
    ('alpha_0p1',   r'$\alpha=0.1$',       lambda k: 0.1),
    ('alpha_1',     r'$\alpha=1$',         lambda k: 1.0),
    ('alpha_10',    r'$\alpha=10$',        lambda k: 10.0),
    ('adaptive',    r'$\alpha_k=5/(1+k)$', lambda k: 5.0 / (1 + k)),
]


def main(num_iters=100, out_dir=None):
    if out_dir is None:
        out_dir = Path(__file__).resolve().parent / 'figures'
    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True)

    results = {}
    for fn_key, _, sched in CASES:
        print(f'running {fn_key}...')
        results[fn_key] = distributed_run(sched, num_iters)

    print('running centralised...')
    opt_obj, p_star = solve_centralised()
    print(f'centralised optimum: {opt_obj:.4f}')

    fig, ax = plt.subplots(figsize=(9, 5.5))
    colours = plt.cm.viridis(np.linspace(0.05, 0.95, len(CASES)))
    for (fn_key, lbl, _), c in zip(CASES, colours):
        ax.plot(results[fn_key]['obj'], label=lbl, color=c, lw=1.6)
    ax.axhline(opt_obj, color='k', ls='--', lw=1.0,
               label=f'centralised optimum ({opt_obj:.1f})')
    ax.set_xlabel('iteration $k$')
    ax.set_ylabel('primal objective')
    ax.set_title('Primal objective vs iteration for six step-size choices')
    ax.set_yscale('symlog', linthresh=1.0)
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, ls=':')
    plt.tight_layout()
    plt.savefig(out_dir / 'task7_objective.png', dpi=140)
    plt.close()

    for fn_key, lbl, _ in CASES:
        r = results[fn_key]
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
        colours_t = plt.cm.viridis(np.linspace(0.05, 0.95, T))
        for t in range(T):
            ax1.plot(r['lam'][:, t], color=colours_t[t], lw=1.2, label=f't={t}')
            ax2.plot(r['slack'][:, t], color=colours_t[t], lw=1.2, label=f't={t}')
        ax1.set_ylabel(r'$\lambda_t$')
        ax1.set_title(f'Multiplier and slack evolution: {lbl}')
        ax1.grid(True, ls=':')
        ax1.legend(ncol=5, fontsize=7, loc='best')
        ax2.axhline(0, color='k', lw=0.6)
        ax2.set_xlabel('iteration $k$')
        ax2.set_ylabel(r'$\sum_n p_{n,t} - P^{\mathrm{mall}}$')
        ax2.grid(True, ls=':')
        plt.tight_layout()
        plt.savefig(out_dir / f'task7_lamslack_{fn_key}.png', dpi=140)
        plt.close()

    totals = p_star.sum(axis=(1, 2))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(np.arange(1, N + 1), totals, color='steelblue')
    ax.set_xticks(np.arange(1, N + 1))
    ax.set_xlabel('store $n$')
    ax.set_ylabel('total heater energy over the day (kWh)')
    ax.set_title('Centralised solution: per-store total energy')
    ax.grid(True, axis='y', ls=':')
    plt.tight_layout()
    plt.savefig(out_dir / 'task7_per_store_energy.png', dpi=140)
    plt.close()

    results_path = out_dir / 'task7_summary.txt'
    with open(results_path, 'w') as f:
        f.write(f'Centralised optimum: {opt_obj:.6f}\n\n')
        f.write('Final objective and max |slack| over t in [0, T-2]:\n')
        for fn_key, lbl, _ in CASES:
            r = results[fn_key]
            slack_no_terminal = r['slack'][-1, :-1]
            f.write(f'  {fn_key:14s}: '
                    f'obj={r["obj"][-1]:12.4f}  '
                    f'max|slack|={np.max(np.abs(slack_no_terminal)):8.4f}\n')
        f.write('\nCentralised: total heater energy per store:\n')
        for n in range(N):
            f.write(f'  store n={n+1:2d} (w_n={W[n+1]:2d}): {totals[n]:.4f} kWh\n')

    print(f'\nWrote summary to {results_path}')
    return results, opt_obj


if __name__ == '__main__':
    main()
