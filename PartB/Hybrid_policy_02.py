import numpy as np
from collections import deque

import pyomo.environ as pyo
from pyomo.opt import SolverFactory, TerminationCondition

import v2_SystemCharacteristics as SystemCharacteristics


_PARAMS = SystemCharacteristics.get_fixed_data()
_T_TOTAL = int(_PARAMS["num_timeslots"])

_MAX_L = 5
_BRANCHING = {
    1: [],
    2: [6],
    3: [3, 2],
    4: [3, 2, 2],
    5: [3, 2, 2, 2],
}
_M_PER_STAGE = [12, 10, 10, 10]

_ETA = np.array([[ 1.40381742e+02, -4.40269607e+00, -4.36370499e+00,
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

_SOLVER = SolverFactory("gurobi")
_SOLVER.options["TimeLimit"] = 8
_SOLVER.options["MIPGap"] = 0.01
_SOLVER.options["OutputFlag"] = 0


def _price_sample(current_price, previous_price):
    mean_price = 4.0
    reversion_strength = 0.12
    price_cap = 12.0
    price_floor = 0.0

    noise = np.random.normal(0.0, 0.5)
    mean_reversion = reversion_strength * (mean_price - current_price)
    next_price = current_price + 0.6 * (current_price - previous_price) + mean_reversion + noise

    if next_price < 0 and np.random.rand() > 0.2:
        next_price = np.random.uniform(0.0, mean_price * 0.3)

    return float(max(min(next_price, price_cap), price_floor))


def _occ_sample(r1, r2):
    mean1, mean2 = 35.0, 25.0
    rev = 0.25
    coupling = 0.1

    r1_next = r1 + rev * (mean1 - r1) + coupling * (r2 - r1) + np.random.normal(0.0, 3.0)
    r2_next = r2 + rev * (mean2 - r2) + coupling * (r1 - r2) + np.random.normal(0.0, 2.5)

    return float(np.clip(r1_next, 20, 50)), float(np.clip(r2_next, 10, 30))


def _kmedoids(X, k, max_iter=20):
    n = X.shape[0]
    if k >= n:
        return np.arange(n), np.arange(n)

    centroid = X.mean(axis=0)
    seed0 = int(np.argmin(np.linalg.norm(X - centroid, axis=1)))
    medoid_idx = [seed0]

    for _ in range(1, k):
        min_d = np.full(n, np.inf)
        for mi in medoid_idx:
            d = np.linalg.norm(X - X[mi], axis=1)
            min_d = np.minimum(min_d, d)
        medoid_idx.append(int(np.argmax(min_d)))

    medoid_idx = np.array(medoid_idx, dtype=int)

    for _ in range(max_iter):
        D = np.stack([np.linalg.norm(X - X[mi], axis=1) for mi in medoid_idx], axis=1)
        labels = np.argmin(D, axis=1)
        new_med = medoid_idx.copy()

        for j in range(k):
            members = np.where(labels == j)[0]
            if len(members) == 0:
                continue
            Xm = X[members]
            diff = Xm[:, None, :] - Xm[None, :, :]
            DD = np.sqrt((diff * diff).sum(axis=-1))
            new_med[j] = members[int(np.argmin(DD.sum(axis=1)))]

        if np.array_equal(new_med, medoid_idx):
            break
        medoid_idx = new_med

    D = np.stack([np.linalg.norm(X - X[mi], axis=1) for mi in medoid_idx], axis=1)
    labels = np.argmin(D, axis=1)
    return labels, medoid_idx


def _build_tree(state, L):
    if L > _MAX_L:
        L = _MAX_L

    branching = _BRANCHING.get(L, [])

    nodes = [{
        "id": 0,
        "stage": 0,
        "parent": None,
        "prob": 1.0,
        "price": float(state["price_t"]),
        "price_prev": float(state["price_previous"]),
        "occ1": float(state["Occ1"]),
        "occ2": float(state["Occ2"]),
    }]

    q = deque([0])
    while q:
        pid = q.popleft()
        par = nodes[pid]
        s = par["stage"]

        if s >= L - 1:
            continue

        b = branching[s]
        M_s = _M_PER_STAGE[s] if s < len(_M_PER_STAGE) else _M_PER_STAGE[-1]
        M_s = max(M_s, b)

        raw = np.empty((M_s, 3), dtype=float)
        for i in range(M_s):
            p_next = _price_sample(par["price"], par["price_prev"])
            o1_next, o2_next = _occ_sample(par["occ1"], par["occ2"])
            raw[i, 0] = p_next
            raw[i, 1] = o1_next
            raw[i, 2] = o2_next

        mu = raw.mean(axis=0)
        sd = raw.std(axis=0)
        sd = np.where(sd < 1e-9, 1.0, sd)
        Xz = (raw - mu) / sd

        labels, medoid_idx = _kmedoids(Xz, b)

        for j in range(b):
            members = np.where(labels == j)[0]
            if len(members) == 0:
                continue
            prob_frac = len(members) / M_s
            mi = medoid_idx[j]
            p_val, o1_val, o2_val = raw[mi, 0], raw[mi, 1], raw[mi, 2]

            nodes.append({
                "id": len(nodes),
                "stage": s + 1,
                "parent": pid,
                "prob": par["prob"] * prob_frac,
                "price": float(p_val),
                "price_prev": par["price"],
                "occ1": float(o1_val),
                "occ2": float(o2_val),
            })
            q.append(nodes[-1]["id"])

    return nodes


def _solve(state, nodes, cur_time):
    prm = _PARAMS
    Pmax = prm["heating_max_power"]
    Pvent = prm["ventilation_power"]

    zx = prm["heat_exchange_coeff"]
    zl = prm["thermal_loss_coeff"]
    zh = prm["heating_efficiency_coeff"]
    zc = prm["heat_vent_coeff"]
    zo = prm["heat_occupancy_coeff"]

    hoe = prm["humidity_occupancy_coeff"]
    hve = prm["humidity_vent_coeff"]

    T_low = prm["temp_min_comfort_threshold"]
    T_OK = prm["temp_OK_threshold"]
    T_high = prm["temp_max_comfort_threshold"]
    H_high = prm["humidity_threshold"]
    outdoor = prm["outdoor_temperature"]

    M_T = 60.0
    M_H = 150.0

    def t_out(stage):
        idx = min(max(cur_time + stage, 0), len(outdoor) - 1)
        return outdoor[idx]

    m = pyo.ConcreteModel()
    N_ids = [n["id"] for n in nodes]
    m.N = pyo.Set(initialize=N_ids, ordered=True)
    m.R = pyo.Set(initialize=[1, 2], ordered=True)

    children_of = {n["id"]: [] for n in nodes}
    for n in nodes[1:]:
        children_of[n["parent"]].append(n["id"])
    leaf_ids = [n["id"] for n in nodes if not children_of[n["id"]]]
    m.L = pyo.Set(initialize=leaf_ids, ordered=True)

    m.p = pyo.Var(m.N, m.R, bounds=(0, Pmax))
    m.v = pyo.Var(m.N, within=pyo.Binary)
    m.s = pyo.Var(m.N, within=pyo.Binary)
    m.T = pyo.Var(m.N, m.R, bounds=(-20, 60))
    m.H = pyo.Var(m.N, bounds=(-50, 200))
    m.yL = pyo.Var(m.N, m.R, within=pyo.Binary)
    m.yH = pyo.Var(m.N, m.R, within=pyo.Binary)
    m.yOK = pyo.Var(m.N, m.R, within=pyo.Binary)
    m.u = pyo.Var(m.N, m.R, within=pyo.Binary)

    m.T_post = pyo.Var(m.L, m.R, bounds=(-20, 60))
    m.H_post = pyo.Var(m.L, bounds=(-50, 200))

    T1_0 = float(state["T1"])
    T2_0 = float(state["T2"])
    H_0 = float(state["H"])

    m.c_T1 = pyo.Constraint(expr=m.T[0, 1] == T1_0)
    m.c_T2 = pyo.Constraint(expr=m.T[0, 2] == T2_0)
    m.c_H = pyo.Constraint(expr=m.H[0] == H_0)

    lo1 = int(state.get("low_override_r1", 0))
    lo2 = int(state.get("low_override_r2", 0))
    u1_0 = 1 if (lo1 == 1 or T1_0 <= T_low) else 0
    u2_0 = 1 if (lo2 == 1 or T2_0 <= T_low) else 0
    m.c_u1_0 = pyo.Constraint(expr=m.u[0, 1] == u1_0)
    m.c_u2_0 = pyo.Constraint(expr=m.u[0, 2] == u2_0)

    vc = int(state.get("vent_counter", 0))
    k_past = 0
    if vc == 1:
        k_past = 2
    elif vc == 2:
        k_past = 1

    if k_past > 0:
        def _past_up(m, nid):
            if nodes[nid]["stage"] < k_past:
                return m.v[nid] == 1
            return pyo.Constraint.Skip
        m.c_pastup = pyo.Constraint(m.N, rule=_past_up)

    if vc == 0:
        m.c_s0 = pyo.Constraint(expr=m.s[0] == m.v[0])
    else:
        m.c_s0 = pyo.Constraint(expr=m.s[0] == 0)

    def _temp_dyn(m, nid, r):
        if nid == 0:
            return pyo.Constraint.Skip
        pid = nodes[nid]["parent"]
        par = nodes[pid]
        rp = 2 if r == 1 else 1
        occ_r = par["occ1"] if r == 1 else par["occ2"]
        return m.T[nid, r] == (
            m.T[pid, r]
            + zx * (m.T[pid, rp] - m.T[pid, r])
            - zl * (m.T[pid, r] - t_out(par["stage"]))
            + zh * m.p[pid, r]
            - zc * m.v[pid]
            + zo * occ_r
        )
    m.c_Tdyn = pyo.Constraint(m.N, m.R, rule=_temp_dyn)

    def _hum_dyn(m, nid):
        if nid == 0:
            return pyo.Constraint.Skip
        pid = nodes[nid]["parent"]
        par = nodes[pid]
        return m.H[nid] == m.H[pid] + hoe * (par["occ1"] + par["occ2"]) - hve * m.v[pid]
    m.c_Hdyn = pyo.Constraint(m.N, rule=_hum_dyn)

    def _yL_ub(m, nid, r):
        return m.T[nid, r] <= T_low + M_T * (1 - m.yL[nid, r])
    def _yL_lb(m, nid, r):
        return m.T[nid, r] >= T_low - M_T * m.yL[nid, r]
    m.c_yL_ub = pyo.Constraint(m.N, m.R, rule=_yL_ub)
    m.c_yL_lb = pyo.Constraint(m.N, m.R, rule=_yL_lb)

    def _yH_lb(m, nid, r):
        return m.T[nid, r] >= T_high - M_T * (1 - m.yH[nid, r])
    def _yH_ub(m, nid, r):
        return m.T[nid, r] <= T_high + M_T * m.yH[nid, r]
    m.c_yH_lb = pyo.Constraint(m.N, m.R, rule=_yH_lb)
    m.c_yH_ub = pyo.Constraint(m.N, m.R, rule=_yH_ub)

    def _yOK_lb(m, nid, r):
        return m.T[nid, r] >= T_OK - M_T * (1 - m.yOK[nid, r])
    def _yOK_ub(m, nid, r):
        return m.T[nid, r] <= T_OK + M_T * m.yOK[nid, r]
    m.c_yOK_lb = pyo.Constraint(m.N, m.R, rule=_yOK_lb)
    m.c_yOK_ub = pyo.Constraint(m.N, m.R, rule=_yOK_ub)

    def _u_a(m, nid, r):
        if nid == 0:
            return pyo.Constraint.Skip
        return m.u[nid, r] >= m.yL[nid, r]
    def _u_b(m, nid, r):
        if nid == 0:
            return pyo.Constraint.Skip
        return m.u[nid, r] >= m.u[nodes[nid]["parent"], r] - m.yOK[nid, r]
    def _u_c(m, nid, r):
        if nid == 0:
            return pyo.Constraint.Skip
        return m.u[nid, r] <= m.yL[nid, r] + m.u[nodes[nid]["parent"], r]
    def _u_d(m, nid, r):
        if nid == 0:
            return pyo.Constraint.Skip
        return m.u[nid, r] <= m.yL[nid, r] + 1 - m.yOK[nid, r]
    m.c_u_a = pyo.Constraint(m.N, m.R, rule=_u_a)
    m.c_u_b = pyo.Constraint(m.N, m.R, rule=_u_b)
    m.c_u_c = pyo.Constraint(m.N, m.R, rule=_u_c)
    m.c_u_d = pyo.Constraint(m.N, m.R, rule=_u_d)

    def _heat_hi(m, nid, r):
        return m.p[nid, r] <= Pmax * (1 - m.yH[nid, r])
    def _heat_lo(m, nid, r):
        return m.p[nid, r] >= Pmax * (m.u[nid, r] - m.yH[nid, r])
    m.c_heat_hi = pyo.Constraint(m.N, m.R, rule=_heat_hi)
    m.c_heat_lo = pyo.Constraint(m.N, m.R, rule=_heat_lo)

    def _hum_force(m, nid):
        return m.H[nid] <= H_high + M_H * m.v[nid]
    m.c_hum_force = pyo.Constraint(m.N, rule=_hum_force)

    def _s_a(m, nid):
        if nid == 0:
            return pyo.Constraint.Skip
        return m.s[nid] >= m.v[nid] - m.v[nodes[nid]["parent"]]
    def _s_b(m, nid):
        if nid == 0:
            return pyo.Constraint.Skip
        return m.s[nid] <= m.v[nid]
    def _s_c(m, nid):
        if nid == 0:
            return pyo.Constraint.Skip
        return m.s[nid] <= 1 - m.v[nodes[nid]["parent"]]
    m.c_s_a = pyo.Constraint(m.N, rule=_s_a)
    m.c_s_b = pyo.Constraint(m.N, rule=_s_b)
    m.c_s_c = pyo.Constraint(m.N, rule=_s_c)

    def _mu_p(m, nid):
        if nid == 0:
            return pyo.Constraint.Skip
        return m.v[nid] >= m.s[nodes[nid]["parent"]]
    def _mu_gp(m, nid):
        if nid == 0:
            return pyo.Constraint.Skip
        pid = nodes[nid]["parent"]
        gpid = nodes[pid]["parent"]
        if gpid is None:
            return pyo.Constraint.Skip
        return m.v[nid] >= m.s[gpid]
    m.c_mu_p = pyo.Constraint(m.N, rule=_mu_p)
    m.c_mu_gp = pyo.Constraint(m.N, rule=_mu_gp)

    def _Tpost_rule(m, lid, r):
        n = nodes[lid]
        rp = 2 if r == 1 else 1
        occ_r = n["occ1"] if r == 1 else n["occ2"]
        return m.T_post[lid, r] == (
            m.T[lid, r]
            + zx * (m.T[lid, rp] - m.T[lid, r])
            - zl * (m.T[lid, r] - t_out(n["stage"]))
            + zh * m.p[lid, r]
            - zc * m.v[lid]
            + zo * occ_r
        )
    m.c_Tpost = pyo.Constraint(m.L, m.R, rule=_Tpost_rule)

    def _Hpost_rule(m, lid):
        n = nodes[lid]
        return m.H_post[lid] == m.H[lid] + hoe * (n["occ1"] + n["occ2"]) - hve * m.v[lid]
    m.c_Hpost = pyo.Constraint(m.L, rule=_Hpost_rule)

    def _obj(m):
        sp_cost = sum(
            nodes[i]["prob"] * nodes[i]["price"] *
            (Pvent * m.v[i] + sum(m.p[i, r] for r in m.R))
            for i in m.N
        )
        vfa_cost = 0.0
        for lid in leaf_ids:
            n = nodes[lid]
            tau_next = cur_time + n["stage"] + 1
            if tau_next >= _T_TOTAL:
                continue
            th = _ETA[tau_next]
            feats_expr = (
                th[0]
                + th[1] * m.T_post[lid, 1]
                + th[2] * m.T_post[lid, 2]
                + th[3] * m.H_post[lid]
                + th[4] * n["price"]
                + th[5] * n["occ1"]
                + th[6] * n["occ2"]
            )
            vfa_cost = vfa_cost + n["prob"] * feats_expr
        return sp_cost + vfa_cost

    m.Obj = pyo.Objective(rule=_obj, sense=pyo.minimize)

    res = _SOLVER.solve(m, tee=False)
    tc = res.solver.termination_condition
    if tc not in (
        TerminationCondition.optimal,
        TerminationCondition.feasible,
        TerminationCondition.maxTimeLimit,
    ):
        raise RuntimeError(f"Solver returned {tc}")

    p1_val = pyo.value(m.p[0, 1])
    p2_val = pyo.value(m.p[0, 2])
    v_val = pyo.value(m.v[0])

    if p1_val is None or p2_val is None or v_val is None:
        raise RuntimeError("No solution extracted from the model")

    return float(p1_val), float(p2_val), int(round(v_val))


_EXCEPTION_REPORTED = False


def select_action(state):
    global _EXCEPTION_REPORTED

    cur_time = int(state.get("current_time", 0))
    remaining = max(1, _T_TOTAL - cur_time)
    L = min(remaining, _MAX_L)

    try:
        nodes = _build_tree(state, L)
        p1, p2, v = _solve(state, nodes, cur_time)
    except Exception:
        if not _EXCEPTION_REPORTED:
            import traceback
            traceback.print_exc()
            _EXCEPTION_REPORTED = True
        p1, p2, v = 0.0, 0.0, 0

    Pmax = _PARAMS["heating_max_power"]
    return {
        "HeatPowerRoom1": float(max(0.0, min(Pmax, p1))),
        "HeatPowerRoom2": float(max(0.0, min(Pmax, p2))),
        "VentilationON": int(v),
    }


class Policy:
    def select_action(self, state):
        return select_action(state)
