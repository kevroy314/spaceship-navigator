"""Relations between the difficulty measures and per-instance success.

For every task in the four frozen 150 s evaluation suites (data/eval/suites.pkl,
2,388 tasks) this computes the closed-form quantities the paper uses to describe
a level -- eta at each target, the task-level demand D = eta / A and authority A
from `spacenav.demand.terms` at the 150 s clock, the family, the number of
targets, the rendezvous flag, the order-free flag and the distance to the first
target -- and joins them to the per-task outcome of the final checkpoint of each
local training run (data/rl/<run>/traj_03000.pkl, `per_task`) and to the
reference pilot's stored outcome.  It then reports:

  1. Spearman correlations of agent and pilot success with the measures, pooled
     and within each suite;
  2. logistic models of agent success on family alone, D alone, family + D and
     family + D + targets + rendezvous, with and without suite dummies, scored
     by held-out deviance and AUC under 10-fold cross-validation, plus the
     likelihood-ratio test of each nesting on the full data;
  3. agent and pilot success by D decile with 95% Wilson intervals;
  4. the share of clock-sensitive tasks (a coasting craft first contacts a target
     only after 150 s) among single- and multi-target instances, from
     paper/figures/noop_data.npz.  Those were measured on the 600 s suites,
     which are different levels from the 150 s suites, so this part is not
     joined to the agent outcomes.

    python scripts/correlations.py            # measure, analyse, draw
    python scripts/correlations.py --analyse  # reuse paper/figures/corr_data.npz

The physics part is a few thousand `gravity_at` calls; it runs on the CPU
backend by default so that it never competes with a training job for the card.
"""
import argparse
import os
import pathlib
import pickle
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "paper" / "figures"
DATA = OUT / "corr_data.npz"

SUITES = ["easy", "medium", "hard", "blind"]
LBL = {"easy": "Easy", "medium": "Medium", "hard": "Hard", "blind": "Blind"}
# final checkpoints with per-task outcomes on disk: two seeds of the no-channel
# arm and two of the all-four arm.  ppo_v3 is excluded: a different lineage
# (no instrument channels existed, MAX_WAYPOINTS 4) whose per-task outcomes are
# not comparable with the paper's tables.
RUNS = ["ppo_v5_noinstr", "ppo_v5_noinstr_s1", "ppo_v4", "ppo_v4_s1"]
ARRIVED = 1
CLOCK_S = 150.0


# ------------------------------------------------------------------ measure
def measure():
    import jax
    import jax.numpy as jnp
    from spacenav import demand
    from spacenav import env as E
    from spacenav import physics as P
    from spacenav.levels.build import load_pool

    suites = pickle.load(open(ROOT / "data" / "eval" / "suites.pkl", "rb"))
    pools = {}
    out = {"runs": np.array(RUNS)}

    def per_task(level, task):
        bp, _ = E.snapshot_state(level, task.snapshot)
        wpos, _ = E.waypoint_states(task, bp, bp * 0.0)
        full = P.gravity_at(wpos, bp, level, 0)
        dom = P.gravity_at(wpos, bp, level, 1)
        gmag = jnp.sqrt(jnp.sum(full * full, -1) + 1e-12)
        eta = jnp.sqrt(jnp.sum((full - dom) ** 2, -1) + 1e-18) / gmag
        eta = jnp.where(task.wp_active, eta, jnp.nan)
        g_wp = jnp.where(task.wp_active, gmag, jnp.nan)
        dist = jnp.sqrt(jnp.sum((wpos - task.start_pos[None]) ** 2, -1))
        # the first target: the first in sequence, or the nearest when the tour
        # may be flown in any order (env.current_leg at the start)
        k = wpos.shape[0]
        seq = jnp.where(task.wp_active, jnp.arange(k), k + 10)
        near = jnp.where(task.wp_active, dist, jnp.inf)
        first = jnp.where(task.order_free, jnp.argmin(near), jnp.argmin(seq))
        t = demand.terms(level, task, CLOCK_S)
        return dict(eta_wp=eta, g_wp=g_wp, first_dist=dist[first],
                    D=t.D, A=t.A, eta_bar=t.eta, g_bar=t.g, T=t.T, dv=t.dv,
                    feasible=t.feasible)

    f = jax.jit(jax.vmap(per_task))
    for s in SUITES:
        sd = suites[s]
        if sd["pool"] not in pools:
            pools[sd["pool"]] = load_pool(ROOT / "data" / "pools" / f"{sd['pool']}.npz")[0]
        idx = np.asarray(sd["level_idx"])
        levels = jax.tree.map(lambda x: x[idx], pools[sd["pool"]])
        task = sd["task"]
        r = jax.tree.map(np.asarray, f(levels, task))
        n = len(idx)
        wa = np.asarray(task.wp_active)
        vtol = np.asarray(task.wp_v_tol)
        rdv = (wa & (vtol < 100.0)).any(1)
        arr = []
        for run in RUNS:
            tr = pickle.load(open(ROOT / "data" / "rl" / run / "traj_03000.pkl", "rb"))
            st = np.asarray(tr["per_task"][s]["status"])
            assert len(st) == n, (run, s, len(st), n)
            arr.append(st == ARRIVED)
        arr = np.stack(arr, 1)                                   # (n, runs)
        pilot = np.asarray(sd["pilot"]["status"]) == ARRIVED
        p = f"suite_{s}/"
        out.update({
            p + "level_idx": idx, p + "family": np.asarray(sd["family"]),
            p + "n_targets": wa.sum(1).astype(np.int8),
            p + "rendezvous": rdv, p + "order_free": np.asarray(task.order_free).astype(bool),
            p + "eta_wp": r["eta_wp"].astype(np.float32), p + "g_wp": r["g_wp"].astype(np.float32),
            p + "first_dist": r["first_dist"].astype(np.float32),
            p + "D": r["D"].astype(np.float32), p + "A": r["A"].astype(np.float32),
            p + "eta_bar": r["eta_bar"].astype(np.float32), p + "g_bar": r["g_bar"].astype(np.float32),
            p + "T": r["T"].astype(np.float32), p + "dv": r["dv"].astype(np.float32),
            p + "feasible": r["feasible"].astype(bool),
            p + "agent_arrived": arr, p + "pilot_arrived": pilot,
        })
        print(f"{s}: n={n}, agent {arr.mean(0).round(3)}, pilot {pilot.mean():.3f}, "
              f"median D {np.median(r['D']):.4f}, median max-eta {np.median(np.nanmax(r['eta_wp'], 1)):.4f}")
    # clock-sensitivity, from the zero-action measurement on the 600 s suites
    nd = np.load(OUT / "noop_data.npz")
    for s in SUITES:
        for k in ("first_contact_s", "n_targets", "family", "rendezvous", "status"):
            out[f"noop_{s}/{k}"] = nd[f"suite_{s}/{k}"]
        out[f"noop_{s}/clock_s"] = nd[f"suite_{s}/clock_s"]
    np.savez_compressed(DATA, **out)
    print("wrote", DATA)


# ------------------------------------------------------------------ analyse
def load():
    d = np.load(DATA, allow_pickle=False)
    rows = {}
    keys = ["level_idx", "family", "n_targets", "rendezvous", "order_free", "first_dist",
            "D", "A", "eta_bar", "g_bar", "T", "dv", "feasible", "agent_arrived",
            "pilot_arrived", "eta_wp", "g_wp"]
    for k in keys:
        rows[k] = np.concatenate([d[f"suite_{s}/{k}"] for s in SUITES], 0)
    rows["suite"] = np.concatenate([[s] * len(d[f"suite_{s}/D"]) for s in SUITES])
    rows["eta_max"] = np.nanmax(rows["eta_wp"], 1)
    rows["agent"] = rows["agent_arrived"].mean(1)
    rows["pilot"] = rows["pilot_arrived"].astype(float)
    rows["runs"] = [str(x) for x in d["runs"]]
    noop = {s: {k: d[f"noop_{s}/{k}"] for k in
                ("first_contact_s", "n_targets", "family", "rendezvous", "status", "clock_s")}
            for s in SUITES}
    return rows, noop


def wilson(k, n, z=1.96):
    k, n = np.asarray(k, float), np.asarray(n, float)
    p = k / np.maximum(n, 1)
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return p, c - h, c + h


def auc(score, y):
    """Mann-Whitney AUC: P(score of a success > score of a failure)."""
    from scipy.stats import rankdata
    y = np.asarray(y, bool)
    r = rankdata(score)
    n1, n0 = y.sum(), (~y).sum()
    if n1 == 0 or n0 == 0:
        return np.nan
    return (r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def fit_logit(X, k, m, ridge=1e-3):
    """Binomial logistic regression: k successes of m trials per row.  A light
    ridge on the non-intercept weights keeps separated family cells finite."""
    from scipy.optimize import minimize
    Xb = np.concatenate([np.ones((len(X), 1)), X], 1)
    mask = np.ones(Xb.shape[1]); mask[0] = 0

    def nll(w):
        z = Xb @ w
        ll = k * z - m * np.logaddexp(0, z)
        return -ll.sum() + 0.5 * ridge * np.sum(mask * w * w)

    def grad(w):
        z = Xb @ w
        p = 1 / (1 + np.exp(-z))
        return -(Xb.T @ (k - m * p)) + ridge * mask * w

    r = minimize(nll, np.zeros(Xb.shape[1]), jac=grad, method="L-BFGS-B")
    return r.x


def predict(w, X):
    Xb = np.concatenate([np.ones((len(X), 1)), X], 1)
    return 1 / (1 + np.exp(-(Xb @ w)))


def deviance(p, k, m):
    """Binomial deviance per trial, -2 log-likelihood / sum(m)."""
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return -2 * np.sum(k * np.log(p) + (m - k) * np.log(1 - p)) / m.sum()


def design(rows, terms):
    cols, names = [], []
    fams = sorted(set(rows["family"]))
    suites = SUITES
    for t in terms:
        if t == "family":
            for f in fams[1:]:
                cols.append((rows["family"] == f).astype(float)); names.append("fam:" + f)
        elif t == "suite":
            for s in suites[1:]:
                cols.append((rows["suite"] == s).astype(float)); names.append("suite:" + s)
        elif t == "logD":
            cols.append(np.log10(rows["D"])); names.append("log10 D")
        elif t == "logeta":
            cols.append(np.log10(rows["eta_max"])); names.append("log10 eta_max")
        elif t == "logA":
            cols.append(np.log10(rows["A"])); names.append("log10 A")
        elif t == "nt":
            cols.append(rows["n_targets"].astype(float)); names.append("n_targets")
        elif t == "rdv":
            cols.append(rows["rendezvous"].astype(float)); names.append("rendezvous")
        elif t == "dist":
            cols.append(rows["first_dist"] / 100.0); names.append("first_dist/100")
        else:
            raise KeyError(t)
    X = np.stack(cols, 1) if cols else np.zeros((len(rows["D"]), 0))
    # standardise continuous columns so the ridge treats them alike
    for j, nm in enumerate(names):
        if not (nm.startswith("fam:") or nm.startswith("suite:")):
            X[:, j] = (X[:, j] - X[:, j].mean()) / (X[:, j].std() + 1e-12)
    return X, names


def cv_model(rows, terms, k, m, folds=10, reps=5, seed=0):
    X, names = design(rows, terms)
    n = len(k)
    devs, aucs = [], []
    for rep in range(reps):
        rng = np.random.default_rng(seed + rep)
        perm = rng.permutation(n)
        p_out = np.zeros(n)
        for f in range(folds):
            te = perm[f::folds]
            tr = np.setdiff1d(perm, te)
            w = fit_logit(X[tr], k[tr], m[tr])
            p_out[te] = predict(w, X[te])
        devs.append(deviance(p_out, k, m))
        # AUC against the trial-level binary outcome: each task contributes m
        # outcomes (one per run), all scored with the task's held-out p
        sc = np.repeat(p_out, m.astype(int))
        yb = np.concatenate([np.r_[np.ones(int(a)), np.zeros(int(b - a))] for a, b in zip(k, m)])
        aucs.append(auc(sc, yb))
    w_full = fit_logit(X, k, m)
    dev_full = deviance(predict(w_full, X), k, m) * m.sum()
    return dict(terms=terms, names=names, dev_cv=float(np.mean(devs)), dev_cv_sd=float(np.std(devs)),
                auc_cv=float(np.mean(aucs)), auc_cv_sd=float(np.std(aucs)),
                dev_full=float(dev_full), df=X.shape[1] + 1, w=w_full)


def analyse(draw=True):
    from scipy.stats import spearmanr, fisher_exact, chi2
    rows, noop = load()
    n = len(rows["D"])
    R = {}  # everything reported, for the npz / the coordinator
    print("=" * 78)
    print(f"tasks: {n} across {SUITES}; agent outcomes from runs {rows['runs']} "
          f"(final checkpoint, update 3000); pilot from suites.pkl")
    for s in SUITES:
        sel = rows["suite"] == s
        print(f"  {s:7s} n={sel.sum():4d}  agent {100*rows['agent'][sel].mean():5.1f}%  "
              f"pilot {100*rows['pilot'][sel].mean():5.1f}%  median D {np.median(rows['D'][sel]):.4f} "
              f"[IQR {np.percentile(rows['D'][sel],25):.4f}-{np.percentile(rows['D'][sel],75):.4f}]  "
              f"median eta_max {np.median(rows['eta_max'][sel]):.4f}  median A {np.median(rows['A'][sel]):.2f}  "
              f"n_targets {np.bincount(rows['n_targets'][sel])[1:].tolist()}  rdv {rows['rendezvous'][sel].sum()}  "
              f"order_free {rows['order_free'][sel].sum()}  infeasible {(~rows['feasible'][sel]).sum()}")
    print(f"  per-run suite success: " + "; ".join(
        f"{r}: " + "/".join(f"{100*rows['agent_arrived'][rows['suite']==s, i].mean():.1f}" for s in SUITES)
        for i, r in enumerate(rows["runs"])))

    # ---- 1. Spearman
    print("\n1. Spearman rank correlations (rho, p); agent success = fraction of 4 runs arrived")
    feats = [("eta_max", "max eta at a target"), ("eta_bar", "eta (route mean, demand.terms)"),
             ("D", "demand D"), ("A", "authority A"), ("n_targets", "targets"),
             ("first_dist", "first-target distance"), ("rendezvous", "rendezvous flag")]
    strata = [("pooled", np.ones(n, bool))] + [(s, rows["suite"] == s) for s in SUITES]
    for who in ("agent", "pilot"):
        print(f"  {who}:")
        hdr = "    {:24s}".format("") + "".join(f"{nm:>22s}" for nm, _ in strata)
        print(hdr)
        for key, label in feats:
            line = f"    {label:24s}"
            for nm, sel in strata:
                x, y = rows[key][sel].astype(float), rows[who][sel]
                if np.std(x) == 0 or np.std(y) == 0:
                    line += f"{'--':>22s}"; R[f"rho_{who}_{key}_{nm}"] = np.nan; continue
                rho, p = spearmanr(x, y)
                R[f"rho_{who}_{key}_{nm}"] = rho; R[f"p_{who}_{key}_{nm}"] = p
                line += f"{rho:+.3f} (p={p:.1e})".rjust(22)
            print(line)
    rho, p = spearmanr(rows["agent"], rows["pilot"]); R["rho_agent_pilot"] = rho; R["p_agent_pilot"] = p
    print(f"  agent vs pilot success: rho={rho:+.3f} (p={p:.1e})")
    rho, p = spearmanr(np.log10(rows["D"]), rows["n_targets"])
    print(f"  D vs n_targets (pooled): rho={rho:+.3f}; D vs A: rho={spearmanr(rows['D'], rows['A'])[0]:+.3f}; "
          f"D vs eta_max: rho={spearmanr(rows['D'], rows['eta_max'])[0]:+.3f}")

    # ---- 2. models
    k = rows["agent_arrived"].sum(1).astype(float)
    m = np.full(n, rows["agent_arrived"].shape[1], float)
    specs = [
        ("family",                  ["family"]),
        ("D",                       ["logD"]),
        ("eta_max",                 ["logeta"]),
        ("family + D",              ["family", "logD"]),
        ("family + targets + rdv",  ["family", "nt", "rdv"]),
        ("family + D + targets + rdv", ["family", "logD", "nt", "rdv"]),
        ("family + eta_max + A + targets + rdv", ["family", "logeta", "logA", "nt", "rdv"]),
        ("suite + family",          ["suite", "family"]),
        ("suite + family + D",      ["suite", "family", "logD"]),
        ("suite + family + targets + rdv", ["suite", "family", "nt", "rdv"]),
        ("suite + family + D + targets + rdv", ["suite", "family", "logD", "nt", "rdv"]),
        ("suite + family + D + targets + rdv + dist", ["suite", "family", "logD", "nt", "rdv", "dist"]),
        # does D's pull come from eta (the representation error) or from A (the
        # tank against the field's impulse over the estimated tour time)?
        ("suite + family + targets + rdv + A",        ["suite", "family", "nt", "rdv", "logA"]),
        ("suite + family + targets + rdv + eta_max",  ["suite", "family", "nt", "rdv", "logeta"]),
        ("suite + family + targets + rdv + A + eta_max", ["suite", "family", "nt", "rdv", "logA", "logeta"]),
    ]
    print("\n2. Logistic models of agent success (binomial, 4 runs per task), 10-fold CV x 5 repeats")
    print(f"   {'model':44s} {'df':>3s} {'held-out dev/trial':>19s} {'held-out AUC':>16s} {'full-data dev':>14s}")
    fits = {}
    for name, terms in specs:
        r = cv_model(rows, terms, k, m)
        fits[name] = r
        R[f"model_{name}_dev_cv"] = r["dev_cv"]; R[f"model_{name}_auc_cv"] = r["auc_cv"]
        R[f"model_{name}_dev_full"] = r["dev_full"]; R[f"model_{name}_df"] = r["df"]
        print(f"   {name:44s} {r['df']:3d} {r['dev_cv']:9.4f} +- {r['dev_cv_sd']:.4f} "
              f"{r['auc_cv']:8.3f} +- {r['auc_cv_sd']:.3f} {r['dev_full']:14.1f}")
    null = deviance(np.full(n, k.sum() / m.sum()), k, m)
    print(f"   {'(intercept only)':44s} {1:3d} {null:9.4f}")
    R["model_null_dev"] = null
    print("   likelihood-ratio tests of nestings (full data):")
    for a, b in [("family", "family + D"), ("family + targets + rdv", "family + D + targets + rdv"),
                 ("suite + family", "suite + family + D"),
                 ("suite + family + targets + rdv", "suite + family + D + targets + rdv"),
                 ("suite + family + D + targets + rdv", "suite + family + D + targets + rdv + dist"),
                 ("suite + family + targets + rdv", "suite + family + targets + rdv + A"),
                 ("suite + family + targets + rdv", "suite + family + targets + rdv + eta_max"),
                 ("suite + family + targets + rdv + A", "suite + family + targets + rdv + A + eta_max"),
                 ("suite + family + targets + rdv + eta_max", "suite + family + targets + rdv + A + eta_max")]:
        dd = fits[a]["dev_full"] - fits[b]["dev_full"]; df = fits[b]["df"] - fits[a]["df"]
        p = chi2.sf(dd, df); R[f"lrt_{a}__{b}"] = dd; R[f"lrtp_{a}__{b}"] = p
        print(f"     {a:40s} -> {b:44s}  dev drop {dd:7.1f} on {df} df, p={p:.1e}")
    for key in ("suite + family + D + targets + rdv", "suite + family + targets + rdv + A + eta_max"):
        w = fits[key]
        print(f"   coefficients, {key} (standardised continuous terms):")
        for nm, c in zip(w["names"], w["w"][1:]):
            if not nm.startswith("fam:"):
                print(f"     {nm:16s} {c:+.3f}")
                R[f"coef_{key}_{nm}"] = c
    # pilot, same comparison
    kp = rows["pilot"].astype(float); mp = np.ones(n)
    print("   pilot success, same models (held-out AUC):")
    for name in ("family", "D", "family + D", "suite + family", "suite + family + D"):
        r = cv_model(rows, dict(specs)[name], kp, mp)
        R[f"pilot_model_{name}_auc_cv"] = r["auc_cv"]; R[f"pilot_model_{name}_dev_cv"] = r["dev_cv"]
        print(f"     {name:44s} AUC {r['auc_cv']:.3f}  dev {r['dev_cv']:.4f}")

    # ---- 3. deciles of D
    print("\n3. Success by D decile (pooled; Wilson 95% on the per-task mean, n = tasks)")
    edges = np.quantile(rows["D"], np.linspace(0, 1, 11))
    dec = np.clip(np.searchsorted(edges, rows["D"], side="right") - 1, 0, 9)
    R["decile_edges"] = edges
    tab = []
    for i in range(10):
        sel = dec == i
        pa, la, ha = wilson(rows["agent"][sel].sum(), sel.sum())
        pp, lp, hp = wilson(rows["pilot"][sel].sum(), sel.sum())
        comp = {s: int((rows["suite"][sel] == s).sum()) for s in SUITES}
        tab.append((i, np.median(rows["D"][sel]), sel.sum(), pa, la, ha, pp, lp, hp))
        print(f"   decile {i}: D {edges[i]:.4f}-{edges[i+1]:.4f} (median {np.median(rows['D'][sel]):.4f}) "
              f"n={sel.sum():3d} agent {100*pa:5.1f} [{100*la:4.1f},{100*ha:4.1f}]  "
              f"pilot {100*pp:5.1f} [{100*lp:4.1f},{100*hp:4.1f}]  suites {comp}")
    R["decile_table"] = np.array(tab)
    # the same, within suite, in terciles
    print("   within-suite terciles of D (agent, pilot):")
    for s in SUITES:
        sel = rows["suite"] == s
        e = np.quantile(rows["D"][sel], [0, 1 / 3, 2 / 3, 1])
        t = np.clip(np.searchsorted(e, rows["D"][sel], side="right") - 1, 0, 2)
        line = f"     {s:7s}"
        for i in range(3):
            ss = t == i
            line += (f"  D<{e[i+1]:.3f}: agent {100*rows['agent'][sel][ss].mean():4.1f} "
                     f"pilot {100*rows['pilot'][sel][ss].mean():4.1f} (n={ss.sum()})")
            R[f"tercile_{s}_{i}_agent"] = rows["agent"][sel][ss].mean(); R[f"tercile_{s}_{i}_n"] = ss.sum()
        print(line)

    # ---- 3b. by family
    print("\n3b. Success by family (pooled across suites) with median D")
    fams = sorted(set(rows["family"]), key=lambda f: np.median(rows["D"][rows["family"] == f]))
    famtab = []
    for f in fams:
        sel = rows["family"] == f
        pa, la, ha = wilson(rows["agent"][sel].sum(), sel.sum())
        pp, lp, hp = wilson(rows["pilot"][sel].sum(), sel.sum())
        famtab.append((f, np.median(rows["D"][sel]), sel.sum(), pa, la, ha, pp, lp, hp,
                       np.median(rows["eta_max"][sel])))
        print(f"   {f:15s} n={sel.sum():3d} median D {np.median(rows['D'][sel]):.4f} median eta_max "
              f"{np.median(rows['eta_max'][sel]):.4f}  agent {100*pa:5.1f} [{100*la:4.1f},{100*ha:4.1f}]  "
              f"pilot {100*pp:5.1f} [{100*lp:4.1f},{100*hp:4.1f}]  suites "
              f"{ {s: int((rows['suite'][sel]==s).sum()) for s in SUITES if (rows['suite'][sel]==s).any()} }")
    rho, p = spearmanr([x[1] for x in famtab], [x[3] for x in famtab])
    print(f"   family-level Spearman, median D vs agent success (n=12 families): rho={rho:+.3f} p={p:.3f}")
    R["fam_rho_D_agent"] = rho; R["fam_p_D_agent"] = p

    # ---- 4. clock-sensitivity (600 s suites, zero-action craft, not joinable)
    print("\n4. Clock-sensitivity from the zero-action measurement on the 600 s suites "
          "(different levels from the 150 s suites; not joined to agent outcomes)")
    print("   clock-sensitive = first contact with any target after 150 s and before the run ended (900 s)")
    tot = {"single": [0, 0, 0], "multi": [0, 0, 0]}  # sensitive_900, sensitive_600, n
    print(f"   {'suite':7s} {'single: sens/n (<=600 s)':>28s} {'multi: sens/n (<=600 s)':>28s}  Fisher p (900)")
    for s in SUITES:
        fc = noop[s]["first_contact_s"]
        tc = np.nanmin(np.where(np.isnan(fc), np.inf, fc), axis=1)
        nt = noop[s]["n_targets"]
        single, multi = nt == 1, nt > 1
        sens9 = (tc > CLOCK_S) & np.isfinite(tc)
        sens6 = (tc > CLOCK_S) & (tc <= 600.0)
        early = tc <= CLOCK_S
        a = [int(sens9[single].sum()), int(sens6[single].sum()), int(single.sum())]
        b = [int(sens9[multi].sum()), int(sens6[multi].sum()), int(multi.sum())]
        for key, v in (("single", a), ("multi", b)):
            for i in range(3):
                tot[key][i] += v[i]
        pf = fisher_exact([[a[0], a[2] - a[0]], [b[0], b[2] - b[0]]])[1] if a[2] and b[2] else np.nan
        R[f"clock_{s}_single"] = a; R[f"clock_{s}_multi"] = b; R[f"clock_{s}_fisher_p"] = pf
        R[f"clock_{s}_early_single"] = int(early[single].sum()); R[f"clock_{s}_early_multi"] = int(early[multi].sum())
        print(f"   {s:7s} {a[0]:3d}/{a[2]:3d} ({a[1]:3d})".ljust(36)
              + f"{b[0]:3d}/{b[2]:3d} ({b[1]:3d})".rjust(22) + f"   {pf:.3f}"
              + f"    contact<=150 s: single {early[single].sum()}/{single.sum()}, multi {early[multi].sum()}/{multi.sum()}")
    a, b = tot["single"], tot["multi"]
    pf = fisher_exact([[a[0], a[2] - a[0]], [b[0], b[2] - b[0]]])[1]
    R["clock_pooled_single"] = a; R["clock_pooled_multi"] = b; R["clock_pooled_fisher_p"] = pf
    print(f"   pooled  single {a[0]}/{a[2]} = {100*a[0]/a[2]:.1f}% (<=600 s: {a[1]});  "
          f"multi {b[0]}/{b[2]} = {100*b[0]/b[2]:.1f}% (<=600 s: {b[1]});  Fisher p={pf:.3f}")
    # the share of all clock-sensitive tasks that are multi-target, against the
    # multi-target share of all tasks
    print(f"   of {a[0]+b[0]} clock-sensitive tasks, {b[0]} ({100*b[0]/(a[0]+b[0]):.0f}%) are multi-target; "
          f"multi-target tasks are {b[2]}/{a[2]+b[2]} ({100*b[2]/(a[2]+b[2]):.0f}%) of all tasks")
    # by target count; then per target, since a tour with more targets has more
    # chances of a late contact even if each target's chance is the same
    print("   by target count (pooled), with the per-target rate and the first-target rate:")
    for ntv in (1, 2, 3):
        c = s9 = pairs = late_pairs = first_late = 0
        for s in SUITES:
            fc = noop[s]["first_contact_s"]; tc = np.nanmin(np.where(np.isnan(fc), np.inf, fc), axis=1)
            sel = noop[s]["n_targets"] == ntv
            c += int(sel.sum()); s9 += int(((tc > CLOCK_S) & np.isfinite(tc))[sel].sum())
            f = fc[sel]
            pairs += int(np.isfinite(f).sum() + np.isinf(f).sum())   # active slots (nan = inactive)
            late_pairs += int(((f > CLOCK_S) & np.isfinite(f)).sum())
            first_late += int(((f[:, 0] > CLOCK_S) & np.isfinite(f[:, 0])).sum())
        R[f"clock_nt{ntv}"] = [s9, c]; R[f"clock_nt{ntv}_pairs"] = [late_pairs, pairs]
        R[f"clock_nt{ntv}_first"] = [first_late, c]
        print(f"     {ntv} target(s): tasks {s9}/{c} = {100*s9/max(c,1):.1f}%;  "
              f"target slots {late_pairs}/{pairs} = {100*late_pairs/max(pairs,1):.1f}%;  "
              f"first target only {first_late}/{c} = {100*first_late/max(c,1):.1f}%")
    fs = sum(R[f"clock_nt{v}_first"][0] for v in (1,)); fm = sum(R[f"clock_nt{v}_first"][0] for v in (2, 3))
    ps = sum(R[f"clock_nt{v}_pairs"][0] for v in (1,)); pm = sum(R[f"clock_nt{v}_pairs"][0] for v in (2, 3))
    qs = sum(R[f"clock_nt{v}_pairs"][1] for v in (1,)); qm = sum(R[f"clock_nt{v}_pairs"][1] for v in (2, 3))
    pf_pairs = fisher_exact([[ps, qs - ps], [pm, qm - pm]])[1]
    pf_first = fisher_exact([[fs, a[2] - fs], [fm, b[2] - fm]])[1]
    R["clock_pairs_single"] = [ps, qs]; R["clock_pairs_multi"] = [pm, qm]; R["clock_pairs_fisher_p"] = pf_pairs
    R["clock_first_single"] = [fs, a[2]]; R["clock_first_multi"] = [fm, b[2]]; R["clock_first_fisher_p"] = pf_first
    print(f"   per target slot: single {ps}/{qs} = {100*ps/qs:.1f}%, multi {pm}/{qm} = {100*pm/qm:.1f}%, Fisher p={pf_pairs:.3f}")
    print(f"   first target only: single {fs}/{a[2]} = {100*fs/a[2]:.1f}%, multi {fm}/{b[2]} = {100*fm/b[2]:.1f}%, Fisher p={pf_first:.3f}")

    # stash the report numbers next to the data
    rep = {k: np.asarray(v) for k, v in R.items() if not isinstance(v, str)}
    rep_path = pathlib.Path(os.environ.get("CORR_REPORT", str(OUT / "corr_report.npz")))
    np.savez_compressed(rep_path, **rep)
    if draw:
        figure(rows, famtab, edges)
    return R


# ------------------------------------------------------------------- figure
def figure(rows, famtab, edges):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    BLUE, ORANGE, GREY = "#2a78d6", "#eb6834", "#6b7280"
    plt.rcParams.update({
        "figure.dpi": 150, "savefig.dpi": 150, "savefig.bbox": "tight",
        "font.family": "serif", "font.serif": ["DejaVu Serif"], "font.size": 8.5,
        "axes.labelsize": 8.5, "axes.titlesize": 9, "legend.fontsize": 8,
        "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "lines.solid_capstyle": "round", "pdf.fonttype": 42,
    })
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.6), gridspec_kw=dict(width_ratios=[1, 1, 1.3]))

    def binned(ax, x, title, xlabel):
        e = np.quantile(x, np.linspace(0, 1, 11))
        b = np.clip(np.searchsorted(e, x, side="right") - 1, 0, 9)
        for who, col, lab, dx in (("agent", BLUE, "learned policy", 0.985),
                                  ("pilot", ORANGE, "reference pilot", 1.015)):
            xs, ps, lo, hi = [], [], [], []
            for i in range(10):
                sel = b == i
                p, l, h = wilson(rows[who][sel].sum(), sel.sum())
                xs.append(np.median(x[sel]) * dx); ps.append(100 * p); lo.append(100 * l); hi.append(100 * h)
            xs, ps, lo, hi = map(np.array, (xs, ps, lo, hi))
            ax.errorbar(xs, ps, yerr=[ps - lo, hi - ps], color=col, lw=1.1, marker="o", ms=3.0,
                        capsize=0, elinewidth=0.8, label=lab, zorder=3)
        ax.set_xscale("log"); ax.set_ylim(0, 100); ax.grid(lw=0.4, alpha=0.5)
        ax.set_xlabel(xlabel); ax.set_title(title, loc="left")

    binned(axes[0], rows["D"], "(a) by demand", r"$\mathcal{D}$, deciles of 2,388 tasks")
    axes[0].set_ylabel("arrived (%)")
    binned(axes[1], rows["eta_max"], r"(b) by $\eta$ at the targets", r"max $\eta$ over a task's targets")
    axes[1].set_yticklabels([])
    # (a) has data in every corner; (b) has nothing below 30 % at low eta
    axes[1].legend(frameon=False, loc="lower left", fontsize=6.8, handlelength=1.3,
                   handletextpad=0.5, borderaxespad=0.3)

    # (c) families as rows, ordered by their median D (printed at the right),
    # success along x with its Wilson interval
    ax = axes[2]
    order = sorted(famtab, key=lambda r: r[1])
    for i, (f, medD, nn, pa, la, ha, pp, lp, hp, _) in enumerate(order):
        ax.plot([100 * la, 100 * ha], [i - 0.16] * 2, color=BLUE, lw=0.8, alpha=0.45, zorder=2)
        ax.plot([100 * lp, 100 * hp], [i + 0.16] * 2, color=ORANGE, lw=0.8, alpha=0.45, zorder=2)
        ax.scatter([100 * pa], [i - 0.16], s=16, color=BLUE, zorder=3, linewidths=0)
        ax.scatter([100 * pp], [i + 0.16], s=16, color=ORANGE, zorder=3, linewidths=0)
        ax.text(103, i, f"{medD:.4f}" if medD < 0.01 else f"{medD:.3f}", fontsize=6.2,
                va="center", ha="left", color="#444", clip_on=False)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([r[0].replace("_", " ") for r in order], fontsize=6.8)
    ax.set_ylim(-0.6, len(order) - 0.4)
    ax.set_xlim(0, 100); ax.set_xticks([0, 25, 50, 75, 100]); ax.grid(axis="x", lw=0.4, alpha=0.5)
    ax.set_xlabel(r"arrived (%); the family's median $\mathcal{D}$ at right")
    ax.set_title("(c) by family", loc="left")
    fig.tight_layout(w_pad=0.8)
    fig.savefig(OUT / "corr.pdf"); fig.savefig(OUT / "corr.png")
    plt.close(fig); print("corr.pdf")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--analyse", action="store_true", help="skip the measurement, reuse corr_data.npz")
    ap.add_argument("--no-figure", action="store_true")
    a = ap.parse_args()
    if not a.analyse:
        measure()
    analyse(draw=not a.no_figure)
