import marimo

__generated_with = "0.16.5"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # E1 — Localization

    **11765 · Navigation and Environment Modelling in Mobile Robotics · MUSI**
    Released Friday 23/10/2026 · **Deadline: Monday 23/11/2026, 23:59** · Individual

    | Part | Content | Weight |
    |---|---|---|
    | A | Experimental design and dead reckoning | 20 % |
    | B | EKF localization | 30 % |
    | C | Particle filter: robustness and global localization (builds on L3) | 35 % |
    | D | Synthesis and comparison (≤ 350 words) | 15 % |

    **Rules.** Ground truth may be used **only** for the initial pose and for evaluation (including diagnostic analyses); it must never be used inside a localization filter. All numbers in your answers must come from your own runs on your assigned cases. Every line of code must be yours to explain; any student must be able to explain their work orally.

    **🛠 TODO** code · **✅ CHECK** automatic feedback · **✍️ ANSWER** written answers. Submit **individually** `E1_StudentCode_Surname.zip` containing `E1_localization.py` and `E1.html`. Use the student-code assignment in the separate PDF handout. Before exporting, set all `RUN_* = True` and execute `uv run marimo export html notebooks/E1_localization.py -o E1.html` from the repository root. Verify that the HTML contains the experiment tables.

    **How each part is organised.** Every part starts with a box *Given / You do / You report*. The answer cells list the questions one by one: answer each numbered item, with numbers from your runs.
    """
    )
    return


@app.cell(hide_code=True)
def _():
    import os
    import time

    import numpy as np
    import pandas as pd
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    return go, make_subplots, np, os, pd, time


@app.cell(hide_code=True)
def _(np, os):
    # ------------------------------------------------------------------
    # PROVIDED — data loading and evaluation. You do not need to modify it.
    # ------------------------------------------------------------------
    def find_data_root():
        for _cand in ["data", "../data", "../../data"]:
            if os.path.isdir(os.path.join(_cand, "MRCLAM_Dataset1")):
                return _cand
        raise FileNotFoundError("MRCLAM data not found: run marimo from the repository root.")

    def load_run(dataset, robot, duration_s=300.0):
        """Load `duration_s` seconds of one MRCLAM robot.

        Returns a dict with
          odom       (K,3) [t, v, w]           velocity commands
          meas       (J,4) [t, barcode, r, phi] measurements of LANDMARKS only
          gt         (N,4) [t, x, y, theta]     ground truth (evaluation only!)
          landmarks  {barcode: (x, y)}          known map
          x0, t0     initial pose (from ground truth) and time
          events     list of odometry (t, 0, v, w) and measurement (t, 1, barcode, r, phi)
                     events sorted by time
          n_robot_meas  number of discarded measurements of other robots
        """
        _d = os.path.join(find_data_root(), dataset)
        _bc = np.loadtxt(f"{_d}/Barcodes.dat")
        _lm = np.loadtxt(f"{_d}/Landmark_Groundtruth.dat")
        _gt = np.loadtxt(f"{_d}/{robot}_Groundtruth.dat")
        _od = np.loadtxt(f"{_d}/{robot}_Odometry.dat")
        _me = np.loadtxt(f"{_d}/{robot}_Measurement.dat").reshape(-1, 4)
        _sub2bc = {int(s): b for s, b in _bc}
        _landmarks = {_sub2bc[int(r[0])]: (r[1], r[2]) for r in _lm}
        _landmark_ids = {_sub2bc[int(r[0])]: int(r[0]) for r in _lm}
        if dataset == "MRCLAM_Dataset1":
            # Fixed E1 course-reference map for Dataset1 (subject id -> x, y in metres).
            # It is applied regardless of your local Landmark_Groundtruth.dat, so that every
            # student analyses exactly the same case A. Do not change these values.
            for _subject, _xy in {11: (3.15071999, 2.38294871), 17: (0.03596156, -2.84396626)}.items():
                _landmarks[_sub2bc[_subject]] = _xy
        _robot_bc = [_sub2bc[i] for i in range(1, 6)]
        _t0 = _od[_od[:, 0] > _gt[0, 0]][0, 0]
        _t1 = _t0 + duration_s
        _od = _od[(_od[:, 0] >= _t0) & (_od[:, 0] <= _t1)]
        _me = _me[(_me[:, 0] > _t0) & (_me[:, 0] <= _t1)]
        _n_robot = int(np.isin(_me[:, 1], _robot_bc).sum())
        _me = _me[np.isin(_me[:, 1], list(_landmarks))]
        _gt = _gt[(_gt[:, 0] >= _gt[_gt[:, 0] <= _t0][-1, 0]) & (_gt[:, 0] <= _t1)]
        _th = np.unwrap(_gt[:, 3])
        _x0 = np.array(
            [
                np.interp(_t0, _gt[:, 0], _gt[:, 1]),
                np.interp(_t0, _gt[:, 0], _gt[:, 2]),
                (np.interp(_t0, _gt[:, 0], _th) + np.pi) % (2 * np.pi) - np.pi,
            ]
        )
        _ev = [(t, 0, v, w) for t, v, w in _od] + [(t, 1, b, r, ph) for t, b, r, ph in _me]
        _ev.sort(key=lambda e: (e[0], e[1]))
        return dict(
            odom=_od, meas=_me, gt=_gt, landmarks=_landmarks, x0=_x0, t0=_t0,
            events=_ev, n_robot_meas=_n_robot, landmark_ids=_landmark_ids, dataset=dataset, robot=robot,
        )

    def gt_at(gt, t):
        """Ground-truth pose interpolated at times t (theta unwrapped, then wrapped)."""
        _th = np.unwrap(gt[:, 3])
        return np.stack(
            [
                np.interp(t, gt[:, 0], gt[:, 1]),
                np.interp(t, gt[:, 0], gt[:, 2]),
                (np.interp(t, gt[:, 0], _th) + np.pi) % (2 * np.pi) - np.pi,
            ],
            axis=-1,
        )

    def evaluate(states, gt, sample_times=None):
        """ATE: RMS XY error at COMMON timestamps; FTE: last common sample.
        Without sample_times, uses the states' timestamps (only for standalone checks).
        State is taken at the most recent time <= each requested instant (causal hold).
        Ground truth is interpolated in time, with theta unwrapped before interpolation.
        """
        _s = np.asarray(states, float)
        _ts = _s[:, 0] if sample_times is None else np.asarray(sample_times, float)
        _ts = _ts[(_ts >= _s[0,0]) & (_ts <= _s[-1,0]) & (_ts >= gt[0,0]) & (_ts <= gt[-1,0])]
        if len(_ts) == 0:
            raise ValueError("No common time samples. Increase the duration or inspect the data.")
        _ix = np.searchsorted(_s[:,0], _ts, side="right") - 1
        _v = _s[_ix, 1:]
        _g = gt_at(gt, _ts)
        _e = np.hypot(_v[:,0]-_g[:,0], _v[:,1]-_g[:,1])
        _etheta = (_v[-1,2]-_g[-1,2]+np.pi) % (2*np.pi)-np.pi
        return {"ATE (m)":float(np.sqrt(np.mean(_e**2))),
                "FTE (m)":float(_e[-1]),
                "final heading error (deg)":float(abs(np.degrees(_etheta)))}

    def eval_run(states, run, after_s=0.0):
        """Compare algorithms on the SAME odometry-timestamp grid for each case."""
        _ts = run["odom"][:,0]
        if after_s:
            _ts = _ts[_ts-run["t0"] >= after_s]
        return evaluate(states, run["gt"], sample_times=_ts)

    def path_length(gt):
        return float(np.sum(np.hypot(np.diff(gt[:, 1]), np.diff(gt[:, 2]))))
    return eval_run, gt_at, load_run, path_length


@app.cell(hide_code=True)
def _(go, gt_at, make_subplots, np):
    # PROVIDED — plotting helpers
    _COLORS = ["#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#8c564b"]

    def plot_trajectories(run, estimates, title="Trajectories", ellipses=None):
        """estimates: dict name -> array [t, x, y, theta]; ellipses: list of (mean_xy, cov_2x2)."""
        _fig = go.Figure()
        _gt_vis = run["gt"][np.linspace(0, len(run["gt"])-1, min(350, len(run["gt"]))).astype(int)]
        _fig.add_trace(go.Scatter(x=_gt_vis[:, 1], y=_gt_vis[:, 2], mode="lines", name="ground truth", line=dict(color="#1f77b4", dash="dot")))
        _bcs = list(run["landmarks"].keys())
        _ids = run.get("landmark_ids", {})
        _lab = [f"L{_ids[_b]}" if _b in _ids else f"{int(_b)}" for _b in _bcs]
        _fig.add_trace(go.Scatter(
            x=[run["landmarks"][_b][0] for _b in _bcs], y=[run["landmarks"][_b][1] for _b in _bcs],
            mode="markers+text", name="landmarks (map)", text=_lab, textposition="top center",
            textfont=dict(size=11, color="#555555"),
            hovertext=[f"<b>{_l}</b><br>barcode {int(_b)}" for _l, _b in zip(_lab, _bcs)], hoverinfo="text",
            marker=dict(symbol="star", size=11, color="gold", line=dict(width=0.5, color="#8a6d00"))))
        for _i, (_name, _s) in enumerate(estimates.items()):
            if _s is not None:
                _s = _s[np.linspace(0, len(_s)-1, min(350, len(_s))).astype(int)]
                _fig.add_trace(go.Scatter(x=_s[:, 1], y=_s[:, 2], mode="lines", name=_name, line=dict(color=_COLORS[_i % 5])))
        if ellipses:
            _phi = np.linspace(0, 2 * np.pi, 60)
            _c = np.stack([np.cos(_phi), np.sin(_phi)])
            for _m, _S in ellipses:
                _lam, _vec = np.linalg.eigh(_S)
                _p = _vec @ (3 * np.sqrt(np.maximum(_lam, 0))[:, None] * _c)
                _fig.add_trace(go.Scatter(x=_m[0] + _p[0], y=_m[1] + _p[1], mode="lines", line=dict(color="gray", width=1), showlegend=False))
        _fig.update_layout(title=title, height=520, plot_bgcolor="white", xaxis_title="x (m)", yaxis_title="y (m)", yaxis=dict(scaleanchor="x", scaleratio=1))
        return _fig

    def plot_errors(run, estimates):
        _fig = make_subplots(rows=2, cols=1, shared_xaxes=True, subplot_titles=("Position error (m)", "Heading error (deg)"))
        for _i, (_name, _s) in enumerate(estimates.items()):
            if _s is None:
                continue
            _s = _s[np.linspace(0, len(_s)-1, min(350, len(_s))).astype(int)]
            _g = gt_at(run["gt"], _s[:, 0])
            _t = _s[:, 0] - run["t0"]
            _fig.add_trace(go.Scatter(x=_t, y=np.hypot(_s[:, 1] - _g[:, 0], _s[:, 2] - _g[:, 1]), mode="lines", name=_name, line=dict(color=_COLORS[_i % 5])), row=1, col=1)
            _fig.add_trace(go.Scatter(x=_t, y=np.degrees((_s[:, 3] - _g[:, 2] + np.pi) % (2 * np.pi) - np.pi), mode="lines", name=_name, showlegend=False, line=dict(color=_COLORS[_i % 5])), row=2, col=1)
        _fig.update_xaxes(title_text="time (s)", row=2, col=1)
        _fig.update_layout(height=480, plot_bgcolor="white")
        return _fig
    return plot_errors, plot_trajectories


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ---
    ## 0 · Your functions from L1, L2 and L3

    ### 🛠 TODO 0
    Paste your tested functions from L1, L2 and L3 in the three cells below (same names and signatures). The CHECK cell repeats the tests of the labs. Use only your assigned dataset/robot pair in case B; case A is shared.

    **Reproducibility:** Numerical settings and RUN flags are SOURCE-CODE variables, not sliders. You may explore interactively, but save your final settings in the `.py` file. The HTML export starts a fresh execution, so enable all four RUN flags before exporting.
    """
    )
    return


@app.cell
def _():
    # ---- from L1 ----
    def wrap_angle(a):
        return None

    def motion_euler(x, v, w, dt):
        return None
    return motion_euler, wrap_angle


@app.cell
def _():
    # ---- from L2 ----
    def G_jacobian(mu, v, w, dt):
        return None

    def V_jacobian(mu, v, w, dt):
        return None

    def control_noise(v, w, alphas):
        return None

    def ekf_predict(mu, Sigma, v, w, dt, alphas):
        return None, None

    def h_landmark(mu, landmark):
        return None

    def H_landmark(mu, landmark):
        return None

    def ekf_update(mu, Sigma, z, landmark, Q):
        return None, None, None, None, None

    def accept_measurement(nu, S, threshold):
        return None
    return accept_measurement, control_noise, ekf_predict, ekf_update


@app.cell
def _():
    # ---- from L3 ----
    def sample_motion_model(P, v, w, dt, alphas, rng):
        return None

    def measurement_log_likelihood(P, z, landmark, sigma):
        return None

    def normalize_log_weights(logw):
        return None

    def effective_sample_size(w):
        return None

    def low_variance_resample(P, w, rng):
        return None
    return (
        effective_sample_size,
        low_variance_resample,
        measurement_log_likelihood,
        normalize_log_weights,
        sample_motion_model,
    )


@app.cell(hide_code=True)
def _(
    accept_measurement,
    control_noise,
    effective_sample_size,
    ekf_predict,
    ekf_update,
    low_variance_resample,
    measurement_log_likelihood,
    mo,
    motion_euler,
    normalize_log_weights,
    np,
    sample_motion_model,
    wrap_angle,
):
    def _num_jac(f, x, eps=1e-6):
        _x = np.asarray(x, float)
        _J = np.zeros((np.asarray(f(_x)).size, _x.size))
        for _i in range(_x.size):
            _d = np.zeros_like(_x)
            _d[_i] = eps
            _J[:, _i] = (np.asarray(f(_x + _d)) - np.asarray(f(_x - _d))) / (2 * eps)
        return _J

    def _check():
        try:
            _mu = np.array([1.0, 2.0, 0.7])
            _m, _S = ekf_predict(_mu, np.eye(3) * 0.01, 0.3, 0.2, 0.05, (0.1, 0.01, 0.01, 0.1))
            _G = _num_jac(lambda x: motion_euler(x, 0.3, 0.2, 0.05), _mu)
            _V = _num_jac(lambda u: motion_euler(_mu, u[0], u[1], 0.05), [0.3, 0.2])
            _Sref = _G @ (np.eye(3) * 0.01) @ _G.T + _V @ control_noise(0.3, 0.2, (0.1, 0.01, 0.01, 0.1)) @ _V.T
            _mu2, _S2, _nu, _Sin, _K = ekf_update(np.array([0.0, 0.0, np.pi - 0.02]), np.diag([0.1, 0.1, 0.05]), (3.0, np.pi - 0.01), (3.0, 0.1), np.diag([0.01, 0.0025]))
            _r = [
                ("wrap_angle", np.allclose(wrap_angle(np.array([3 * np.pi / 2, -3 * np.pi / 2])), [-np.pi / 2, np.pi / 2])),
                ("motion_euler", np.allclose(motion_euler(np.zeros(3), 1.0, np.pi / 2, 1.0), [1, 0, np.pi / 2])),
                ("ekf_predict (mean and covariance)", np.allclose(_m, motion_euler(_mu, 0.3, 0.2, 0.05)) and np.allclose(_S, _Sref, atol=1e-8)),
                ("ekf_update (innovation wrapped, Σ decreases)", abs(_nu[1]) < 0.5 and np.trace(_S2) < 0.25),
                ("accept_measurement", bool(accept_measurement(np.array([0.2, 0.1]), np.diag([0.04, 0.01]), 9.21)) and not bool(accept_measurement(np.array([1.0, 0.0]), np.diag([0.04, 0.01]), 9.21))),
                ("sample_motion_model (mean over 20 000 samples)", np.allclose(sample_motion_model(np.tile(_mu, (20000, 1)), 0.5, 0.4, 0.5, (0.1, 0.05, 0.05, 0.1), np.random.default_rng(1)).mean(0), motion_euler(_mu, 0.5, 0.4, 0.5), atol=5e-3)),
                ("measurement_log_likelihood", np.isclose(measurement_log_likelihood(np.zeros((1, 3)), (3.0, 0.0), (3.0, 0.0), (0.1, 0.05))[0], -np.log(2 * np.pi * 0.005))),
                ("normalize_log_weights / effective_sample_size", np.isclose(normalize_log_weights(np.array([-1000.0, -1001.0]))[0], 1 / (1 + np.exp(-1))) and np.isclose(effective_sample_size(np.ones(4) / 4), 4)),
                ("low_variance_resample", [int(np.sum(low_variance_resample(np.arange(4.0)[:, None] * np.ones((1, 3)), np.array([0.5, 0.25, 0.25, 0.0]), np.random.default_rng(3))[:, 0] == _i)) for _i in range(4)] == [2, 1, 1, 0]),
            ]
        except Exception as _e:
            return f"⏳ not pasted yet ({type(_e).__name__})"
        return "<br>".join(f"{'✅' if _ok else '❌'} {_n}" for _n, _ok in _r)

    mo.md("**CHECK — L1/L2/L3 functions**<br>" + _check())
    return


@app.cell(hide_code=True)
def _(accept_measurement, ekf_predict, ekf_update, np):
    # PROVIDED — dead reckoning and EKF over the event stream (same as in L1/L2)
    def dead_reckoning(run, model, subsample=1):
        """Integrate at every odometry time; refresh velocity commands only every k samples.

        k=1 retains every command. k>1 simulates stale/decimated commands
        without changing the Euler integration time steps or evaluation grid.
        """
        if not isinstance(subsample, int) or subsample < 1:
            raise ValueError("subsample must be an integer >= 1")
        _x = run["x0"].copy()
        _t = run["t0"]
        _u = (0.0, 0.0)
        _out = [[_t, *_x]]
        _k = 0
        for _e in run["events"]:
            if _e[1] != 0:
                continue
            _dt = _e[0] - _t
            if _dt > 0:
                _x = model(_x, _u[0], _u[1], _dt)
                if _x is None:
                    return None
                _t = _e[0]
                _out.append([_t, *_x])
            if _k % subsample == 0:
                _u = (_e[2], _e[3])
            _k += 1
        return np.asarray(_out)

    def run_ekf(run, alphas, Q, gate_threshold=None):
        _mu = run["x0"].copy()
        _S = np.diag([1e-6] * 3)
        _t = run["t0"]
        _u = (0.0, 0.0)
        _out = [[_t, *_mu]]
        _info = {"accepted": 0, "rejected": 0, "first_update": None,
                 "accepted_by_landmark": {}, "rejected_by_landmark": {}}
        for _e in run["events"]:
            _dt = _e[0] - _t
            if _dt > 0:
                _mu, _S = ekf_predict(_mu, _S, _u[0], _u[1], _dt, alphas)
                if _mu is None:
                    return None, None
                _t = _e[0]
            if _e[1] == 0:
                _u = (_e[2], _e[3])
                _out.append([_t, *_mu])
                continue
            _res = ekf_update(_mu, _S, (_e[3], _e[4]), run["landmarks"][_e[2]], Q)
            if _res[0] is None:
                return None, None
            if gate_threshold is not None and not accept_measurement(_res[2], _res[3], gate_threshold):
                _info["rejected"] += 1
                _bc = int(_e[2])
                _info["rejected_by_landmark"][_bc] = _info["rejected_by_landmark"].get(_bc, 0) + 1
                continue
            if _info["first_update"] is None:
                _info["first_update"] = dict(t=_t - run["t0"], z=(_e[3], _e[4]), nu=_res[2], S=_res[3], K=_res[4])
            _mu, _S = _res[0], _res[1]
            _info["accepted"] += 1
            _bc = int(_e[2])
            _info["accepted_by_landmark"][_bc] = _info["accepted_by_landmark"].get(_bc, 0) + 1
            _out.append([_t, *_mu])
        return np.array(_out), _info
    return dead_reckoning, run_ekf


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ---
    ## Part A — Experimental design and dead reckoning (20 %)

    | | |
    |---|---|
    | **Given** | Two fixed cases, used in **all** parts: case A = Dataset1/Robot1 (shared by everybody) and case B = your assigned dataset/robot, both over the first 300 s. A table of characteristics of both cases, dead reckoning with your L1 `motion_euler`, and a command-decimation experiment. |
    | **You do** | Set `STUDENT_CODE` (next cell) to your code from the handout. Choose `a3_sub` (k ≥ 2) for A3. |
    | **You report** | A1–A3 below. |

    The cases are prescribed: do not change them, and do not use Dataset9. For Dataset1 the loader applies the fixed **E1 course-reference map** (written in the loader code), regardless of local changes to the original MRCLAM map files, so that everyone analyses the same case A. Do not modify the loader.
    """
    )
    return


@app.cell
def _():
    # Each student uses the code printed next to their name in the assignment handout.
    # Only change STUDENT_CODE; do not edit the two prescribed cases or the duration.
    STUDENT_CODE = "E1-0"  # TODO: replace with your own code (E1-01 ... E1-09)
    ASSIGNED_CASES = {
        "E1-01": ("MRCLAM_Dataset2", "Robot1"),
        "E1-02": ("MRCLAM_Dataset2", "Robot2"),
        "E1-03": ("MRCLAM_Dataset3", "Robot1"),
        "E1-04": ("MRCLAM_Dataset3", "Robot2"),
        "E1-05": ("MRCLAM_Dataset4", "Robot1"),
        "E1-06": ("MRCLAM_Dataset4", "Robot2"),
        "E1-07": ("MRCLAM_Dataset5", "Robot1"),
        "E1-08": ("MRCLAM_Dataset5", "Robot2"),
        "E1-09": ("MRCLAM_Dataset6", "Robot1"),
    }
    caseA_ds, caseA_rb = "MRCLAM_Dataset1", "Robot1"
    if STUDENT_CODE not in ASSIGNED_CASES:
        raise ValueError(f"Set STUDENT_CODE to your own code from the handout (E1-01 ... E1-09), not {STUDENT_CODE!r}.")
    caseB_ds, caseB_rb = ASSIGNED_CASES[STUDENT_CODE]
    e1_duration = 300.0  # common 300-second evaluation window
    return STUDENT_CODE, caseA_ds, caseA_rb, caseB_ds, caseB_rb, e1_duration


@app.cell(hide_code=True)
def _(
    STUDENT_CODE,
    caseA_ds,
    caseA_rb,
    caseB_ds,
    caseB_rb,
    e1_duration,
    load_run,
    mo,
    np,
    path_length,
    pd,
):
    cases = {
        "A": load_run(caseA_ds, caseA_rb, e1_duration),
        "B": load_run(caseB_ds, caseB_rb, e1_duration),
    }
    _rows = []
    for _k, _r in cases.items():
        _dur = _r["gt"][-1, 0] - _r["gt"][0, 0]
        _rows.append(
            {
                "case": f"{_k}: {_r['dataset'][-8:]}/{_r['robot']}",
                "duration (s)": round(_dur),
                "path (m)": round(path_length(_r["gt"]), 1),
                "landmark meas./min": round(60 * len(_r["meas"]) / _dur, 1),
                "robot meas. discarded": _r["n_robot_meas"],
                "mean |ω| cmd (rad/s)": round(float(np.mean(abs(_r["odom"][:, 2]))), 3),
                "distinct landmarks seen": len(set(_r["meas"][:, 1])),
            }
        )
    mo.vstack([mo.md(f"**A1 — {STUDENT_CODE}, assigned case characteristics**"), pd.DataFrame(_rows)])
    return (cases,)


@app.cell(hide_code=True)
def _(
    cases,
    dead_reckoning,
    eval_run,
    mo,
    motion_euler,
    pd,
    plot_trajectories,
):
    dr_res = {_k: dead_reckoning(_r, motion_euler) for _k, _r in cases.items()}
    if any(_v is None for _v in dr_res.values()):
        _out = mo.md("⏳ Paste your L1 functions first.")
    else:
        _out = mo.vstack(
            [
                mo.md("**A2 — dead reckoning**"),
                pd.DataFrame([{"case": _k, **eval_run(_s, cases[_k])} for _k, _s in dr_res.items()]).round(3),
                plot_trajectories(cases["A"], {"DR": dr_res["A"]}, title="A — DR, shared Case A"),
            ]
        )
    _out
    return (dr_res,)


@app.cell(hide_code=True)
def _(cases, dr_res, mo, plot_trajectories):
    # Case B in its own cell (one plot per output keeps the marimo output small).
    _out = mo.md("") if dr_res.get("B") is None else plot_trajectories(cases["B"], {"DR": dr_res["B"]}, title=f"B — DR, assigned case ({cases['B']['dataset'][-8:]}/{cases['B']['robot']})")
    _out
    return


@app.cell(hide_code=True)
def _(cases, dr_res, mo, plot_errors):
    # A2: positional and heading error over time for each case.
    _case = "A"
    _out = mo.md("Complete your L1 functions first.") if dr_res.get(_case) is None else plot_errors(cases[_case], {"DR": dr_res[_case]})
    _out
    return


@app.cell(hide_code=True)
def _(cases, dr_res, mo, plot_errors):
    _case = "B"
    _out = mo.md("Complete your L1 functions first.") if dr_res.get(_case) is None else plot_errors(cases[_case], {"DR": dr_res[_case]})
    _out
    return


@app.cell
def _():
    a3_sub = 5  # choose k>=2 for your deliberate odometry-command decimation
    return (a3_sub,)


@app.cell(hide_code=True)
def _(a3_sub, cases, dead_reckoning, eval_run, mo, motion_euler, pd):
    _rows = []
    for _k, _r in cases.items():
        for _kk in (1, a3_sub):  # controlled comparison: native rate vs decimated commands
            _s = dead_reckoning(_r, motion_euler, _kk)
            if _s is None:
                break
            _rows.append({"case": _k, "k": _kk, **eval_run(_s, _r)})
    pd.DataFrame(_rows).round(3) if _rows else mo.md("⏳ Paste your L1 functions first.")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    #### ✍️ ANSWERS — Part A

    ### A1 — Characterising the two cases (5 %)
    Using the characteristics table:
    1. Report at least **two quantitative differences** between case A and case B (e.g. path length, turning rate, landmark measurement rate, number of landmarks seen).
    2. **Predict**, before looking at the results, which case should be harder for dead reckoning and which for landmark-based localization, and why.

    Your prediction is assessed on its reasoning, not on whether it turns out to be right; you will revisit it in Part D.

    > *Your answer…*

    ### A2 — Dead-reckoning drift (5 %)
    1. Report ATE, FTE and final heading error of dead reckoning for both cases.
    2. Does the position error grow monotonically over time? Use the two **error-versus-time** plots, not just ATE/FTE, to support your answer.
    3. Does the result agree with your prediction in A1?

    > *Your answer…*

    ### A3 — Where does the drift come from? A controlled experiment (10 %)
    The table above runs dead reckoning with all commands (k = 1) and keeping **one command out of k** (`a3_sub`). Between two retained commands the last retained velocity is held: this is deliberate **command decimation** (information is discarded), not merely a larger integration step.
    1. Report the table for your k and explain which information is lost when k > 1.
    2. At the native rate (k = 1), what sources of drift are possible? Explain why comparing k = 1 with k > 1 **cannot by itself** separate Euler integration error from command/motion mismatch. Propose one additional controlled comparison that could separate them (e.g. Euler vs exact unicycle integration with the same commands).
    3. Do both cases react to decimation in the same way? If not, propose an explanation.

    > *Your answer…*
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ---
    ## Part B — EKF localization (30 %)

    | | |
    |---|---|
    | **Given** | The EKF of L2 (your functions) run on the event stream, with and without a validation gate; a table DR / EKF / gated EKF on the same evaluation timestamps; the details of the first accepted update; the number of accepted/rejected measurements per landmark; tools for diagnosing rejected measurements (B3.2); a framework that runs your sensitivity configurations (B4). |
    | **You do** | Choose the EKF parameters in the next cell (they are also the motion noise of the particle filter in Part C). Write a short diagnostic for B3.2. Define the B4 configurations and set `RUN_B4 = True`. |
    | **You report** | B2, B3 (B3.1 and B3.2) and B4 in the answer cell at the end of Part B. |
    """
    )
    return


@app.cell
def _(np):
    # Final reproducible configuration. Change these numbers in the SOURCE,
    # not only in temporary interactive controls; the HTML export executes this code.
    e1_alphas = (0.20, 0.02, 0.02, 0.50)
    b_sigma_r = 0.30   # metres
    b_sigma_phi = 0.30 # radians
    e1_Q = np.diag([b_sigma_r**2, b_sigma_phi**2])
    e1_gate = 9.21     # chi-square threshold, 2 DoF, ~99%
    return e1_Q, e1_alphas, e1_gate


@app.cell(hide_code=True)
def _(cases, dr_res, e1_Q, e1_alphas, e1_gate, eval_run, mo, pd, run_ekf):
    ekf_res, ekf_plain, ekf_details = {}, {}, {}
    _rows = []
    first_ekf_update = None
    first_ekf_case = None
    for _key, _run in cases.items():
        _p, _ = run_ekf(_run, e1_alphas, e1_Q)
        _g, _info = run_ekf(_run, e1_alphas, e1_Q, gate_threshold=e1_gate)
        if _p is None or _g is None or dr_res[_key] is None:
            continue
        ekf_plain[_key], ekf_res[_key], ekf_details[_key] = _p, _g, _info
        if first_ekf_update is None and _info["first_update"] is not None:
            first_ekf_update, first_ekf_case = _info["first_update"], _key
        for _method, _s in [("DR", dr_res[_key]), ("EKF", _p), ("EKF gated", _g)]:
            _row = {"case": _key, "method": _method, **eval_run(_s, _run)}
            if _method == "EKF gated":
                _row.update(accepted=_info["accepted"], rejected=_info["rejected"])
            _rows.append(_row)
    # Only small numerical tables are rendered here; plots are in SEPARATE cells.
    if not _rows:
        _display = mo.md("Paste and check your L1/L2 implementations first.")
    else:
        _display = mo.vstack([mo.md("**B3 — DR vs EKF (identical evaluation timestamps)**"), pd.DataFrame(_rows).round(3)])
    _display
    return ekf_details, ekf_plain, ekf_res, first_ekf_case, first_ekf_update


@app.cell(hide_code=True)
def _(first_ekf_case, first_ekf_update, mo, np):
    if first_ekf_update is None:
        _out = mo.md("**B2:** No gated EKF measurement was accepted; inspect the gate and model settings. This is an experimental finding, not a notebook error.")
    else:
        _out = mo.md(
            f"**B2 — first accepted update (case {first_ekf_case}, t={first_ekf_update['t']:.2f} s)**\n\n"
            f"- measurement z = `{np.round(first_ekf_update['z'],3).tolist()}`\n"
            f"- innovation ν = `{np.round(first_ekf_update['nu'],3).tolist()}`\n"
            f"- S = `{np.round(first_ekf_update['S'],4).tolist()}`\n"
            f"- K = `{np.round(first_ekf_update['K'],4).tolist()}`"
        )
    _out
    return


@app.cell(hide_code=True)
def _(cases, ekf_details, mo, pd):
    _rows = []
    for _key, _run in cases.items():
        _info = ekf_details.get(_key)
        if _info is None:
            continue
        _observed_ids = sorted(set(_info["accepted_by_landmark"]) | set(_info["rejected_by_landmark"]))
        for _barcode in _observed_ids:
            _nacc = _info["accepted_by_landmark"].get(_barcode, 0)
            _nrej = _info["rejected_by_landmark"].get(_barcode, 0)
            _rows.append({"case": _key, "landmark": f"L{_run['landmark_ids'].get(_barcode, '?')}",
                          "barcode": int(_barcode), "accepted": _nacc, "rejected": _nrej,
                          "rejection rate (%)": round(100*_nrej/max(1,_nacc+_nrej),1)})
    if _rows:
        _out = mo.vstack([mo.md("**B3 — Gating by landmark.** Which landmarks account for the rejections? Their cause is investigated in B3.2."),
                   pd.DataFrame(_rows).sort_values(["case", "rejected"], ascending=[True, False]).groupby("case", group_keys=False).head(5)])
    else:
        _out = mo.md("Gating diagnostics will appear after the EKF runs.")
    _out
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ---
    ### Tools for B3.2 — diagnosing the rejected measurements

    The gate rejects measurements whose innovation is improbable **for the filter**. That tells you *that* prediction and measurement are inconsistent, not *why*. Two kinds of explanation are possible, and they call for very different fixes:

    - **sensor-related errors:** some range/bearing measurements are wrong (false detections, occlusions, wrong barcode reading, noise heavier than Gaussian);
    - **systematic map or data-association errors:** the measurements are fine, but the position the filter **expects** for some landmark is wrong (wrong coordinates in the map, or measurements attributed to the wrong landmark).

    Both produce large innovations, so the gate alone cannot distinguish them: you need evidence that does **not** depend on the filter. For this diagnostic you may use the ground-truth pose (analysis only, never inside a filter). The cell below gives, for every landmark measurement, the residual with respect to the **ground-truth pose and the map**, $e = z - h(x_\text{GT}, m)$, with the bearing wrapped.
    """
    )
    return


@app.cell(hide_code=True)
def _(cases, gt_at, np, pd):
    # PROVIDED — residuals from the ground-truth pose (ANALYSIS ONLY, never inside a filter)
    def gt_residuals(run):
        """DataFrame with one row per landmark measurement: t (s since start), landmark, barcode,
        r, phi (measured), x_gt, y_gt, theta_gt (ground-truth pose at that instant), e_r, e_phi."""
        _m = run["meas"]
        _X = gt_at(run["gt"], _m[:, 0])
        _L = np.array([run["landmarks"][_b] for _b in _m[:, 1]])
        _dx, _dy = _L[:, 0] - _X[:, 0], _L[:, 1] - _X[:, 1]
        _er = _m[:, 2] - np.hypot(_dx, _dy)
        _ep = (_m[:, 3] - (np.arctan2(_dy, _dx) - _X[:, 2]) + np.pi) % (2 * np.pi) - np.pi
        _ep = (_ep + np.pi) % (2 * np.pi) - np.pi
        return pd.DataFrame({
            "t": _m[:, 0] - run["t0"], "landmark": [f"L{run['landmark_ids'].get(_b, '?')}" for _b in _m[:, 1]],
            "barcode": _m[:, 1].astype(int), "r": _m[:, 2], "phi": _m[:, 3],
            "x_gt": _X[:, 0], "y_gt": _X[:, 1], "theta_gt": _X[:, 2], "e_r": _er, "e_phi": _ep,
        })

    diag_resid = {_k: gt_residuals(_r) for _k, _r in cases.items()}
    return (diag_resid,)


@app.cell(hide_code=True)
def _(diag_resid, mo, np, pd):
    _rows = []
    for _k, _df in diag_resid.items():
        for _lab, _g in _df.groupby("landmark"):
            _rows.append({"case": _k, "landmark": _lab, "measurements": len(_g),
                          "mean e_r (m)": _g.e_r.mean(), "std e_r (m)": _g.e_r.std(),
                          "mean e_φ (rad)": _g.e_phi.mean(), "std e_φ (rad)": _g.e_phi.std(),
                          "% |e_r| > 1 m": 100 * np.mean(np.abs(_g.e_r) > 1.0)})
    _tab = pd.DataFrame(_rows).sort_values(["case", "% |e_r| > 1 m"], ascending=[True, False]).round(3)
    mo.vstack([mo.md("**B3.2 — residuals w.r.t. the ground-truth pose, per landmark** (sorted by the fraction of large residuals)"), _tab])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### 🛠 TODO B3.2a — your diagnostic
    Write code (a plot and/or a table) that helps you decide between the two kinds of explanation for the landmarks that concentrate the rejections. You choose the method; `diag_resid["A"]` contains everything you need. A good diagnostic answers questions such as:

    - Are the large residuals **random** (scattered, appearing at some instants) or **systematic** (similar every time that landmark is seen, at any time of the run)?
    - Do they affect **all** landmarks a little, or a **few** landmarks completely?
    - From the ground-truth pose and the measured $(r,\phi)$, **where would the observed landmark have to be** for the measurement to be exact? Is that position stable over time? Does it coincide with anything in the map?
    """
    )
    return


@app.cell
def _(mo):
    # TODO B3.2a — your diagnostic. Store what you want to show in diag_out (a figure, a DataFrame or mo.vstack([...])).
    diag_out = mo.md("⏳ TODO B3.2a: write your diagnostic here.")
    diag_out
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### 🛠 TODO B3.2b — test your hypothesis
    If your diagnostic points to a map/association error, formulate the **smallest** change to the map that would explain the data and test it: write it in `map_overrides` as `{barcode: (x, y)}` (barcodes are in the tables above). The provided cell re-runs the EKF of case A with the modified map **only for this test**; the rest of E1 keeps the map as distributed. If your evidence points to a sensor-related cause instead, leave the dictionary empty and explain why in B3.2.
    """
    )
    return


@app.cell
def _():
    map_overrides = {}  # TODO B3.2b, e.g. {12: (1.0, 2.0)}
    return (map_overrides,)


@app.cell(hide_code=True)
def _(
    cases,
    e1_Q,
    e1_alphas,
    e1_gate,
    eval_run,
    map_overrides,
    mo,
    pd,
    run_ekf,
):
    # PROVIDED — gated EKF on case A with the original map and with your modified map
    _rows = []
    _orig = cases["A"]
    _test = dict(_orig)
    _test["landmarks"] = dict(_orig["landmarks"])
    for _b, _xy in map_overrides.items():
        if float(_b) in _test["landmarks"]:
            _test["landmarks"][float(_b)] = (float(_xy[0]), float(_xy[1]))
    for _name, _r in [("map as distributed", _orig), ("your modified map", _test)]:
        _p, _ = run_ekf(_r, e1_alphas, e1_Q)
        _g, _info = run_ekf(_r, e1_alphas, e1_Q, gate_threshold=e1_gate)
        if _p is None:
            break
        _worst = sorted(_info["rejected_by_landmark"].items(), key=lambda kv: -kv[1])[:3]
        _rows.append({"map": _name, "EKF ATE (m)": eval_run(_p, _r)["ATE (m)"], "EKF gated ATE (m)": eval_run(_g, _r)["ATE (m)"],
                      "rejected": _info["rejected"],
                      "most rejected": ", ".join(f"L{_r['landmark_ids'][float(_bb)]}: {_n}" for _bb, _n in _worst) or "—"})
    _note = "" if map_overrides else " (no override defined: both rows are identical)"
    (mo.vstack([mo.md("**B3.2b — hypothesis test on case A**" + _note), pd.DataFrame(_rows).round(3)]) if _rows
     else mo.md("Complete Part B first."))
    return


@app.cell(hide_code=True)
def _(cases, dr_res, ekf_plain, ekf_res, mo, plot_trajectories):
    # One plot per output cell to avoid marimo's "output is too large" error.
    _case = "A"  # Change to "B" to inspect the assigned case.
    if _case in ekf_res:
        _out = plot_trajectories(cases[_case], {"DR": dr_res[_case], "EKF": ekf_plain[_case],
                         "EKF gated": ekf_res[_case]}, title=f"B — Case {_case} trajectories")
    else:
        _out = mo.md("Complete Part B to see the trajectory plot.")
    _out
    return


@app.cell(hide_code=True)
def _(cases, dr_res, ekf_plain, ekf_res, mo, plot_errors):
    _case = "A"  # Change to "B" for the other case.
    if _case in ekf_res:
        _out = plot_errors(cases[_case], {"DR": dr_res[_case], "EKF": ekf_plain[_case], "EKF gated": ekf_res[_case]})
    else:
        _out = mo.md("Complete Part B to see the error plot.")
    _out
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### 🛠 TODO B4 — sensitivity of EKF **accuracy and measurement acceptance**
    **Objective:** Test how ATE, FTE and the number of accepted/rejected measurements respond to the assumed motion uncertainty, sensor uncertainty and gating decision. This is not merely a search for the lowest ATE.

    Define at least **four** configurations in `ekf_experiments`: a **baseline** and variations that change **one factor at a time** with respect to it (process noise α, measurement noise σ, gate threshold or no gate). Each entry is a dict with keys `name`, `alphas` (4-tuple), `sigma_r`, `sigma_phi`, `gate` (χ² threshold or `None`). You may add configurations to investigate unexpected results. Set `RUN_B4 = True` when the list is final: the experiment then runs on both cases and during the HTML export.
    """
    )
    return


@app.cell
def _():
    ekf_experiments = [
        # {"name": "baseline", "alphas": (0.2, 0.02, 0.02, 0.5), "sigma_r": 0.3, "sigma_phi": 0.3, "gate": 9.21},
    ]
    return (ekf_experiments,)


@app.cell
def _():
    # Set True for the final run and for the HTML export.
    RUN_B4 = False
    return (RUN_B4,)


@app.cell(hide_code=True)
def _(RUN_B4, cases, ekf_experiments, eval_run, mo, np, pd, run_ekf):
    b4_table = pd.DataFrame()
    if not RUN_B4:
        _out = mo.md("B4 paused. Define your experiment configurations, then set RUN_B4 = True for final export.")
    elif len(ekf_experiments) < 4:
        _out = mo.md("B4: define at least four named one-factor-at-a-time configurations.")
    else:
        _rows = []
        for _cfg in ekf_experiments:
            for _key, _run in cases.items():
                _s, _info = run_ekf(_run, _cfg["alphas"],
                    np.diag([_cfg["sigma_r"]**2, _cfg["sigma_phi"]**2]),
                    gate_threshold=_cfg["gate"])
                if _s is not None:
                    _rows.append({"config": _cfg["name"], "case": _key, **eval_run(_s, _run),
                                  "accepted": _info["accepted"], "rejected": _info["rejected"]})
        b4_table = pd.DataFrame(_rows)
        _out = b4_table.round(3)
    _out
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    #### ✍️ ANSWERS — Part B

    ### B2 — Understanding an EKF update 
    Using the first accepted measurement update reported by the notebook:
    1. Explain the meaning of the measurement $z$, the innovation $\nu$, the innovation covariance $S$ and the Kalman gain $K$.
    2. Compare each innovation component with its expected standard deviation, obtained from the diagonal of $S$. Is this measurement consistent with the filter prediction?
    3. Compute the actual pose correction $\Delta\mu = K\nu$. Interpret the corrections in $x$, $y$ and heading using appropriate units.
    4. Explain why these corrections are small or large, considering the predicted pose uncertainty and the measurement noise.

    Support your explanation with numerical values from your run.

    > *Your answer…*

    ### B3 — Gating and measurement inconsistencies 

    **B3.1 — Performance comparison.** Compare dead reckoning, EKF without gating and EKF with gating on both cases. Report ATE, FTE and the numbers of accepted and rejected observations. Does gating improve localization in both cases? Explain the differences.

    > *Your answer…*

    **B3.2 — Investigating rejected landmarks.** Using the shared Dataset1/Robot1 case:
    1. Identify the landmarks responsible for most of the rejected measurements.
    2. Investigate why these particular observations appear inconsistent with the EKF prediction, while observations of other landmarks are accepted.
    3. Formulate at least two possible explanations, considering sensor errors, incorrect landmark positions or data-association problems.
    4. Propose and carry out a small diagnostic analysis to support or reject one of your hypotheses (TODO B3.2a, and B3.2b if relevant). You are free to add a plot, a quantitative comparison or a short piece of Python code. Include at least one piece of supporting evidence and explain what it shows.

    **Can you distinguish between a sensor-related error and a systematic inconsistency in the landmark map? What additional evidence would help?** Ground truth may be used for this diagnostic analysis but must not be incorporated into the localization filters.

    > *Your answer…*

    ### B4 — EKF parameter sensitivity 
    Design and execute at least four EKF configurations (TODO B4): a baseline and controlled variations of the process noise, the measurement noise and the validation gate, changing **one factor at a time** with respect to a reference configuration. You may include additional experiments to investigate unexpected results. For each configuration:
    1. Report the localization error (ATE and FTE) and the number of accepted and rejected observations.
    2. Explain how the parameter change influences the predicted covariance $\bar\Sigma$, the innovation covariance $S$, the Kalman gain $K$ and the Mahalanobis distance $d^2$.
    3. Compare the effects in cases A and B.
    4. Investigate whether a poorly tuned uncertainty model can cause valid observations to be rejected and degrade localization.

    If you observe a case where gating makes localization worse, explain the mechanism. Otherwise, explain why your tested configurations did not produce this behaviour. Support your conclusions with controlled experiments and numerical results, rather than only reporting which configuration achieves the smallest error.

    > *Your answer…*
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ---
    ## Part C — Particle filter: robustness and global localization (35 %)

    | | |
    |---|---|
    | **Given** | MCL (`run_pf`) assembled from your L3 functions; a tracking run with the parameters of the next cells; a framework that runs your particle-number study with three fixed seeds (C2); a global-localization experiment comparing an EKF started with four different headings and the PF started from uniformly spread particles (C3). |
    | **You do** | Implement `robust_log_likelihood` (C1) and `uniform_particles` (C3); choose the particle counts and ε values for C2; set `RUN_PF_TRACKING`, `RUN_C2` and `RUN_GLOBAL` to True. |
    | **You report** | C1–C3 in the answer cell at the end of Part C. The comparison of all methods is part of Part D. |

    In L3 you built MCL with a Gaussian likelihood, which, like the EKF without gating, fuses every measurement as if it were valid. Here you make it robust to inconsistent measurements, study it experimentally, and use it where the EKF is weak: **global localization**.

    ### 🛠 TODO C1 — robust likelihood
    Implement `robust_log_likelihood(P, z, landmark, sigma, eps)`: a mixture of your Gaussian likelihood from L3 and a uniform outlier component,

    $$
    p(z\mid x)=(1-\varepsilon)\,p_\text{Gauss}(z\mid x)+\varepsilon\,\frac{1}{10\,\text{m}}\cdot\frac{1}{2\pi}.
    $$

    The outlier density is uniform for valid measurements ($0\le r\le10$ m, $-\pi\le\phi<\pi$); outside this support it is zero. Work in log space, with no `log(0)` or underflow (hint: `np.logaddexp`). With $\varepsilon=0$, recover the L3 Gaussian likelihood.
    """
    )
    return


@app.function
def robust_log_likelihood(P, z, landmark, sigma, eps):
    return None


@app.cell(hide_code=True)
def _(measurement_log_likelihood, mo, np):
    def _check():
        try:
            _P = np.array([[0.0, 0.0, 0.0], [2.0, 1.0, 0.5], [0.0, 0.0, np.pi]])
            _l = (3.0, 0.0)
            _ll = robust_log_likelihood(_P, (3.0, 0.0), _l, (0.1, 0.05), 0.05)
            _llo = robust_log_likelihood(_P[:1], (9.0, 2.0), _l, (0.1, 0.05), 0.05)
            _lfar = robust_log_likelihood(np.array([[50.0, 50.0, 0.0]]), (3.0, 0.0), _l, (0.01, 0.01), 0.05)
            _r = [
                ("shape (N,) and correct particle most likely", np.shape(_ll) == (3,) and np.argmax(_ll) == 0),
                ("ε = 0 reduces to the Gaussian likelihood of L3", np.allclose(robust_log_likelihood(_P, (3.0, 0.0), _l, (0.1, 0.05), 0.0), measurement_log_likelihood(_P, (3.0, 0.0), _l, (0.1, 0.05)))),
                ("outlier → finite floor log(ε/(20π))", np.isclose(_llo[0], np.log(0.05 / (20 * np.pi)), atol=1e-6)),
                ("no −∞ / NaN even when the Gaussian underflows", np.all(np.isfinite(_lfar))),
            ]
        except Exception as _e:
            return f"⏳ not implemented yet ({type(_e).__name__})"
        return "<br>".join(f"{'✅' if _ok else '❌'} {_n}" for _n, _ok in _r)

    mo.md("**CHECK — C1**<br>" + _check())
    return


@app.cell(hide_code=True)
def _(
    effective_sample_size,
    low_variance_resample,
    normalize_log_weights,
    np,
    sample_motion_model,
):
    # PROVIDED — MCL with your L3 functions and your robust likelihood.
    # P0: optional (N, 3) initial particle set (default: all at the known initial pose).
    def run_pf(run, N, alphas, sigma, eps, seed=0, P0=None):
        if normalize_log_weights(np.zeros(2)) is None or effective_sample_size(np.ones(2) / 2) is None:
            return None, None
        _rng = np.random.default_rng(seed)
        _P = np.tile(run["x0"], (N, 1)) if P0 is None else P0.copy()
        _logw = np.zeros(len(_P))
        _t, _u = run["t0"], (0.0, 0.0)
        _nres = 0

        def _mean(w):
            return np.array([w @ _P[:, 0], w @ _P[:, 1], np.arctan2(w @ np.sin(_P[:, 2]), w @ np.cos(_P[:, 2]))])

        # Crucial: for global localization do not insert ground truth x0 into the estimate.
        _out = [[_t, *_mean(np.ones(len(_P)) / len(_P))]]

        for _e in run["events"]:
            if _e[0] > _t:
                _P = sample_motion_model(_P, _u[0], _u[1], _e[0] - _t, alphas, _rng)
                if _P is None:
                    return None, None
                _t = _e[0]
            if _e[1] == 0:
                _u = (_e[2], _e[3])
                _out.append([_t, *_mean(normalize_log_weights(_logw))])
                continue
            _ll = robust_log_likelihood(_P, (_e[3], _e[4]), run["landmarks"][_e[2]], sigma, eps)
            if _ll is None:
                return None, None
            _logw = _logw + _ll
            if not np.isfinite(np.max(_logw)):
                _logw = np.zeros(len(_P))  # all particles impossible: filter lost
            _w = normalize_log_weights(_logw)
            _out.append([_t, *_mean(_w)])
            if effective_sample_size(_w) < len(_P) / 2:
                _P = low_variance_resample(_P, _w, _rng)
                if _P is None:
                    return None, None
                _logw = np.zeros(len(_P))
                _nres += 1
        return np.array(_out), {"resamplings": _nres}
    return (run_pf,)


@app.cell
def _():
    # Tracking parameters; saved in source for a reproducible HTML export.
    c_N = 500
    c_sr = 0.30
    c_sp = 0.30
    c_eps = 0.05
    RUN_PF_TRACKING = False
    return RUN_PF_TRACKING, c_N, c_eps, c_sp, c_sr


@app.cell(hide_code=True)
def _(
    RUN_PF_TRACKING,
    c_N,
    c_eps,
    c_sp,
    c_sr,
    cases,
    e1_alphas,
    eval_run,
    mo,
    pd,
    run_pf,
    time,
):
    pf_res = {}
    _rows = []
    if not RUN_PF_TRACKING:
        _out = mo.md("Tracking comparison paused. Set RUN_PF_TRACKING = True before exporting.")
    else:
        for _key, _run in cases.items():
            _tic = time.perf_counter()
            _state, _info = run_pf(_run, c_N, e1_alphas, (c_sr, c_sp), c_eps, seed=0)
            if _state is None:
                continue
            pf_res[_key] = _state
            _rows.append({"case": _key, "method": "PF tracking", **eval_run(_state, _run),
                         "runtime (s)": round(time.perf_counter()-_tic,2),
                         "resamplings": _info["resamplings"]})
        if _rows:
            _out = pd.DataFrame(_rows).round(3)
        else:
            _out = mo.md("Complete L3 and C1 before running the PF.")
    _out
    return (pf_res,)


@app.cell(hide_code=True)
def _(cases, dr_res, ekf_res, mo, pf_res, plot_trajectories):
    _case = "A"  # Change to "B" to inspect the assigned case.
    if _case in pf_res and _case in ekf_res:
        _out = plot_trajectories(cases[_case], {"DR": dr_res[_case], "EKF gated": ekf_res[_case],
                          "PF tracking": pf_res[_case]}, title=f"C — Case {_case}: DR / EKF / PF")
    else:
        _out = mo.md("Tracking plot appears when Part B and C tracking have run.")
    _out
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### 🛠 TODO C2 — particle-number study
    Define `pf_particle_counts` (at least four values spanning at least two orders of magnitude, e.g. 20 → 2000) and `pf_eps_values` (must include 0 and one value > 0). Setting `RUN_C2 = True` runs every combination on both cases with three fixed seeds and reports the mean and standard deviation of the ATE and the run time. It can take a few minutes.
    """
    )
    return


@app.cell
def _():
    pf_particle_counts = []
    pf_eps_values = []
    return pf_eps_values, pf_particle_counts


@app.cell
def _():
    # Enable after defining valid C2 configurations; True is required in final HTML.
    RUN_C2 = False
    return (RUN_C2,)


@app.cell(hide_code=True)
def _(
    RUN_C2,
    c_sp,
    c_sr,
    cases,
    e1_alphas,
    eval_run,
    mo,
    np,
    pd,
    pf_eps_values,
    pf_particle_counts,
    run_pf,
    time,
):
    c2_table = pd.DataFrame()
    if not RUN_C2:
        _out = mo.md("C2 paused. Choose your N and ε values; set RUN_C2 = True for the final export.")
    elif len(pf_particle_counts) < 4 or 0.0 not in pf_eps_values or max(pf_particle_counts) < 100*min(pf_particle_counts):
        _out = mo.md("C2: define at least four N values spanning >=100x and include ε=0.")
    else:
        _rows = []
        for _N in pf_particle_counts:
            for _eps in pf_eps_values:
                for _key, _run in cases.items():
                    _ates, _times = [], []
                    for _seed in (0,1,2):
                        _tic = time.perf_counter()
                        _state, _info = run_pf(_run, _N, e1_alphas, (c_sr,c_sp), _eps, seed=_seed)
                        if _state is None:
                            break
                        _times.append(time.perf_counter()-_tic)
                        _ates.append(eval_run(_state, _run)["ATE (m)"])
                    if len(_ates) == 3:
                        _rows.append({"N":_N, "ε":_eps, "case":_key,
                         "ATE mean (m)":float(np.mean(_ates)), "ATE std (m)":float(np.std(_ates)),
                         "runtime mean (s)":float(np.mean(_times))})
        c2_table = pd.DataFrame(_rows)
        _out = c2_table.round(3) if not c2_table.empty else mo.md("PF implementation is incomplete.")
    _out
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### 🛠 TODO C3 — global localization
    Now the initial pose is **unknown**. Implement `uniform_particles(N, landmarks, rng)`: $N$ particles with $x,y$ uniform in the bounding box of the landmarks enlarged by 0.5 m, and $\theta$ uniform in $[-\pi,\pi)$.

    The comparison is with an EKF that also ignores the initial pose: its mean starts at the centre of the landmarks with one of four headings (0°, 90°, 180°, −90°) and a very large covariance, $\Sigma_0=\operatorname{diag}(3^2, 6^2, \pi^2)$. Errors are evaluated **from 60 s onward**, on the same timestamp grid as the tracking experiments, and a run counts as a success if that ATE is below `success_ate_threshold_m`. The particle mean is not a meaningful single pose while the posterior is multimodal.
    """
    )
    return


@app.function
def uniform_particles(N, landmarks, rng):
    return None


@app.cell(hide_code=True)
def _(cases, mo, np):
    # CHECK C3: shape, bounds and reproducibility of the initial particle set.
    try:
        _landmarks = cases["A"]["landmarks"]
        _P = uniform_particles(200, _landmarks, np.random.default_rng(42))
        if _P is None:
            raise ValueError("uniform_particles has not been implemented")
        _P = np.asarray(_P, dtype=float)
        _L = np.asarray(list(_landmarks.values()), dtype=float)
        _lo, _hi = _L.min(axis=0)-0.5, _L.max(axis=0)+0.5
        _shape = _P.shape == (200, 3)
        _bounds = _shape and np.all((_P[:, :2] >= _lo) & (_P[:, :2] <= _hi)) and np.all((_P[:, 2] >= -np.pi) & (_P[:, 2] < np.pi))
        _repeat = uniform_particles(200, _landmarks, np.random.default_rng(42))
        _deterministic = _shape and np.allclose(_P, _repeat)
        _msg = ("✅" if _shape and _bounds and _deterministic else "❌") + f" C3 particles: shape={_shape}, bounds={_bounds}, seeded repeatability={_deterministic}"
    except Exception as _err:
        _msg = f"⏳ Implement uniform_particles first ({type(_err).__name__})."
    mo.md("**CHECK — C3 uniform initialization**\n\n" + _msg)
    return


@app.cell
def _():
    # Global localization: unknown initial pose, 4 EKF headings and 3 PF seeds.
    g_N = 3000
    g_seeds = 3
    success_ate_threshold_m = 1.0  # modify only if you justify the criterion
    RUN_GLOBAL = False
    return RUN_GLOBAL, g_N, g_seeds, success_ate_threshold_m


@app.cell(hide_code=True)
def _(
    RUN_GLOBAL,
    accept_measurement,
    c_eps,
    c_sp,
    c_sr,
    cases,
    e1_Q,
    e1_alphas,
    e1_gate,
    ekf_predict,
    ekf_update,
    eval_run,
    g_N,
    g_seeds,
    mo,
    np,
    pd,
    run_pf,
    success_ate_threshold_m,
):
    global_summary = pd.DataFrame()
    global_paths = {}
    if not RUN_GLOBAL:
        _out = mo.md("Global-localization experiment paused. Set RUN_GLOBAL = True for the final export.")
    elif ekf_predict(np.zeros(3), np.eye(3), 0.1, 0.1, 0.1, e1_alphas)[0] is None or ekf_update(np.zeros(3), np.eye(3), (1.0, 0.0), (1.0, 0.0), e1_Q)[0] is None:
        _out = mo.md("⏳ Paste your L2 functions (Part 0) before running the global-localization experiment.")
    else:
        def _ekf_from_unknown(_run, _mu0, _Sigma0):
            _mu, _Sigma, _t, _u = _mu0.copy(), _Sigma0.copy(), _run["t0"], (0.0, 0.0)
            _out = [[_t, *_mu]]
            for _ev in _run["events"]:
                if _ev[0] > _t:
                    _mu, _Sigma = ekf_predict(_mu, _Sigma, _u[0], _u[1], _ev[0] - _t, e1_alphas)
                    _t = _ev[0]
                if _ev[1] == 0:
                    _u = (_ev[2], _ev[3])
                else:
                    _res = ekf_update(_mu, _Sigma, (_ev[3], _ev[4]), _run["landmarks"][_ev[2]], e1_Q)
                    if accept_measurement(_res[2], _res[3], e1_gate):
                        _mu, _Sigma = _res[0], _res[1]
                _out.append([_t, *_mu])
            return np.asarray(_out)

        _rows=[]
        for _key, _run in cases.items():
            _center = np.asarray(list(_run["landmarks"].values())).mean(axis=0)
            _e_ates, _p_ates=[] , []
            for _angle in (0.0, np.pi/2, np.pi, -np.pi/2):
                _est = _ekf_from_unknown(_run, np.array([*_center, _angle]), np.diag([9.,36.,np.pi**2]))
                _e_ates.append(eval_run(_est, _run, after_s=60)["ATE (m)"])
                if _angle == 0.0:
                    global_paths[(_key,"EKF")] = _est
            for _seed in range(g_seeds):
                _P0 = uniform_particles(g_N, _run["landmarks"], np.random.default_rng(100+_seed))
                if _P0 is None:
                    break
                _est, _info = run_pf(_run, g_N, e1_alphas, (c_sr,c_sp), c_eps, seed=_seed, P0=_P0)
                if _est is None:
                    break
                _p_ates.append(eval_run(_est, _run, after_s=60)["ATE (m)"])
                if _seed == 0:
                    global_paths[(_key,"PF")] = _est
            for _method, _values in (("EKF (4 headings)",_e_ates), (f"PF ({g_seeds} seeds)",_p_ates)):
                if _values:
                    _rows.append({"case": _key, "method":_method, "runs":len(_values),
                                  "successes": sum(v < success_ate_threshold_m for v in _values),
                                  "success criterion": f"ATE < {success_ate_threshold_m:.1f} m",
                                  "ATE after 60 s mean (m)": float(np.mean(_values)),
                                  "ATE after 60 s std (m)": float(np.std(_values)),
                                  "ATE values (m)":", ".join(f"{v:.2f}" for v in _values)})
        global_summary = pd.DataFrame(_rows)
        _out = global_summary.round(3) if not global_summary.empty else mo.md("Implement C3 and verify the L1-L3 functions first.")
    _out
    return (global_paths,)


@app.cell(hide_code=True)
def _(cases, global_paths, mo, plot_errors):
    _case = "A"
    _paths = {name:global_paths[(_case,name)] for name in ("EKF","PF") if (_case,name) in global_paths}
    if _paths:
        _out = plot_errors(cases[_case], _paths)
    else:
        _out = mo.md("Global-localization plot appears after RUN_GLOBAL=True and C3 is implemented.")
    _out
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    #### ✍️ ANSWERS — Part C

    ### C1 — Robust likelihood (10 %, including the code and its CHECK)
    1. Explain the two terms of the mixture and why the uniform component bounds the penalty that one inconsistent measurement can impose on a particle (give the value of the log-likelihood floor for your ε).
    2. Using the C2 table, compare ε = 0 and ε > 0 in both cases. Where does ε matter and where does it not? Relate this to what you found in B3.
    3. In the case where it matters, why can more particles **not** compensate for ε = 0?
    4. Compare the robust likelihood with the EKF gate: soft down-weighting vs. hard rejection. When is each of them risky?

    > *Your answer…*

    ### C2 — Number of particles (12 %)
    1. Report the mean ± std of the ATE and the run time as a function of $N$ (with ε > 0) for both cases.
    2. Where are the diminishing returns? Are they the same for both cases?
    3. Is the spread across the three seeds small enough to rank the configurations? What would you change in the experiment to make the conclusions more reliable?

    > *Your answer…*

    ### C3 — Global localization (13 %, including the code)
    1. Report how many runs succeed under the declared criterion for the EKF (four initial headings) and for the PF (all seeds), in both cases.
    2. Explain the EKF successes and failures: why can the same filter converge from one initial heading and fail from another?
    3. Describe any PF failure or slow convergence you observe and propose a remedy.
    4. Use the error plot to discuss the first seconds of the PF: what does the particle mean represent before convergence?

    > *Your answer…*
    """
    )
    return


@app.cell(hide_code=True)
def _(cases, dr_res, ekf_plain, ekf_res, eval_run, mo, pd, pf_res):
    _rows=[]
    for _key,_run in cases.items():
        _methods = [("Dead reckoning",dr_res.get(_key)),
                    ("EKF ungated",ekf_plain.get(_key)),
                    ("EKF gated",ekf_res.get(_key)),
                    ("PF tracking",pf_res.get(_key))]
        for _method,_state in _methods:
            if _state is not None:
                _rows.append({"case":_key,"method":_method,**eval_run(_state,_run)})
    if _rows:
        _out = mo.vstack([mo.md("**Part D — FINAL COMPARISON.** All four methods are evaluated on the same odometry timestamp grid, with the same initial pose."),pd.DataFrame(_rows).round(3)])
    else:
        _out = mo.md("Final comparison is available after the implementations run.")
    _out
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ---
    ## Part D — Synthesis and comparison (15 %) · maximum 350 words

    | | |
    |---|---|
    | **Given** | The final comparison table above: DR, ungated EKF, gated EKF and PF tracking on both cases, on the same timestamps; your results from Parts A–C. |
    | **You report** | One coherent text (≤ 350 words) answering the four points below, citing **at least two** numerical results from your runs. |

    1. Compare DR, ungated EKF, gated EKF and PF tracking for **both** cases. Which algorithm has the lowest ATE, and is it also best by FTE? Discuss the expected computational trade-off, using the PF runtime measurements from C2; do not claim a measured EKF/PF speed ratio unless you measure both.
    2. Explain which information distinguishes the filters from dead reckoning and what uncertainty representation each uses. Relate your explanation to at least one observation from global localization (C3).
    3. Discuss whether the gated EKF was always better than the ungated EKF and whether PF performance was sensitive to seed or particle count. Which filter would you deploy on these robots, and under which assumptions?
    4. Which lessons are likely to generalize, and which depend on your assigned robot/dataset (revisit your prediction of A1)? State one limitation of this evaluation.

    > *Your answer…*

    ---
    ### ✔️ Before submitting
    - [ ] All CHECK cells green · all ANSWER cells completed with your own numbers.
    - [ ] `STUDENT_CODE` set to your own code.
    - [ ] RUN_B4, RUN_PF_TRACKING, RUN_C2 and RUN_GLOBAL are True; their output tables appear in E1.html.
    - [ ] Notebook runs from top to bottom · `.py` + `.html` in `E1_StudentCode_Surname.zip` · individual submission · deadline **23/11/2026, 23:59**.
    """
    )
    return


if __name__ == "__main__":
    app.run()
