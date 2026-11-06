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
    # E2 — SLAM (assessed)

    **11765 · Navigation and Environment Modelling in Mobile Robotics · MUSI**
    Released Friday 06/11/2026 · **Deadline: Monday 14/12/2026, 23:59** · Individual

    | Part | Content | Weight |
    |---|---|---|
    | 0 | Your functions from L2, L3 and L4 (required, not graded) | — |
    | A | Cases, sensor noise and the EKF-SLAM baseline | 15 % |
    | B | EKF-SLAM with **unknown** data association | 25 % |
    | C | FastSLAM 1.0 (builds on L3) | 30 % |
    | D | Consistency after a long observation gap; filtering vs. graph-based SLAM | 20 % |
    | E | Synthesis (≤ 500 words) | 10 % |

    **What you will do.** In L4 the filter received the identity (barcode) of every landmark. Part A reuses your L4 filter as the baseline; Part B removes the identities and asks you to solve data association; Part C replaces the joint Gaussian by particles that carry one small landmark EKF each (FastSLAM); Part D studies why an EKF can become overconfident and what a graph-based (smoothing) back-end would change. You do not reimplement the L4 filter and you do not repeat the parameter grid of E1.

    **Rules.** Ground truth (trajectories **and** landmark positions) is used **only** for the initial pose, for evaluation and for the offline diagnostics provided in the notebook; it never enters a filter. Barcodes are used as known correspondences in Parts A, C and D; in Part B they are used only to *evaluate* your associations. All numbers in your answers must come from your own runs on your two cases. Report the seeds of every random experiment and never present a favourable seed as the only evidence. A failed experiment with a convincing diagnosis can receive full credit. You must be able to explain every implementation you submit.

    **Cases.** Case A = **Dataset6 / Robot2**, shared by everyone · case B = the dataset/robot assigned to your **E2 student code** (`STUDENT_CODE`, see the assignment handout). First 300 s of each. 

    **Fair comparison.** Trajectory metrics (ATE, FTE, final heading error) are computed at the **same odometry timestamps** with the most recent estimate available at each of them, as in E1. Map errors use the final landmark estimates; the rigid alignment of the map is a diagnostic, not a correction applied by SLAM.

    **🛠 TODO** code · **✅ CHECK** automatic feedback · **✍️ ANSWER** written answers. Submit **individually** `E2_StudentCode_Surname.zip` containing `E2_slam.py` and `E2.html`. Settings and `RUN_*` flags are source-code variables, because the HTML export re-executes the file. Before exporting, set `RUN_B2`, `RUN_C_DEMO`, `RUN_C2` and `RUN_D` to `True` and run `uv run marimo export html notebooks/E2_slam.py -o E2.html` from the repository root; check that the HTML contains the experiment tables.

    **Recommended reading.** Thrun, Burgard & Fox, *Probabilistic Robotics*: Sec. 10.3 (EKF-SLAM with unknown correspondences, Table 10.2), Ch. 11 (GraphSLAM), Ch. 13 (FastSLAM, Table 13.1) · T. Bailey, J. Nieto, J. Guivant, M. Stevens & E. Nebot, "Consistency of the EKF-SLAM algorithm", *IROS* 2006 · T. Bailey, J. Nieto & E. Nebot, "Consistency of the FastSLAM algorithm", *ICRA* 2006 · G. P. Huang, A. I. Mourikis & S. I. Roumeliotis, "Observability-based rules for designing consistent EKF SLAM estimators", *IJRR* 29(5), 2010 · J. Neira & J. D. Tardós, "Data association in stochastic mapping using the joint compatibility test", *IEEE T-RA* 17(6), 2001 · G. Grisetti, R. Kümmerle, C. Stachniss & W. Burgard, "A tutorial on graph-based SLAM", *IEEE ITS Magazine* 2(4), 2010.
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
          odom          (K,3) [t, v, w]             velocity commands
          meas          (J,4) [t, barcode, r, phi]  measurements of LANDMARKS only
          gt            (N,4) [t, x, y, theta]      ground truth (evaluation only!)
          landmarks     {barcode: (x, y)}           ground-truth map (in SLAM: evaluation only!)
          landmark_ids  {barcode: subject number}   landmark labels (L6 ... L20)
          x0, t0        initial pose (from ground truth) and time
          events        odometry (t, 0, v, w) and measurement (t, 1, barcode, r, phi) events, sorted by time
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
            odom=_od, meas=_me, gt=_gt, landmarks=_landmarks, landmark_ids=_landmark_ids, x0=_x0, t0=_t0,
            events=_ev, n_robot_meas=_n_robot, dataset=dataset, robot=robot,
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
        """Compare at supplied timestamps using the most recent filtered state (causal hold).

        For fair comparisons across methods, pass run['odom'][:, 0] as sample_times.
        ATE is RMS XY error; FTE and heading error use the final common timestamp.
        """
        _s = np.asarray(states, float)
        _ts = _s[:, 0] if sample_times is None else np.asarray(sample_times, float)
        _ts = _ts[(_ts >= _s[0, 0]) & (_ts <= _s[-1, 0]) &
                  (_ts >= gt[0, 0]) & (_ts <= gt[-1, 0])]
        if not len(_ts):
            raise ValueError("No common evaluation timestamps. Check trajectory coverage.")
        _idx = np.searchsorted(_s[:, 0], _ts, side="right") - 1
        _v = _s[_idx, 1:]
        _g = gt_at(gt, _ts)
        _e = np.hypot(_v[:, 0] - _g[:, 0], _v[:, 1] - _g[:, 1])
        _eh = (_v[-1, 2] - _g[-1, 2] + np.pi) % (2*np.pi) - np.pi
        return {"ATE (m)": float(np.sqrt(np.mean(_e**2))),
                "FTE (m)": float(_e[-1]),
                "final heading error (deg)": float(abs(np.degrees(_eh)))}

    def align_map(est, ref):
        """Rigid 2-D alignment (rotation + translation, no scale) that best maps the
        estimated landmarks `est` (n,2) onto `ref` (n,2) in the least-squares sense
        (Kabsch/Umeyama). Returns (R, t, est_aligned, angle_deg)."""
        _ce, _cr = est.mean(0), ref.mean(0)
        _U, _, _Vt = np.linalg.svd((est - _ce).T @ (ref - _cr))
        _D = np.diag([1.0, np.sign(np.linalg.det(_Vt.T @ _U.T))])
        _R = _Vt.T @ _D @ _U.T
        _t = _cr - _R @ _ce
        return _R, _t, est @ _R.T + _t, float(np.degrees(np.arctan2(_R[1, 0], _R[0, 0])))

    def path_length(gt):
        return float(np.sum(np.hypot(np.diff(gt[:, 1]), np.diff(gt[:, 2]))))
    return align_map, evaluate, gt_at, load_run, path_length


@app.cell(hide_code=True)
def _(gt_at, np):
    # PROVIDED — extra evaluation helpers for E2
    def measurement_gaps(run, min_gap=10.0):
        """Time intervals (seconds since t0) longer than `min_gap` without any landmark measurement."""
        _t = np.concatenate([[0.0], run["meas"][:, 0] - run["t0"], [run["odom"][-1, 0] - run["t0"]]])
        _d = np.diff(_t)
        return [(float(_t[_i]), float(_t[_i + 1])) for _i in np.where(_d >= min_gap)[0]]

    def nees_position(states, pos_cov, gt, sample_times=None):
        """2D position NEES = e.T @ P_xy^-1 @ e; optionally sample at common timestamps.

        Inputs: states (K,4) [t,x,y,theta], pos_cov (K,2,2), gt (N,4),
        sample_times (J,) optional. Returns NEES (J,) or (K,).
        Uses causal (last available) state and covariance when sample_times is given.
        """
        _s = np.asarray(states, float)
        _P = np.asarray(pos_cov, float)
        if sample_times is not None:
            _t = np.asarray(sample_times, float)
            _t = _t[(_t >= _s[0, 0]) & (_t <= _s[-1, 0]) &
                    (_t >= gt[0, 0]) & (_t <= gt[-1, 0])]
            if not len(_t):
                raise ValueError("No NEES evaluation timestamps within trajectory coverage")
            _idx = np.searchsorted(_s[:, 0], _t, side="right") - 1
            _s, _P = _s[_idx], _P[_idx]
        else:
            _t = _s[:, 0]
        _g = gt_at(gt, _t)
        _e = _s[:, 1:3] - _g[:, :2]
        _P = _P + 1e-12 * np.eye(2)
        return np.einsum("ki,ki->k", _e, np.linalg.solve(_P, _e[..., None])[..., 0])

    def landmark_groups(run, d_min=0.5):
        """{barcode: group index}: landmarks closer than d_min metres to each other (directly or through a chain)
        form one group. Used ONLY to evaluate associations in Part B."""
        _bcs = sorted(run["landmarks"])
        _P = np.array([run["landmarks"][_b] for _b in _bcs])
        _parent = list(range(len(_bcs)))

        def _root(_i):
            while _parent[_i] != _i:
                _i = _parent[_i]
            return _i

        for _i in range(len(_bcs)):
            for _j in range(_i + 1, len(_bcs)):
                if np.hypot(*(_P[_i] - _P[_j])) < d_min:
                    _parent[_root(_i)] = _root(_j)
        _ids = {}
        return {_b: _ids.setdefault(_root(_i), len(_ids)) for _i, _b in enumerate(_bcs)}

    def long_gap_reobservations(run, min_gap=30.0):
        """Reference-only diagnostic: repeated sightings of the SAME landmark after >= min_gap seconds.

        Input: run (loaded dataset), min_gap (seconds since previous sighting of that barcode).
        Returns: list of {barcode, time_s, gap_s} sorted chronologically; the first
        sighting of each barcode is excluded. Does NOT imply a spatial loop closure.
        """
        _last = {}
        _events = []
        for _t, _b, _range, _bearing in run["meas"]:
            _b = int(_b)
            if _b in _last and _t - _last[_b] >= min_gap:
                _events.append({"barcode": _b, "time_s": float(_t - run["t0"]),
                                "gap_s": float(_t - _last[_b])})
            _last[_b] = _t
        return _events

    def sensor_residuals(run, h_fn, min_meas=10):
        """OFFLINE calibration diagnostic: residuals z - h(x_gt, m_gt) of all landmark measurements.
        Pooled std and robust std (1.4826·MAD) of range (m) and bearing (rad), and the largest |mean residual|
        of a single landmark with at least `min_meas` measurements (a per-landmark bias)."""
        _m = run["meas"]
        _g = gt_at(run["gt"], _m[:, 0])
        _res = np.array([_z[2:] - h_fn(_x, run["landmarks"][_z[1]]) for _z, _x in zip(_m, _g)])
        _res[:, 1] = (_res[:, 1] + np.pi) % (2 * np.pi) - np.pi
        _mad = lambda _x: 1.4826 * float(np.median(np.abs(_x - np.median(_x))))
        _bias = np.array([_res[_m[:, 1] == _b].mean(0) for _b in set(_m[:, 1]) if np.sum(_m[:, 1] == _b) >= min_meas]).reshape(-1, 2)
        return {"mean range residual (m)": float(_res[:, 0].mean()),
                "range std (m)": float(_res[:, 0].std(ddof=1)),
                "range robust std (m)": _mad(_res[:, 0]),
                "max |mean range by landmark| (m)": float(np.abs(_bias[:, 0]).max()) if len(_bias) else float("nan"),
                "mean bearing residual (rad)": float(_res[:, 1].mean()),
                "bearing std (rad)": float(_res[:, 1].std(ddof=1)),
                "bearing robust std (rad)": _mad(_res[:, 1]),
                "max |mean bearing by landmark| (rad)": float(np.abs(_bias[:, 1]).max()) if len(_bias) else float("nan")}
    return (
        landmark_groups,
        long_gap_reobservations,
        measurement_gaps,
        nees_position,
        sensor_residuals,
    )


@app.cell(hide_code=True)
def _(go, gt_at, make_subplots, np):
    # PROVIDED — plotting helpers
    _COLORS = ["#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#8c564b"]

    def landmark_label(run, barcode):
        """'L<subject>' if the subject number is known, otherwise the barcode."""
        _ids = run.get("landmark_ids", {})
        return f"L{_ids[barcode]}" if barcode in _ids else f"{int(barcode)}"

    def ellipse_xy(mean, cov, k=3.0, n=60):
        """Points of the k-sigma ellipse of a 2-D Gaussian."""
        _phi = np.linspace(0, 2 * np.pi, n)
        _lam, _vec = np.linalg.eigh(cov)
        _p = _vec @ (k * np.sqrt(np.maximum(_lam, 0))[:, None] * np.stack([np.cos(_phi), np.sin(_phi)]))
        return mean[0] + _p[0], mean[1] + _p[1]

    def plot_slam(run, res, title="EKF-SLAM", extra=None):
        """Ground truth (trajectory + landmarks ★), SLAM trajectory and SLAM map (✕ with 3σ ellipses).
        A thin dotted line joins each estimated landmark to its ground-truth position.
        extra: dict name -> array [t, x, y, theta] of other trajectories (e.g. dead reckoning)."""
        _fig = go.Figure()
        _fig.add_trace(go.Scatter(x=run["gt"][:, 1], y=run["gt"][:, 2], mode="lines", name="ground truth", line=dict(color="#1f77b4", dash="dot")))
        _bcs = list(run["landmarks"].keys())
        _fig.add_trace(go.Scatter(
            x=[run["landmarks"][b][0] for b in _bcs], y=[run["landmarks"][b][1] for b in _bcs],
            mode="markers+text", name="landmarks (ground truth)", text=[landmark_label(run, b) for b in _bcs],
            textposition="top center", textfont=dict(size=11, color="#555555"),
            hovertext=[f"<b>{landmark_label(run, b)}</b> (barcode {int(b)}) — ground truth" for b in _bcs], hoverinfo="text",
            marker=dict(symbol="star", size=11, color="gold", line=dict(width=0.5, color="#8a6d00")),
        ))
        for _i, (_name, _s) in enumerate((extra or {}).items()):
            if _s is not None:
                _fig.add_trace(go.Scatter(x=_s[:, 1], y=_s[:, 2], mode="lines", name=_name, line=dict(color=_COLORS[(_i + 1) % 5], width=1)))
        if res is not None:
            _s = res["states"]
            _fig.add_trace(go.Scatter(x=_s[:, 1], y=_s[:, 2], mode="lines", name="SLAM trajectory", line=dict(color=_COLORS[0])))
            _mu, _Sig = res["mu"], res["Sigma"]
            _ex, _ey, _lab = [], [], []
            for _j, _b in enumerate(res["ids"]):
                _m = _mu[3 + 2 * _j: 5 + 2 * _j]
                _ex.append(_m[0])
                _ey.append(_m[1])
                _lab.append(landmark_label(run, _b))
                _x, _y = ellipse_xy(_m, _Sig[3 + 2 * _j: 5 + 2 * _j, 3 + 2 * _j: 5 + 2 * _j])
                _fig.add_trace(go.Scatter(x=_x, y=_y, mode="lines", line=dict(color="#d62728", width=1), showlegend=False, hoverinfo="skip"))
                if _b in run["landmarks"]:
                    _g = run["landmarks"][_b]
                    _fig.add_trace(go.Scatter(x=[_m[0], _g[0]], y=[_m[1], _g[1]], mode="lines", line=dict(color="gray", width=1, dash="dot"), showlegend=False, hoverinfo="skip"))
            _fig.add_trace(go.Scatter(
                x=_ex, y=_ey, mode="markers+text", name="SLAM map (3σ)", text=_lab, textposition="bottom center",
                textfont=dict(size=10, color="#d62728"), marker=dict(symbol="x", size=9, color="#d62728"),
                hovertext=[f"<b>{l}</b> — estimate" for l in _lab], hoverinfo="text",
            ))
        _fig.update_layout(title=title, height=560, plot_bgcolor="white", xaxis=dict(title="x (m)", constrain="domain"), yaxis=dict(title="y (m)", scaleanchor="x", scaleratio=1))
        return _fig

    def plot_errors(run, estimates):
        """Position and heading error vs time for several estimates [t, x, y, theta]."""
        _fig = make_subplots(rows=2, cols=1, shared_xaxes=True, subplot_titles=("Position error (m)", "Heading error (deg)"))
        for _i, (_name, _s) in enumerate(estimates.items()):
            if _s is None:
                continue
            _g = gt_at(run["gt"], _s[:, 0])
            _t = _s[:, 0] - run["t0"]
            _fig.add_trace(go.Scatter(x=_t, y=np.hypot(_s[:, 1] - _g[:, 0], _s[:, 2] - _g[:, 1]), mode="lines", name=_name, line=dict(color=_COLORS[_i % 5])), row=1, col=1)
            _fig.add_trace(go.Scatter(x=_t, y=np.degrees((_s[:, 3] - _g[:, 2] + np.pi) % (2 * np.pi) - np.pi), mode="lines", name=_name, showlegend=False, line=dict(color=_COLORS[_i % 5])), row=2, col=1)
        _fig.update_xaxes(title_text="time (s)", row=2, col=1)
        _fig.update_layout(height=480, plot_bgcolor="white")
        return _fig
    return landmark_label, plot_errors, plot_slam


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Provided functions from L1–L3
    `wrap_angle`, `motion_euler`, `G_jacobian`, `V_jacobian`, `control_noise`, `dead_reckoning` and `num_jacobian` (CHECK cells only) are the same as in L4.
    """
    )
    return


@app.cell(hide_code=True)
def _(np):
    # PROVIDED — L1/L2 building blocks, dead reckoning and a numerical Jacobian for the CHECK cells
    def wrap_angle(a):
        return (a + np.pi) % (2 * np.pi) - np.pi

    def motion_euler(x, v, w, dt):
        return np.array([x[0] + v * dt * np.cos(x[2]),
                         x[1] + v * dt * np.sin(x[2]),
                         wrap_angle(x[2] + w * dt)])

    def G_jacobian(x, v, w, dt):
        """∂g/∂x of the Euler model (3x3)."""
        return np.array([[1.0, 0.0, -v * dt * np.sin(x[2])],
                         [0.0, 1.0, v * dt * np.cos(x[2])],
                         [0.0, 0.0, 1.0]])

    def V_jacobian(x, v, w, dt):
        """∂g/∂u of the Euler model (3x2)."""
        return np.array([[dt * np.cos(x[2]), 0.0],
                         [dt * np.sin(x[2]), 0.0],
                         [0.0, dt]])

    def control_noise(v, w, alphas):
        """M_t (2x2), Thrun's convention."""
        return np.diag([alphas[0] * v**2 + alphas[1] * w**2,
                        alphas[2] * v**2 + alphas[3] * w**2])

    def dead_reckoning(run):
        _x = run["x0"].copy()
        _t = run["t0"]
        _u = (0.0, 0.0)
        _out = [[_t, *_x]]
        for _e in run["events"]:
            if _e[1] != 0:
                continue
            if _e[0] > _t:
                _x = motion_euler(_x, _u[0], _u[1], _e[0] - _t)
                _t = _e[0]
                _out.append([_t, *_x])
            _u = (_e[2], _e[3])
        return np.array(_out)

    def num_jacobian(f, x, eps=1e-6):
        """Central finite differences. Used only by the CHECK cells, not by the filter."""
        _x = np.asarray(x, float)
        _f0 = np.asarray(f(_x))
        _J = np.zeros((_f0.size, _x.size))
        for _i in range(_x.size):
            _d = np.zeros_like(_x)
            _d[_i] = eps
            _J[:, _i] = (np.asarray(f(_x + _d)) - np.asarray(f(_x - _d))) / (2 * eps)
        return _J
    return (
        G_jacobian,
        V_jacobian,
        control_noise,
        dead_reckoning,
        motion_euler,
        num_jacobian,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ---
    ## 0 · Your functions from L2, L3 and L4

    ### 🛠 TODO 0
    Paste **your own tested functions** from L2, L3 and L4, retaining their signatures. These are prerequisites, not new assessed tasks. The CHECK repeats numerical tests.

    **L2 measurement model:** `h_landmark(mu, landmark)` takes robot pose `(3,)` and landmark position `(2,)`, and returns expected `[range(m), bearing(rad)]` `(2,)`; `H_landmark` takes the same arguments and returns its `(2,3)` Jacobian.

    **L4 joint EKF:** `slam_predict(mu, Sigma, v, w, dt, alphas)` takes the joint mean `(3+2n,)`, covariance `(3+2n,3+2n)`, control velocities, time increment and noise parameters; returns `(mu_bar, Sigma_bar)` of the same sizes. `h_slam(mu,j)` and `H_slam(mu,j)` take a zero-based landmark slot and return `(2,)` and `(2,3+2n)`; `add_landmark(mu,Sigma,z,Q)` returns a two-coordinate-augmented state/covariance; `slam_update(mu_bar,Sigma_bar,z,j,Q)` returns `(mu, Sigma, nu, S, K)` of shapes `(3+2n,)`, `(3+2n,3+2n)`, `(2,)`, `(2,2)`, `(3+2n,2)`.

    **L3 particle utilities:** `sample_motion_model(P,v,w,dt,alphas,rng)` propagates pose particles `(M,3)` and returns `(M,3)`; `normalize_log_weights(logw)` returns normalised `(M,)` weights; `effective_sample_size(w)` returns a scalar; and `low_variance_resample(P,w,rng)` returns a resampled array with the same shape as `P` (also works on 1D index arrays).
    """
    )
    return


@app.cell
def _():
    # PASTE your tested L2 measurement model (keep these names and signatures).
    def h_landmark(mu, landmark):
        return None

    def H_landmark(mu, landmark):
        return None
    return H_landmark, h_landmark


@app.cell
def _():
    # PASTE your tested L4 EKF-SLAM functions; these are prerequisites, not new E2 tasks.
    def slam_predict(mu, Sigma, v, w, dt, alphas):
        return None, None

    def h_slam(mu, j):
        return None

    def H_slam(mu, j):
        return None

    def add_landmark(mu, Sigma, z, Q):
        return None, None

    def slam_update(mu_bar, Sigma_bar, z, j, Q):
        return None, None, None, None, None
    return H_slam, add_landmark, h_slam, slam_predict, slam_update


@app.cell
def _():
    # PASTE your L3 particle utilities; preserve the behaviour on a 1D index array.
    def sample_motion_model(P, v, w, dt, alphas, rng):
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
        normalize_log_weights,
        sample_motion_model,
    )


@app.cell(hide_code=True)
def _(
    G_jacobian,
    H_landmark,
    H_slam,
    V_jacobian,
    add_landmark,
    control_noise,
    effective_sample_size,
    h_landmark,
    h_slam,
    low_variance_resample,
    mo,
    motion_euler,
    normalize_log_weights,
    np,
    num_jacobian,
    sample_motion_model,
    slam_predict,
    slam_update,
):
    def _check():
        try:
            _rng = np.random.default_rng(0)
            _mu = np.array([1.0, -0.5, 0.4, 3.0, 1.0, -1.0, 2.0])
            _A = _rng.normal(size=(7, 7))
            _Sig = 0.01 * _A @ _A.T + 0.01 * np.eye(7)
            _Q = np.diag([0.1, 0.05]) ** 2
            _Fx = np.zeros((3, 7))
            _Fx[:, :3] = np.eye(3)
            _u, _dt, _a = (0.3, 0.2), 0.1, (0.1, 0.01, 0.01, 0.1)
            _Gf = np.eye(7) + _Fx.T @ (G_jacobian(_mu[:3], *_u, _dt) - np.eye(3)) @ _Fx
            _V = V_jacobian(_mu[:3], *_u, _dt)
            _mb, _Sb = slam_predict(_mu, _Sig, *_u, _dt, _a)
            _ma, _Sa = add_landmark(_mu, _Sig, (2.0, 0.3), _Q)
            _z = (2.0, 0.3)
            _a0 = _mu[2] + _z[1]
            _Gr = np.array([[1, 0, -_z[0]*np.sin(_a0)], [0, 1, _z[0]*np.cos(_a0)]])
            _Gz = np.array([[np.cos(_a0), -_z[0]*np.sin(_a0)], [np.sin(_a0), _z[0]*np.cos(_a0)]])
            _Scross = _Gr @ _Sig[:3, :]
            # Independent check of the complete mapped-landmark correction.
            _zi = h_slam(_mu, 1) + np.array([0.04, -0.03])
            _Hi = H_slam(_mu, 1)
            _nui = _zi - h_slam(_mu, 1)
            _nui[1] = (_nui[1] + np.pi) % (2 * np.pi) - np.pi
            _Si = _Hi @ _Sig @ _Hi.T + _Q
            _Ki = _Sig @ _Hi.T @ np.linalg.inv(_Si)
            _u0, _S0 = _mu.copy(), _Sig.copy()
            _mc, _Sc, _nu, _Sout, _Kout = slam_update(_mu, _Sig, _zi, 1, _Q)
            _r = [
                ("h_landmark / H_landmark (L2)", np.allclose(H_landmark(_mu[:3], (3.0, 1.0)), num_jacobian(lambda x: h_landmark(x, (3.0, 1.0)), _mu[:3]), atol=1e-6)),
                ("slam_predict (L4)", np.allclose(_Sb, _Gf @ _Sig @ _Gf.T + _Fx.T @ _V @ control_noise(*_u, _a) @ _V.T @ _Fx, atol=1e-12) and np.allclose(_mb[:3], motion_euler(_mu[:3], *_u, _dt))),
                ("h_slam / H_slam (L4)", np.allclose(H_slam(_mu, 1), num_jacobian(lambda y: h_slam(y, 1), _mu), atol=1e-6)),
                ("add_landmark new covariance + cross-covariance (L4)", np.allclose(_Sa[7:, 7:], _Gr @ _Sig[:3, :3] @ _Gr.T + _Gz @ _Q @ _Gz.T) and np.allclose(_Sa[7:, :7], _Scross)),
                ("slam_update mean, covariance, innovation, S, gain (L4)",
                 np.allclose(_mc, _mu + _Ki @ _nui) and
                 np.allclose(_Sc, _Sig - _Ki @ _Si @ _Ki.T, atol=1e-9) and
                 np.allclose(_nu, _nui) and np.allclose(_Sout, _Si) and np.allclose(_Kout, _Ki)),
                ("slam_update does not mutate its inputs", np.array_equal(_mu, _u0) and np.array_equal(_Sig, _S0)),
                ("sample_motion_model (L3)", np.allclose(sample_motion_model(np.tile(_mu[:3], (20000, 1)), 0.5, 0.4, 0.5, (0.1, 0.05, 0.05, 0.1), np.random.default_rng(1)).mean(0), motion_euler(_mu[:3], 0.5, 0.4, 0.5), atol=5e-3)),
                ("normalize_log_weights / effective_sample_size (L3)", np.isclose(normalize_log_weights(np.array([-1000.0, -1001.0]))[0], 1 / (1 + np.exp(-1))) and np.isclose(effective_sample_size(np.ones(4) / 4), 4)),
                ("low_variance_resample (L3), also on an index array", [int(np.sum(low_variance_resample(np.arange(4), np.array([0.5, 0.25, 0.25, 0.0]), np.random.default_rng(3)) == _i)) for _i in range(4)] == [2, 1, 1, 0]),
            ]
        except Exception as _e:
            return f"⏳ not pasted yet ({type(_e).__name__})"
        return "<br>".join(f"{'✅' if _ok else '❌'} {_n}" for _n, _ok in _r)

    mo.md("**✅ CHECK 0 — L2/L3/L4 functions**<br>" + _check())
    return


@app.cell(hide_code=True)
def _(add_landmark, np, slam_predict, slam_update, time):
    # PROVIDED — EKF-SLAM with KNOWN correspondences (the L4 driver, plus the NIS of every update).
    def run_ekf_slam(run, alphas, Q, blackout=None, decouple=False):
        """blackout: (start, end) in seconds since t0 — landmark measurements in that window are ignored.
        decouple: if True, the robot–map and landmark–landmark covariances are set to zero after every
                  measurement (a deliberately WRONG filter, used only in Part 4).
        Returns a dict: states [t, x, y, theta], robot_cov (K,3,3), mu, Sigma, ids (barcode of each slot),
        lm_std {barcode: [(t, std)]}, first_seen {barcode: {...}}, n_obs, info."""
        _mu = run["x0"].copy()
        _Sigma = np.diag([1e-6] * 3)
        if slam_predict(_mu, _Sigma, 0.1, 0.1, 0.1, alphas)[0] is None or add_landmark(_mu, _Sigma, (1.0, 0.0), Q)[0] is None:
            return None
        _t, _u = run["t0"], (0.0, 0.0)
        _ids, _slot = [], {}
        _states, _rcov = [[_t, *_mu]], [_Sigma[:3, :3].copy()]
        _lm_std, _first, _nobs = {}, {}, {}
        _info = {"updates": 0, "new landmarks": 0, "ignored (blackout)": 0}
        _nis = []
        _tic = time.perf_counter()
        for _e in run["events"]:
            if _e[0] > _t:
                _mu, _Sigma = slam_predict(_mu, _Sigma, _u[0], _u[1], _e[0] - _t, alphas)
                _t = _e[0]
            if _e[1] == 0:
                _u = (_e[2], _e[3])
                _states.append([_t, *_mu[:3]])
                _rcov.append(_Sigma[:3, :3].copy())
                continue
            if blackout is not None and blackout[0] <= _t - run["t0"] < blackout[1]:
                _info["ignored (blackout)"] += 1
                continue
            _b, _z = _e[2], (_e[3], _e[4])
            if _b not in _slot:
                _first[_b] = {"t": _t - run["t0"], "robot std (m)": float(np.sqrt(_Sigma[0, 0] + _Sigma[1, 1]))}
                _slot[_b] = len(_ids)
                _ids.append(_b)
                _mu, _Sigma = add_landmark(_mu, _Sigma, _z, Q)
                _info["new landmarks"] += 1
            else:
                _res = slam_update(_mu, _Sigma, _z, _slot[_b], Q)
                if _res[0] is None:
                    return None
                _mu, _Sigma = _res[0], _res[1]
                _nis.append((_t - run["t0"], _b, float(_res[2] @ np.linalg.solve(_res[3], _res[2]))))
                _info["updates"] += 1
            _Sigma = 0.5 * (_Sigma + _Sigma.T)
            if decouple:
                _D = np.zeros_like(_Sigma)
                _D[:3, :3] = _Sigma[:3, :3]
                for _k in range(len(_ids)):
                    _D[3 + 2 * _k: 5 + 2 * _k, 3 + 2 * _k: 5 + 2 * _k] = _Sigma[3 + 2 * _k: 5 + 2 * _k, 3 + 2 * _k: 5 + 2 * _k]
                _Sigma = _D
            _nobs[_b] = _nobs.get(_b, 0) + 1
            _d = np.diag(_Sigma)[3:].reshape(-1, 2).sum(1)
            for _k, _bb in enumerate(_ids):
                _lm_std.setdefault(_bb, []).append((_t - run["t0"], float(np.sqrt(_d[_k]))))
            _states.append([_t, *_mu[:3]])
            _rcov.append(_Sigma[:3, :3].copy())
        _info["run time (s)"] = round(time.perf_counter() - _tic, 2)
        return dict(states=np.array(_states), robot_cov=np.array(_rcov), mu=_mu, Sigma=_Sigma, ids=_ids,
                    lm_std=_lm_std, first_seen=_first, n_obs=_nobs, info=_info, nis=_nis)

    def map_table(run, res, align_fn, landmark_label_fn):
        """Per-landmark evaluation of a SLAM map against the ground-truth map."""
        import pandas as _pd
        _bcs = [_b for _b in res["ids"] if _b in run["landmarks"]]
        _est = np.array([res["mu"][3 + 2 * res["ids"].index(_b): 5 + 2 * res["ids"].index(_b)] for _b in _bcs])
        _ref = np.array([run["landmarks"][_b] for _b in _bcs])
        _R, _tr, _al, _ang = align_fn(_est, _ref)
        _rows = []
        for _i, _b in enumerate(_bcs):
            _j = res["ids"].index(_b)
            _P = res["Sigma"][3 + 2 * _j: 5 + 2 * _j, 3 + 2 * _j: 5 + 2 * _j]
            _err = _est[_i] - _ref[_i]
            _rows.append({
                "landmark": landmark_label_fn(run, _b), "barcode": int(_b), "slot": _j,
                "first seen (s)": round(res["first_seen"][_b]["t"], 1), "observations": res["n_obs"][_b],
                "error (m)": round(float(np.hypot(*_err)), 3),
                "3σ radius (m)": round(float(3 * np.sqrt(np.linalg.eigvalsh(_P)[-1])), 3),
                "Mahalanobis² (χ²₂: 99 % < 9.21)": round(float(_err @ np.linalg.solve(_P, _err)), 1),
                "error after alignment (m)": round(float(np.hypot(*(_al[_i] - _ref[_i]))), 3),
            })
        _summary = {
            "map RMSE (m)": float(np.sqrt(np.mean(np.sum((_est - _ref) ** 2, 1)))),
            "map RMSE after rigid alignment (m)": float(np.sqrt(np.mean(np.sum((_al - _ref) ** 2, 1)))),
            "alignment rotation (deg)": _ang,
            "alignment translation (m)": float(np.hypot(*_tr)),
        }
        return _pd.DataFrame(_rows), _summary
    return map_table, run_ekf_slam


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ---
    ## Part A — Cases, sensor noise and the EKF-SLAM baseline (15 %)

    Part A is the **control experiment** of E2: your L4 **EKF-SLAM** filter with known correspondences, run on both cases with the data, time grid and noise model that Parts B–D also use.

    **Measurement-noise diagnostics.** Since MRCLAM provides reference robot poses and landmark positions, we can compare each recorded measurement with the corresponding ideal range–bearing prediction. This is an offline diagnostic: it uses ground-truth information that is not available to the SLAM algorithm during normal operation. Use the residual statistics to inform your choice of \(Q\), while considering systematic biases, outliers and possible reference-data errors.

    **Noise model.** The initial values of \(\sigma_r\) and \(\sigma_\phi\) are suggested starting points, not calibrated sensor specifications. You may change them based on the measurement-residual diagnostics. Report and justify your final values.
    """
    )
    return


@app.cell
def _():
    # Use your E2 code (see the assignment handout). Do not edit the two prescribed cases or the duration.
    STUDENT_CODE = "E2-01"  # TODO: replace with your own code (E2-01 ... E2-09)
    ASSIGNED_CASES = {
        "E2-01": ("MRCLAM_Dataset5", "Robot5"),
        "E2-02": ("MRCLAM_Dataset3", "Robot5"),
        "E2-03": ("MRCLAM_Dataset2", "Robot1"),
        "E2-04": ("MRCLAM_Dataset4", "Robot4"),
        "E2-05": ("MRCLAM_Dataset3", "Robot4"),
        "E2-06": ("MRCLAM_Dataset5", "Robot3"),
        "E2-07": ("MRCLAM_Dataset4", "Robot5"),
        "E2-08": ("MRCLAM_Dataset2", "Robot5"),
        "E2-09": ("MRCLAM_Dataset2", "Robot3"),
    }
    if STUDENT_CODE not in ASSIGNED_CASES:
        raise ValueError(f"Set STUDENT_CODE to your own code from the assignment handout (E2-01 ... E2-09), not {STUDENT_CODE!r}.")
    caseA_ds, caseA_rb = "MRCLAM_Dataset6", "Robot2"
    caseB_ds, caseB_rb = ASSIGNED_CASES[STUDENT_CODE]
    e2_duration = 300.0
    return caseA_ds, caseA_rb, caseB_ds, caseB_rb, e2_duration


@app.cell
def _(np):
    # Noise model of ALL baselines (Parts A-D). Justify it in A2; if you change it, change it here, in the source.
    e2_alphas = (0.20, 0.02, 0.02, 0.50)   # process noise, Thrun's convention (as in L2-L4 and E1)
    e2_sigma_r = 0.20                      # range noise (m)      -> compare with the residual table
    e2_sigma_phi = 0.03                    # bearing noise (rad)  -> compare with the residual table
    e2_Q = np.diag([e2_sigma_r**2, e2_sigma_phi**2])
    return e2_Q, e2_alphas


@app.cell(hide_code=True)
def _(caseA_ds, caseA_rb, caseB_ds, caseB_rb, e2_duration, load_run):
    cases = {
        "A": load_run(caseA_ds, caseA_rb, e2_duration),
        "B": load_run(caseB_ds, caseB_rb, e2_duration),
    }
    return (cases,)


@app.cell(hide_code=True)
def _(
    cases,
    h_landmark,
    landmark_groups,
    measurement_gaps,
    mo,
    np,
    path_length,
    pd,
    sensor_residuals,
):
    _ready = h_landmark(np.zeros(3), (1.0, 0.0)) is not None   # residuals need your L2 model
    _rows = []
    for _k, _r in cases.items():
        _P = np.array(list(_r["landmarks"].values()))
        _D = np.hypot(_P[:, None, 0] - _P[None, :, 0], _P[:, None, 1] - _P[None, :, 1]) + np.diag(np.full(len(_P), np.inf))
        _rows.append({
            "case": _k, "run": f"{_r['dataset'][-8:]}/{_r['robot']}", "path (m)": round(path_length(_r["gt"]), 1),
            "measurements": len(_r["meas"]), "landmarks seen": len(set(_r["meas"][:, 1])),
            "surveyed landmark groups (< 0.5 m)": len(set(landmark_groups(_r).values())),
            "min surveyed landmark distance (m)": round(float(_D.min()), 2),
            "gaps ≥ 10 s (s)": ", ".join(f"{_a:.0f}–{_b:.0f}" for _a, _b in measurement_gaps(_r, 10.0)) or "—",
            **({_kk: round(_v, 3) for _kk, _v in sensor_residuals(_r, h_landmark).items()} if _ready else {}),
        })
    mo.vstack([mo.md("**A — the two cases** (the residual columns appear once `h_landmark` is pasted)"),
               mo.ui.table(pd.DataFrame(_rows), selection=None)])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""**How the next (provided) cell works.** `ekf_known` stores the final filter result of each case (`ekf_known["A"]`, `ekf_known["B"]`), and `_rows` collects two rows per case, dead reckoning and EKF-SLAM, both evaluated at the same odometry timestamps (`sample_times=run["odom"][:, 0]`). `map_table` compares the final map with the ground-truth map only after the run. *Consistent landmarks* counts the landmarks whose error lies inside their 99 % ellipse ($d^2=e^\top P_j^{-1}e<9.21$); the position NEES $e^\top P^{-1}e$ of a consistent filter has mean 2 and stays below 11.83 99.7 % of the time. Both are single-run diagnostics: consecutive samples and landmarks are strongly correlated, so they are evidence, not a statistical test.""")
    return


@app.cell(hide_code=True)
def _(
    align_map,
    cases,
    dead_reckoning,
    e2_Q,
    e2_alphas,
    evaluate,
    h_landmark,
    landmark_label,
    map_table,
    mo,
    nees_position,
    np,
    pd,
    run_ekf_slam,
):
    ekf_known = {}     # final filter result of each case: {"A": result, "B": result}
    _rows = []         # two rows per case: dead reckoning and EKF-SLAM
    _ready = h_landmark(np.zeros(3), (1.0, 0.0)) is not None   # has the L2 model been pasted?
    for _k, _r in (cases.items() if _ready else []):
        _res = run_ekf_slam(_r, e2_alphas, e2_Q)
        ekf_known[_k] = _res
        if _res is None:
            continue
        _T = _r["odom"][:, 0]   # common evaluation timestamps
        _df, _sm = map_table(_r, _res, align_map, landmark_label)
        _m2 = _df["Mahalanobis² (χ²₂: 99 % < 9.21)"].to_numpy()
        _nees = nees_position(_res["states"], _res["robot_cov"][:, :2, :2], _r["gt"], sample_times=_r["odom"][:, 0])
        _rows.append({"case": _k, "method": "dead reckoning",
                      **{_kk: round(_v, 3) for _kk, _v in evaluate(dead_reckoning(_r), _r["gt"], sample_times=_T).items()}})
        _rows.append({"case": _k, "method": "EKF-SLAM (known corr.)",
                      **{_kk: round(_v, 3) for _kk, _v in evaluate(_res["states"], _r["gt"], sample_times=_T).items()},
                      "map RMSE (m)": round(_sm["map RMSE (m)"], 3), "map RMSE aligned (m)": round(_sm["map RMSE after rigid alignment (m)"], 3),
                      "map rotation (deg)": round(_sm["alignment rotation (deg)"], 1),
                      "consistent landmarks": f"{int(np.sum(_m2 < 9.21))}/{len(_m2)}",
                      "median position NEES": round(float(np.median(_nees)), 1), "% NEES < 11.83": round(100 * float(np.mean(_nees < 11.83)), 1)})
    (mo.md("⏳ Paste your functions in TODO 0 first.") if not _rows else mo.vstack([
        mo.md("**A — dead reckoning vs EKF-SLAM with known correspondences** (common odometry timestamps)"),
        mo.ui.table(pd.DataFrame(_rows), selection=None),
    ]))
    return (ekf_known,)


@app.cell(hide_code=True)
def _(cases, dead_reckoning, ekf_known, mo, plot_slam):
    _k = "A"  # one plot per cell keeps the marimo output small
    (mo.md("") if ekf_known.get(_k) is None else
     plot_slam(cases[_k], ekf_known[_k], title=f"EKF-SLAM, known correspondences — case {_k} ({cases[_k]['dataset'][-8:]}/{cases[_k]['robot']})", extra={"dead reckoning": dead_reckoning(cases[_k])}))
    return


@app.cell(hide_code=True)
def _(cases, ekf_known, mo, plot_errors):
    _k = "A"
    mo.md("") if ekf_known.get(_k) is None else plot_errors(cases[_k], {"EKF-SLAM": ekf_known[_k]["states"]})
    return


@app.cell(hide_code=True)
def _(cases, dead_reckoning, ekf_known, mo, plot_slam):
    _k = "B"  # one plot per cell keeps the marimo output small
    (mo.md("") if ekf_known.get(_k) is None else
     plot_slam(cases[_k], ekf_known[_k], title=f"EKF-SLAM, known correspondences — case {_k} ({cases[_k]['dataset'][-8:]}/{cases[_k]['robot']})", extra={"dead reckoning": dead_reckoning(cases[_k])}))
    return


@app.cell(hide_code=True)
def _(cases, ekf_known, mo, plot_errors):
    _k = "B"
    mo.md("") if ekf_known.get(_k) is None else plot_errors(cases[_k], {"EKF-SLAM": ekf_known[_k]["states"]})
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    #### ✍️ ANSWERS — Part A

    **A1** Report at least two numerical differences between your cases, including observation gaps and landmark spacing. Which case do you expect to be more difficult for SLAM, and at what point in the trajectory? Explain your prediction.

    > *Your answer…*

    **A2** Report your chosen motion-noise coefficients α and measurement-noise covariance Q. Compare σr and σφ with the observed range and bearing residuals (mean, standard deviation, and differences between landmarks). Explain how a systematic bias differs from random noise, and justify your choices. You may use the suggested defaults if justified.

    > *Your answer…*

    **A3** Compare dead reckoning and known-correspondence EKF-SLAM for both cases using ATE, FTE, and final heading error. For EKF-SLAM, compare raw and aligned map RMSE. What does the difference between these map errors tell you?

    > *Your answer…*

    **A4** Compare the reported map and robot-position uncertainty with the actual errors using the landmark Mahalanobis² counts and position NEES. Does the filter appear overconfident? Explain why one trajectory cannot establish statistical consistency.

    > *Your answer…*
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Part B — EKF-SLAM with unknown data association (25 %)

    In Part A, the barcode identified which landmark produced each measurement. In this part, **the barcode is not available to the filter**.

    When the robot receives a new range–bearing measurement, it must decide whether to:

    - associate it with a landmark already in the map;
    - initialise a new landmark;
    - discard it because the association is uncertain.

    We will use **nearest-neighbour data association based on the Mahalanobis distance**.

    ### Comparing a measurement with the map

    For each landmark \(j\) already stored in the map:

    1. **Predict its measurement.** Use `h_slam(mu, j)` to calculate the range and bearing that the robot expects to observe.
    2. **Calculate the innovation.** Subtract the prediction from the actual measurement, wrapping the bearing difference to \([-\pi,\pi)\).
    3. **Calculate the innovation covariance.** Use `H_slam(mu, j)` to compute $S_j = H_j \Sigma H_j^\top + Q$. This accounts for uncertainty in the robot pose, the mapped landmarks and the sensor measurement.
    4. **Calculate the squared Mahalanobis distance** between the observation and the prediction, $d_j^2=\nu_j^\top S_j^{-1}\nu_j$.
    5. **Select the best candidate.** Find the landmark index \(j^\star\) with the smallest squared Mahalanobis distance.

    ### Mahalanobis distance and validation gates

    The Mahalanobis distance measures how different a measurement is from its prediction **relative to the expected uncertainty**. Unlike Euclidean distance, it accounts for the covariance \(S_j\).

    In the two-dimensional range–bearing measurement space, points with the same squared Mahalanobis distance form an ellipse around the predicted measurement.

    For Gaussian measurement errors, \(d_j^2\) follows a chi-squared distribution with two degrees of freedom when the association is correct and the model is accurate. The threshold \(\gamma_{\text{gate}}=9.21\) defines an ellipse containing approximately **99% of measurements from that landmark** under these assumptions.

    We use two thresholds, with \(0<\gamma_{\text{gate}}\leq\gamma_{\text{new}}\):

    | Condition | Decision |
    |---|---|
    | \(d_{\min}^2\leq\gamma_{\text{gate}}\) | Associate with the best existing landmark \(j^\star\) |
    | \(d_{\min}^2>\gamma_{\text{new}}\) | Initialise a new landmark |
    | \(\gamma_{\text{gate}}<d_{\min}^2\leq\gamma_{\text{new}}\) | Discard the ambiguous measurement |

    If the map is empty, initialise a new landmark.

    ### 🛠 TODO B1 — `associate(mu, Sigma, z, Q, gate, new_thr)`

    Implement the association procedure described above.

    **Inputs**

    - `mu`: joint robot–map mean, shape `(3+2n,)`. Its first three entries are the robot pose `[x, y, theta]` (m, m, rad), followed by `n` landmark positions `[mx, my]` (m). Landmark `j` is stored at `mu[3+2*j : 5+2*j]`.
    - `Sigma`: covariance of the complete robot–map state, shape `(3+2n, 3+2n)`.
    - `z`: observed measurement `[range, bearing]`, shape `(2,)`, in metres and radians.
    - `Q`: range–bearing measurement-noise covariance, shape `(2, 2)`.
    - `gate`: squared Mahalanobis threshold for accepting an existing landmark (e.g. `9.21`).
    - `new_thr`: squared Mahalanobis threshold above which the observation is treated as a new landmark (e.g. `30.0`).

    **Returns**

    Return a tuple `(j, d2_min)`:

    - `j`: integer identifying what to do with the measurement:
        - Return **the index `j_star` of the best matching landmark** (an integer from `0` to `n-1`) when `d2_min <= gate`.
        - Return `-1` when a new landmark should be created.
        - Return `-2` when the measurement should be discarded.
    - `d2_min`: smallest squared Mahalanobis distance among all mapped landmarks, returned as a float. Return `np.inf` when the map is empty.

    **Implementation requirements**

    Use `h_slam` and `H_slam`, wrap the bearing innovation, and do not modify the input arrays. The barcode must not be used.

    ### Evaluation

    The provided experiment compares the filter's estimated associations with the true barcodes **after the run**. It reports incorrect associations, duplicate landmarks, merged landmarks and discarded measurements.

    It also evaluates associations at two levels:

    - **Barcode level:** each physical landmark has its own identity.
    - **Group level:** nearby landmarks connected by distances below 0.5 m are considered one spatial group for evaluation.

    Use both to identify whether association errors mainly occur between nearby landmarks or between different regions of the map. Group-level agreement does not mean that the individual barcode association was correct.
    """
    )
    return


@app.function
def associate(mu, Sigma, z, Q, gate, new_thr):
    # TODO B1: nearest-neighbour / new / discard decision.
    return None, None


@app.cell(hide_code=True)
def _(H_slam, h_slam, mo, np):
    def _check():
        try:
            _mu = np.array([0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 3.0, -3.0, 0.5])
            _Sig = np.diag([0.01, 0.01, 0.001] + [0.01] * 6)
            _Q = np.diag([0.1, 0.05]) ** 2
            _z1 = h_slam(_mu, 1)
            _H = H_slam(_mu, 1)
            _S = _H @ _Sig @ _H.T + _Q
            # a measurement at Mahalanobis distance exactly 12 from landmark 1 (in the range direction)
            _zmid = _z1 + np.sqrt(12.0) * np.linalg.cholesky(_S)[:, 0]
            _mu_before, _Sig_before = _mu.copy(), _Sig.copy()
            _ = associate(_mu, _Sig, _zmid, _Q, 9.21, 30.0)
            _r = [
                ("empty map → new landmark and d²_min = infinity", associate(np.zeros(3), np.eye(3) * 0.01, (2.0, 0.1), _Q, 9.21, 30.0) == (-1, np.inf)),
                ("exact measurement of slot 1 → (1, ≈0)", associate(_mu, _Sig, _z1, _Q, 9.21, 30.0)[0] == 1 and associate(_mu, _Sig, _z1, _Q, 9.21, 30.0)[1] < 1e-9),
                ("far measurement → new landmark", associate(_mu, _Sig, (6.0, 2.5), _Q, 9.21, 30.0)[0] == -1),
                ("ambiguous (gate < d² < new_thr) → discard", associate(_mu, _Sig, _zmid, _Q, 9.21, 30.0)[0] == -2),
                ("d²_min = νᵀS⁻¹ν", np.isclose(associate(_mu, _Sig, _zmid, _Q, 9.21, 30.0)[1], 12.0, rtol=0.02)),
                ("bearing innovation wrapped", associate(np.array([0.0, 0.0, 0.0, -3.0, -0.05]), np.eye(5) * 0.01, (3.0, np.pi - 0.01), _Q, 9.21, 30.0)[0] == 0),
                ("input arrays unchanged", np.array_equal(_mu, _mu_before) and np.array_equal(_Sig, _Sig_before)),
            ]
        except Exception as _e:
            return f"⏳ not implemented yet ({type(_e).__name__})"
        return "<br>".join(f"{'✅' if _ok else '❌'} {_n}" for _n, _ok in _r)

    mo.md("**✅ CHECK B1**<br>" + _check())
    return


@app.cell(hide_code=True)
def _(
    add_landmark,
    align_map,
    landmark_groups,
    np,
    slam_predict,
    slam_update,
    time,
):
    # PROVIDED — EKF-SLAM with unknown correspondences, and evaluation of the associations with the barcodes.
    def run_ekf_slam_unknown(run, alphas, Q, gate=9.21, new_thr=30.0):
        if associate(np.zeros(3), np.eye(3), (1.0, 0.0), Q, gate, new_thr)[0] is None:
            return None
        _mu = run["x0"].copy()
        _Sigma = np.diag([1e-6] * 3)
        _t, _u = run["t0"], (0.0, 0.0)
        _labels = []                        # for every estimated landmark: barcodes of the measurements associated with it
        _states, _rcov = [[_t, *_mu]], [_Sigma[:3, :3].copy()]
        _info = {"updates": 0, "new landmarks": 0, "discarded": 0}
        _tic = time.perf_counter()
        for _e in run["events"]:
            if _e[0] > _t:
                _mu, _Sigma = slam_predict(_mu, _Sigma, _u[0], _u[1], _e[0] - _t, alphas)
                _t = _e[0]
            if _e[1] == 0:
                _u = (_e[2], _e[3])
                _states.append([_t, *_mu[:3]])
                _rcov.append(_Sigma[:3, :3].copy())
                continue
            _z = (_e[3], _e[4])
            _j, _ = associate(_mu, _Sigma, _z, Q, gate, new_thr)
            if _j == -1:
                _mu, _Sigma = add_landmark(_mu, _Sigma, _z, Q)
                _labels.append([_e[2]])
                _info["new landmarks"] += 1
            elif _j == -2:
                _info["discarded"] += 1
                continue
            else:
                _mu, _Sigma = slam_update(_mu, _Sigma, _z, _j, Q)[:2]
                _labels[_j].append(_e[2])
                _info["updates"] += 1
            _Sigma = 0.5 * (_Sigma + _Sigma.T)
            _states.append([_t, *_mu[:3]])
            _rcov.append(_Sigma[:3, :3].copy())
        _info["run time (s)"] = round(time.perf_counter() - _tic, 2)
        return dict(states=np.array(_states), robot_cov=np.array(_rcov), mu=_mu, Sigma=_Sigma, labels=_labels, info=_info)

    def association_metrics(run, res):
        """Post-run diagnostic only: barcodes have no role in association decisions. Barcode- and group-level counts.
        Each estimated landmark receives its deterministic majority barcode; ties use the smallest ID.
        The wrong-association count is an approximate majority-based diagnostic,
        not ground-truth correspondence during filtering."""
        _maj = [min(set(_l), key=lambda _b: (-_l.count(_b), int(_b))) for _l in res["labels"]]
        _n_assoc = sum(len(_l) for _l in res["labels"])
        _wrong = sum(sum(1 for _b in _l if _b != _m) for _l, _m in zip(res["labels"], _maj))
        _true = set(run["meas"][:, 1])
        _owned = set(_maj)
        # map error: for every true landmark, the estimate with most observations
        _best = {}
        for _k, (_l, _m) in enumerate(zip(res["labels"], _maj)):
            if _m not in _best or len(_l) > len(res["labels"][_best[_m]]):
                _best[_m] = _k
        _bcs = sorted(_best)
        _est = np.array([res["mu"][3 + 2 * _best[_b]: 5 + 2 * _best[_b]] for _b in _bcs])
        _ref = np.array([run["landmarks"][_b] for _b in _bcs])
        _R, _tr, _al, _ang = align_map(_est, _ref)
        # the same counts at the level of landmark GROUPS (landmarks closer than 0.5 m)
        _G = landmark_groups(run)
        _lab_g = [[_G[_b] for _b in _l] for _l in res["labels"]]
        _maj_g = [min(set(_l), key=lambda _q: (-_l.count(_q), _q)) for _l in _lab_g]
        _wrong_g = sum(sum(1 for _q in _l if _q != _m) for _l, _m in zip(_lab_g, _maj_g))
        _true_g = {_G[_b] for _b in _true}
        return {
            "estimated landmarks": len(_maj),
            "true landmarks observed": len(_true),
            "duplicates": len(_maj) - len(_owned),
            "missed or merged": len(_true - _owned),
            "% wrong": round(100 * _wrong / max(_n_assoc, 1), 2),
            "groups observed": len(_true_g),
            "group duplicates": len(_maj_g) - len(set(_maj_g)),
            "groups missed": len(_true_g - set(_maj_g)),
            "% wrong across groups": round(100 * _wrong_g / max(_n_assoc, 1), 2),
            "discarded": res["info"]["discarded"],
            "map RMSE (m)": round(float(np.sqrt(np.mean(np.sum((_est - _ref) ** 2, 1)))), 3),
            "map RMSE aligned (m)": round(float(np.sqrt(np.mean(np.sum((_al - _ref) ** 2, 1)))), 3),
        }

    def unknown_as_known(run, res):
        """Re-labels the result so that plot_slam can draw it (estimated landmark k → its majority barcode)."""
        _maj = [min(set(_l), key=lambda _b: (-_l.count(_b), int(_b))) for _l in res["labels"]]
        _r = dict(res)
        _r["ids"] = _maj
        return _r
    return association_metrics, run_ekf_slam_unknown, unknown_as_known


@app.cell
def _():
    # Thresholds for the interactive run below (squared Mahalanobis distance, chi-square with 2 dof)
    b_gate = 9.21    # associate if min d^2 <= b_gate (99 %)
    b_new = 30.0     # create a new landmark if min d^2 > b_new; discard in between
    return b_gate, b_new


@app.cell(hide_code=True)
def _(
    association_metrics,
    b_gate,
    b_new,
    cases,
    e2_Q,
    e2_alphas,
    evaluate,
    mo,
    pd,
    run_ekf_slam_unknown,
):
    if not (0 < b_gate <= b_new):
        raise ValueError("Require 0 < b_gate <= b_new (squared Mahalanobis thresholds).")
    _g, _n = b_gate, b_new
    ekf_unknown = {_k: run_ekf_slam_unknown(_r, e2_alphas, e2_Q, _g, _n) for _k, _r in cases.items()}
    mo.stop(ekf_unknown["A"] is None, mo.md("⏳ Implement TODO B1 first."))
    _rows = [{"case": _k, **{_kk: round(_v, 3) for _kk, _v in evaluate(_res["states"], cases[_k]["gt"], sample_times=cases[_k]["odom"][:, 0]).items()}, **association_metrics(cases[_k], _res), "run time (s)": _res["info"]["run time (s)"]}
             for _k, _res in ekf_unknown.items()]
    mo.vstack([mo.md(f"**B — unknown association** with γ_gate = {_g}, γ_new = {_n}"), mo.ui.table(pd.DataFrame(_rows), selection=None)])
    return (ekf_unknown,)


@app.cell(hide_code=True)
def _(cases, ekf_unknown, mo, plot_slam, unknown_as_known):
    _k = "A"
    (mo.md("") if ekf_unknown.get(_k) is None else
     plot_slam(cases[_k], unknown_as_known(cases[_k], ekf_unknown[_k]), title=f"EKF-SLAM, unknown association — case {_k} (labels = majority barcode; a repeated label is a duplicate)"))
    return


@app.cell(hide_code=True)
def _(cases, ekf_unknown, mo, plot_slam, unknown_as_known):
    _k = "B"
    (mo.md("") if ekf_unknown.get(_k) is None else
     plot_slam(cases[_k], unknown_as_known(cases[_k], ekf_unknown[_k]), title=f"EKF-SLAM, unknown association — case {_k} (labels = majority barcode; a repeated label is a duplicate)"))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### 🛠 TODO B2 — Effect of data-association thresholds

    The two thresholds determine how the filter handles uncertain measurements:

    - **`gamma_gate`:** maximum squared Mahalanobis distance for matching a measurement to an existing landmark. A smaller value makes the filter more selective; a larger value accepts more associations, including potentially incorrect ones.
    - **`gamma_new`:** minimum squared Mahalanobis distance above which a measurement is considered a new landmark. Increasing this threshold makes the filter more reluctant to create landmarks. Measurements between the two thresholds are discarded.

    The choice of thresholds affects the estimated trajectory **and** the map. For example, a filter may create duplicate landmarks, merge observations from different landmarks, or discard useful measurements.

    **Your task:** Investigate how these thresholds affect EKF-SLAM with unknown data association.

    #### Experimental procedure

    1. Choose **at least six different pairs** of thresholds, satisfying `0 < gamma_gate <= gamma_new`.
    2. Include the following configurations:
        - **No discard region:** `gamma_new = gamma_gate`. Every measurement is either associated with an existing landmark or treated as new. This may create duplicate landmarks.
        - **Intermediate discard region:** for example, `gamma_gate = 9.21`, `gamma_new = 30`.
        - **Very conservative landmark creation:** choose a much larger `gamma_new` (e.g. `200`). This makes new landmarks harder to create and can cause many measurements to be discarded.
    3. Run `run_ekf_slam_unknown` for **both cases A and B** using every threshold pair.
    4. For each run, calculate:
        - **ATE (m):** trajectory error using `evaluate`, with the odometry timestamps as the common evaluation times.
        - **Association metrics:** incorrect associations, duplicate or merged landmarks, and discarded measurements, using `association_metrics`.
    5. Store all results in `b2_rows`, with **one dictionary per case and threshold pair**. At least six pairs on two cases means **at least 12 rows**.

    Keep `e2_alphas`, `e2_Q` and the datasets unchanged throughout the experiment so that only the association thresholds vary.

    #### Provided functions and variables

    - `cases`: dictionary containing the data for cases `"A"` and `"B"`.
    - `e2_alphas`: your chosen motion-noise parameters.
    - `e2_Q`: your chosen range–bearing measurement covariance.
    - `run_ekf_slam_unknown(run, alphas, Q, gate, new_thr)`: runs EKF-SLAM with unknown correspondences and returns the filter result.
    - `evaluate(result["states"], run["gt"], sample_times=run["odom"][:, 0])`: computes the trajectory errors, including ATE.
    - `association_metrics(run, result)`: compares the estimated associations against the true landmark identities and returns a dictionary of association metrics.

    #### Expected output

    Create a list `b2_rows`. Each element must contain:

    - `"case"`: `"A"` or `"B"`.
    - `"gamma_gate"`: the gate threshold used.
    - `"gamma_new"`: the new-landmark threshold used.
    - `"ATE (m)"`: the trajectory ATE.
    - All fields returned by `association_metrics(run, result)`.

    The provided display cell will show your results as a table.

    The filter is deterministic, so **one run per case and threshold pair is sufficient**. Set `RUN_B2 = True` to execute the experiment.

    **Compare both trajectory and association metrics:** a lower ATE does not necessarily mean that the filter has built a better map.
    """
    )
    return


@app.cell
def _():
    RUN_B2 = False  # set True for the final run and the HTML export
    return (RUN_B2,)


@app.cell
def _(RUN_B2, mo, pd):
    mo.stop(not RUN_B2, mo.md("B2 experiment paused: complete the loop and set RUN_B2 = True."))
    b2_rows = []
    # TODO B2: at least six distinct (gate, new_thr) pairs, run on case A and B.
    # For each result append {"case": ..., "gamma_gate": ..., "gamma_new": ...,
    #                         "ATE (m)": ..., **association_metrics(...)}.
    # Include a no-discard setting and a conservative-new-landmark setting.
    mo.ui.table(pd.DataFrame(b2_rows), selection=None) if b2_rows else mo.md("Complete TODO B2 to show results.")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    #### ✍️ ANSWERS — Part B

    **B1** Using your B2 table: what happens at each extreme ($\gamma_{\text{new}}=\gamma_{\text{gate}}$ small; $\gamma_{\text{new}}$ very large)? Which kind of error dominates: duplicate landmarks, wrong associations or discarded information? How does each kind of error affect the trajectory and the map, and why is a discard band between $\gamma_{\text{gate}}$ and $\gamma_{\text{new}}$ useful?

    > *Your answer…*

    **B2** Is there a single setting that works well for both cases? State what "works well" means for you (which metrics) and report it. Compare your best unknown-association result with the known-correspondence baseline of Part A, in each case.

    > *Your answer…*

    **B3** (1) In case A, how do barcode-level and group-level wrong-association rates differ? Explain the difference using the close landmark groups. Does either metric fully describe association quality? (2) Describe one association failure that remains at group level, or one duplicate, in your runs (what, where, when; use the plots). (3) Explain how **joint compatibility** (JCBB) or a **provisional-landmark** scheme (a candidate becomes a landmark only after $k$ consistent sightings) would address it. (4) Why is association harder with unlabelled visual features than with barcoded landmarks?

    > *Your answer…*
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r""" """)
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ---
    ## Part C — FastSLAM 1.0 with known correspondences (30 %)

    In EKF-SLAM we estimated the robot pose and all landmark positions together, with one large state vector and one covariance matrix.

    **FastSLAM uses a different representation.** It keeps $M$ particles; each particle is one possible **robot trajectory** (its current pose is the end of that trajectory) and carries **its own map**: one small Gaussian per landmark. With 200 particles we have 200 trajectory hypotheses and 200 maps, so the same landmark can have slightly different estimates in different particles. Given the trajectory of a particle, its landmarks are independent of each other, which is why each one needs only a $2\times2$ covariance instead of being part of a large joint robot–map covariance.

    In this part the landmark identities are known from their barcodes. The provided FastSLAM driver handles motion sampling, particle weights, resampling and the selection of the landmark to update. **Your task is to implement how one landmark is initialised or updated in every particle.**

    ### Landmark estimates inside each particle

    For the landmark being processed, particle $k$ stores its robot pose $P[k]=(x,y,\theta)$, its estimate of the landmark position $m[k]=(m_x,m_y)$ and the $2\times2$ covariance $C[k]$ of that estimate. The update is done **separately for each particle, treating that particle's pose as known**.

    ### Case 1 — first observation (`new = True`)

    **Initial position.** Place the landmark with the particle pose and the measurement $z=(r,\phi)$:

    $$
    m_x = x + r\cos(\theta+\phi), \qquad m_y = y + r\sin(\theta+\phi).
    $$

    **Initial covariance.** Propagate the measurement noise through the Jacobian of these equations with respect to $z$:

    $$
    G_z=\begin{bmatrix}\cos(\theta+\phi) & -r\sin(\theta+\phi)\\ \sin(\theta+\phi) & r\cos(\theta+\phi)\end{bmatrix},
    \qquad C_{\text{new}} = G_z\,Q\,G_z^\top .
    $$

    **Weight.** The log-weight increment is **zero** for every particle: each particle builds its landmark from the same measurement, so a first observation does not yet favour one particle over another.

    ### Case 2 — re-observation (`new = False`)

    For each particle $k$:

    **Innovation.** Predict the range and bearing from $P[k]$ to $m[k]$ and subtract it from the measurement, wrapping the bearing to $[-\pi,\pi)$:

    $$
    \nu = z - h(P[k], m[k]).
    $$

    **Jacobian.** $H_m$ is the $2\times2$ Jacobian of the range–bearing model with respect to the **landmark position** $(m_x, m_y)$.

    **Innovation covariance and gain.**

    $$
    S = H_m\,C[k]\,H_m^\top + Q, \qquad K = C[k]\,H_m^\top S^{-1}.
    $$

    **Correction** (covariance in Joseph form, which stays symmetric and positive semidefinite):

    $$
    m_{\text{new}} = m[k] + K\nu, \qquad
    C_{\text{new}} = (I-KH_m)\,C[k]\,(I-KH_m)^\top + K Q K^\top .
    $$

    **Log-likelihood of the measurement** for this particle (the log of a 2-D Gaussian density):

    $$
    \Delta\log w_k = -\tfrac12\,\nu^\top S^{-1}\nu - \tfrac12\log\det S - \log(2\pi).
    $$

    It measures how well the particle predicts the actual measurement: particles that predict it better gain relative weight.

    **Important.** This function updates the landmark estimates, not the particle poses. The driver adds the returned increments to the particle log-weights and resamples when needed.

    ### 🛠 TODO C1 — `fastslam_landmark_update(P, m, C, z, Q, new)`

    **Inputs**

    - `P`: array `(M, 3)`, one robot pose `[x, y, theta]` (m, m, rad) per particle.
    - `m`: array `(M, 2)`, particle `k`'s estimate `[mx, my]` (m) of **the landmark being processed** (not the whole map).
    - `C`: array `(M, 2, 2)`, the covariance of each `m[k]`.
    - `z`: measurement `[range, bearing]`, shape `(2,)`, in m and rad; the same for all particles.
    - `Q`: measurement-noise covariance, shape `(2, 2)`.
    - `new`: `True` for the first observation of this landmark, `False` for a re-observation.

    **Returns** the tuple `(m_new, C_new, logw)`:

    - `m_new`: array `(M, 2)`, initialised or corrected landmark position of each particle.
    - `C_new`: array `(M, 2, 2)`, the corresponding covariances.
    - `logw`: array `(M,)`, one log-likelihood increment per particle; `np.zeros(M)` when `new = True`.

    **Requirements**

    - Process each particle with its own pose, landmark mean and covariance.
    - Wrap bearing differences to $[-\pi,\pi)$.
    - Return symmetric covariance matrices.
    - Return new arrays; do not modify `P`, `m` or `C` in place.
    - Do not move the particles, resample or normalise weights here: the driver does it.
    - A Python loop over particles is accepted; vectorising with NumPy is optional.

    **CHECK C1** tests the initialisation and the correction, including the covariance update and the log-likelihood.
    """
    )
    return


@app.function
def fastslam_landmark_update(P, m, C, z, Q, new):
    # TODO C1: per-particle 2D landmark EKF + particle log-likelihood increments.
    return None, None, None


@app.cell(hide_code=True)
def _(add_landmark, mo, np, slam_update):
    def _check():
        try:
            _rng = np.random.default_rng(4)
            _M = 6
            _P = np.column_stack([_rng.normal(0, 0.3, _M), _rng.normal(0, 0.3, _M), _rng.normal(0.2, 0.2, _M)])
            _Q = np.diag([0.1, 0.05]) ** 2
            _z = (2.5, 0.4)
            _m0, _C0, _l0 = fastslam_landmark_update(_P, np.zeros((_M, 2)), np.zeros((_M, 2, 2)), _z, _Q, True)
            _mi, _Ci = _m0 + 0.05, _C0 * 1.5
            _P_before, _mi_before, _Ci_before = _P.copy(), _mi.copy(), _Ci.copy()
            _m1, _C1, _l1 = fastslam_landmark_update(_P, _mi, _Ci, (2.45, 0.38), _Q, False)
            _ok_new, _ok_upd, _ok_w = True, True, True
            for _k in range(_M):
                # reference: your L4 functions with a pose known exactly (zero robot covariance)
                _ma, _Sa = add_landmark(_P[_k], np.zeros((3, 3)), _z, _Q)
                _ok_new &= np.allclose(_m0[_k], _ma[3:]) and np.allclose(_C0[_k], _Sa[3:, 3:])
                _Sig = np.zeros((5, 5))
                _Sig[3:, 3:] = _C0[_k] * 1.5
                _mu = np.concatenate([_P[_k], _m0[_k] + 0.05])
                _mb, _Sb, _nu, _S, _ = slam_update(_mu, _Sig, (2.45, 0.38), 0, _Q)
                _ok_upd &= np.allclose(_m1[_k], _mb[3:]) and np.allclose(_C1[_k], _Sb[3:, 3:], atol=1e-10)
                _ok_w &= np.isclose(_l1[_k], -0.5 * _nu @ np.linalg.solve(_S, _nu) - 0.5 * np.log(np.linalg.det(2 * np.pi * _S)))
            _r = [
                ("shapes (M,2), (M,2,2), (M,)", np.shape(_m1) == (_M, 2) and np.shape(_C1) == (_M, 2, 2) and np.shape(_l1) == (_M,)),
                ("new landmark: inverse model and G_z Q G_zᵀ (= add_landmark with a known pose)", bool(_ok_new) and np.allclose(_l0, np.zeros(_M))),
                ("update = slam_update of L4 with a known pose (landmark part)", bool(_ok_upd)),
                ("log-weight = log N(ν; 0, S)", bool(_ok_w)),
                ("no input mutations", np.array_equal(_P, _P_before) and np.array_equal(_mi, _mi_before) and np.array_equal(_Ci, _Ci_before)),
                ("posterior C symmetric and positive semidefinite", np.allclose(_C1, np.transpose(_C1, (0, 2, 1)), atol=1e-10) and np.all(np.linalg.eigvalsh(_C1) >= -1e-10)),
            ]
        except Exception as _e:
            return f"⏳ not implemented yet ({type(_e).__name__})"
        return "<br>".join(f"{'✅' if _ok else '❌'} {_n}" for _n, _ok in _r)

    mo.md("**✅ CHECK C1**<br>" + _check())
    return


@app.cell(hide_code=True)
def _(
    align_map,
    effective_sample_size,
    evaluate,
    landmark_label,
    low_variance_resample,
    map_table,
    nees_position,
    normalize_log_weights,
    np,
    sample_motion_model,
    time,
):
    # PROVIDED — FastSLAM 1.0 driver (known correspondences) and its evaluation.
    def run_fastslam(run, M, alphas, Q, seed=0, neff_frac=0.5, blackout=None, checkpoint_every=10.0):
        """Returns a dict compatible with plot_slam/map_table (mu, Sigma, ids: weighted mean map) plus
        states (weighted mean pose), pos_cov (weighted particle covariance), ancestry and info."""
        if fastslam_landmark_update(np.zeros((2, 3)), np.zeros((2, 2)), np.zeros((2, 2, 2)), (1.0, 0.0), Q, True)[0] is None:
            return None
        _rng = np.random.default_rng(seed)
        _P = np.tile(run["x0"], (M, 1))
        _logw = np.zeros(M)
        _bcs = sorted(set(run["meas"][:, 1]))
        _slot = {_b: _i for _i, _b in enumerate(_bcs)}
        _m = np.zeros((M, len(_bcs), 2))
        _C = np.zeros((M, len(_bcs), 2, 2))
        _seen = np.zeros(len(_bcs), bool)
        _first, _nobs, _order = {}, {}, []
        _anc, _next_ck = {}, 0.0
        _t, _u = run["t0"], (0.0, 0.0)
        _states, _pcov = [], []
        _nres, _neffs = 0, []
        _tic = time.perf_counter()

        def _record():
            _w = normalize_log_weights(_logw)
            _mx, _my = _w @ _P[:, 0], _w @ _P[:, 1]
            _th = np.arctan2(_w @ np.sin(_P[:, 2]), _w @ np.cos(_P[:, 2]))
            _d = _P[:, :2] - [_mx, _my]
            _states.append([_t, _mx, _my, _th])
            _pcov.append((_w[:, None] * _d).T @ _d)

        _record()
        for _e in run["events"]:
            if _e[0] > _t:
                _P = sample_motion_model(_P, _u[0], _u[1], _e[0] - _t, alphas, _rng)
                _t = _e[0]
            while _t - run["t0"] >= _next_ck:
                _anc[_next_ck] = np.arange(M)
                _next_ck += checkpoint_every
            if _e[1] == 0:
                _u = (_e[2], _e[3])
                # Every odometry timestamp is saved for fair E1-style metrics.
                # Measurement-time states are also retained immediately after each update.
                _record()
                continue
            if blackout is not None and blackout[0] <= _t - run["t0"] < blackout[1]:
                continue
            _b = _e[2]
            _j = _slot[_b]
            if not _seen[_j]:
                _first[_b] = {"t": _t - run["t0"], "robot std (m)": float(np.sqrt(np.trace(_pcov[-1])))}
                _order.append(_b)
            _m[:, _j], _C[:, _j], _lw = fastslam_landmark_update(_P, _m[:, _j], _C[:, _j], (_e[3], _e[4]), Q, not _seen[_j])
            _seen[_j] = True
            _nobs[_b] = _nobs.get(_b, 0) + 1
            _logw = _logw + _lw
            _w = normalize_log_weights(_logw)
            _ne = effective_sample_size(_w)
            _neffs.append(_ne)
            _record()
            if _ne < neff_frac * M:
                _idx = low_variance_resample(np.arange(M), _w, _rng)
                _P, _m, _C = _P[_idx], _m[_idx], _C[_idx]
                _logw = np.zeros(M)
                for _k in _anc:
                    _anc[_k] = _anc[_k][_idx]
                _nres += 1
        _w = normalize_log_weights(_logw)
        # Weighted mean map and MARGINAL (2x2) landmark mixture covariances for plotting.
        # This is NOT a full FastSLAM joint covariance: omitted off-diagonal blocks
        # must not be interpreted as independence in the unconditional posterior.
        _mu = list(_states[-1][1:])
        _blocks = []
        for _b in _order:
            _j = _slot[_b]
            _mean = _w @ _m[:, _j]
            _d = _m[:, _j] - _mean
            _blocks.append(np.einsum("k,kij->ij", _w, _C[:, _j]) + (_w[:, None] * _d).T @ _d)
            _mu += list(_mean)
        _n = len(_mu)
        _Sigma = np.zeros((_n, _n))
        _Sigma[:2, :2] = _pcov[-1]
        for _i, _B in enumerate(_blocks):
            _Sigma[3 + 2 * _i: 5 + 2 * _i, 3 + 2 * _i: 5 + 2 * _i] = _B
        _info = {"resamplings": _nres, "mean N_eff / M": round(float(np.mean(_neffs)) / M, 2), "run time (s)": round(time.perf_counter() - _tic, 2)}
        return dict(states=np.array(_states), pos_cov=np.array(_pcov), mu=np.array(_mu), Sigma=_Sigma, ids=_order,
                    first_seen=_first, n_obs=_nobs, info=_info,
                    ancestry=[(float(_k), int(len(np.unique(_v)))) for _k, _v in sorted(_anc.items())], M=M)

    def fastslam_summary(run, res):
        _df, _sm = map_table(run, res, align_map, landmark_label)
        _nees = nees_position(res["states"], res["pos_cov"], run["gt"], sample_times=run["odom"][:, 0])
        _anc = dict(res["ancestry"])
        _tend = run["odom"][-1, 0] - run["t0"]
        _old = [_v for _k, _v in _anc.items() if _k <= _tend - 60]
        return {**{_k: round(_v, 3) for _k, _v in evaluate(res["states"], run["gt"], sample_times=run["odom"][:, 0]).items()},
                "map RMSE (m)": round(_sm["map RMSE (m)"], 3), "map RMSE aligned (m)": round(_sm["map RMSE after rigid alignment (m)"], 3),
                "|map rotation| (deg)": round(abs(_sm["alignment rotation (deg)"]), 1),
                "median position NEES": round(float(np.median(_nees)), 1), "% NEES < 11.83": round(100 * float(np.mean(_nees < 11.83)), 1),
                "ancestors of the set 60 s before the end": _old[-1] if _old else None,
                **res["info"]}
    return fastslam_summary, run_fastslam


@app.cell
def _():
    c_M = 100          # particles of the demonstration run
    c_seed = 0
    RUN_C_DEMO = False  # set True for the final export
    return RUN_C_DEMO, c_M, c_seed


@app.cell(hide_code=True)
def _(
    RUN_C_DEMO,
    c_M,
    c_seed,
    cases,
    e2_Q,
    e2_alphas,
    ekf_known,
    evaluate,
    fastslam_summary,
    mo,
    pd,
    run_fastslam,
):
    mo.stop(not RUN_C_DEMO, mo.md("FastSLAM demonstration paused: set RUN_C_DEMO = True."))
    fs_res = {_k: run_fastslam(_r, c_M, e2_alphas, e2_Q, seed=c_seed) for _k, _r in cases.items()}
    mo.stop(fs_res["A"] is None, mo.md("⏳ Implement TODO C1 first."))
    _rows = []
    for _k, _r in cases.items():
        _rows.append({"case": _k, "method": f"FastSLAM M={c_M}, seed {c_seed}", **fastslam_summary(_r, fs_res[_k])})
        if ekf_known.get(_k) is not None:
            _rows.append({"case": _k, "method": "EKF-SLAM (Part A)", **{_kk: round(_v, 3) for _kk, _v in evaluate(ekf_known[_k]["states"], _r["gt"], sample_times=_r["odom"][:, 0]).items()}})
    mo.vstack([mo.md("**C — FastSLAM demonstration run**"), mo.ui.table(pd.DataFrame(_rows), selection=None)])
    return (fs_res,)


@app.cell(hide_code=True)
def _(cases, fs_res, mo, plot_slam):
    _k = "A"
    mo.md("") if fs_res.get(_k) is None else plot_slam(cases[_k], fs_res[_k], title=f"FastSLAM 1.0 — case {_k} (weighted mean map, marginal mixture 3σ)")
    return


@app.cell(hide_code=True)
def _(cases, fs_res, mo, plot_slam):
    _k = "B"
    mo.md("") if fs_res.get(_k) is None else plot_slam(cases[_k], fs_res[_k], title=f"FastSLAM 1.0 — case {_k} (weighted mean map, marginal mixture 3σ)")
    return


@app.cell(hide_code=True)
def _(cases, ekf_known, fs_res, go, mo, plot_errors):
    _k = "A"
    mo.stop(fs_res.get(_k) is None)
    _fig = plot_errors(cases[_k], {"FastSLAM": fs_res[_k]["states"], "EKF-SLAM": ekf_known[_k]["states"] if ekf_known.get(_k) is not None else None})
    _anc = go.Figure()
    for _kk, _res in fs_res.items():
        _anc.add_trace(go.Scatter(x=[_a for _a, _ in _res["ancestry"]], y=[_b for _, _b in _res["ancestry"]], mode="lines+markers", name=f"case {_kk}"))
    _anc.update_layout(title="Distinct ancestors, at the end of the run, of the particle set at time t", xaxis_title="t (s)", yaxis_title="distinct ancestors", yaxis_type="log", height=320, plot_bgcolor="white")
    mo.vstack([_fig, _anc])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### 🛠 TODO C2 — Effect of the number of particles

    FastSLAM represents different possible robot trajectories and maps using particles. Increasing the number of particles \(M\) allows the filter to represent more hypotheses, but also increases the computational cost.

    Unlike EKF-SLAM, **FastSLAM is stochastic**: it samples robot motions and resamples particles. Therefore, two runs with the same parameters may produce different trajectories and maps.

    In this experiment, we will investigate:

    - Does increasing \(M\) improve the estimated robot trajectory and map?
    - Does using more particles make the results more stable across runs?
    - How much does the computational cost increase?

    ### Comparing multiple runs

    To compare different particle counts fairly, run FastSLAM several times using different random seeds.

    For each metric, report:

    - **Median:** the middle value of the results, representing typical performance.
    - **Interquartile range (IQR):** the difference between the 75th percentile (Q3) and the 25th percentile (Q1), $\mathrm{IQR}=Q_3-Q_1$.

    A smaller IQR indicates less variation among the central 50% of the runs.

    For example, if five runs produce ATE values of `0.4, 0.5, 0.6, 0.8, 1.0 m`, the median is `0.6 m` and the IQR is `0.3 m` (using pandas' default percentile calculation).

    ### Experimental procedure

    1. Choose **at least three particle counts**, including one with \(M\geq500\). For example: `M = [50, 200, 600]`.

    2. For each particle count, run FastSLAM with **at least five different random seeds**. For example: `seed = 0, 1, 2, 3, 4`.

    3. Repeat the experiment for **both cases A and B**. With three particle counts, five seeds and two cases, this gives **30 runs**.

    4. For each run:
        - Execute `run_fastslam` using the selected particle count and seed.
        - Evaluate the result using `fastslam_summary`.
        - Save the case, particle count, seed and evaluation metrics in `c2_runs`.

    5. Group the results by **case and particle count**. For each group, calculate the median and IQR of:
        - **ATE (m):** robot trajectory error over the complete run.
        - **Aligned map RMSE (m):** landmark position error after rigid map alignment.
        - **Absolute map rotation (deg):** magnitude of the rotation needed to align the estimated map with the reference map.
        - **Run time (s):** time required to execute FastSLAM.

    Keep `e2_alphas`, `e2_Q`, the datasets and all other filter settings unchanged. Only vary the particle count and random seed.

    ### Provided functions and variables

    - `cases`: dictionary containing cases `"A"` and `"B"`. Each value is a `run` dictionary with the dataset for that case.
    - `e2_alphas`: the four motion-noise parameters used in Part A.
    - `e2_Q`: the range–bearing measurement-noise covariance used in Part A.
    - `run_fastslam(run, M, alphas, Q, seed)`: executes one complete FastSLAM run and returns its estimated trajectory, map and diagnostic information.
    - `fastslam_summary(run, result)`: evaluates a completed FastSLAM run against the reference trajectory and landmark positions. It returns a dictionary containing trajectory, map, consistency and computational metrics.
    - `pd`: pandas, used to organise and summarise your results.

    ### Expected output

    **1. `c2_runs` — individual runs**

    Create a list of dictionaries, with one dictionary per experiment. Each dictionary must contain:

    - `"case"`: `"A"` or `"B"`.
    - `"M"`: number of particles used.
    - `"seed"`: random seed used.
    - All metrics returned by `fastslam_summary(run, result)`.

    With the minimum required configurations, `c2_runs` will contain **at least 30 dictionaries**.

    **2. `c2_table` — summary table**

    Create a pandas DataFrame with **one row for each `(case, M)` pair** and the median and IQR of the four metrics listed above.

    For example, with three particle counts and two cases, `c2_table` will contain **six rows**.

    Present each result as `median [IQR]`, clearly identifying the corresponding metric and units. The provided display will show the table.

    Set `RUN_C2 = True` to execute your final experiment.

    **Important:** Do not select only the best seed or change the noise parameters between configurations. Compare the complete set of runs, including unsuccessful ones.
    """
    )
    return


@app.cell
def _():
    RUN_C2 = False  # set True for the final run and the HTML export (several minutes)
    return (RUN_C2,)


@app.cell
def _(RUN_C2, mo, pd):
    mo.stop(not RUN_C2, mo.md("C2 experiment paused: complete the loop and set RUN_C2 = True."))
    c2_runs = []
    # TODO C2: >=3 particle counts (one >=500), >=5 independent seeds, BOTH cases.
    # Append {"case": ..., "M": ..., "seed": ..., **fastslam_summary(...)}.
    # Build one row per (case, M) with median and interquartile range (Q3-Q1)
    # for ATE, aligned map RMSE, absolute map rotation and runtime.
    c2_table = pd.DataFrame()
    mo.ui.table(c2_table, selection=None) if not c2_table.empty else mo.md("Complete TODO C2 to show results.")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    #### ✍️ ANSWERS — Part C

    **C1 — FastSLAM representation and computational cost**

    **C1.1** In FastSLAM, each particle maintains its own landmark estimates. Why can each landmark be updated separately within a particle? How is this different from EKF-SLAM, where the robot and landmarks share a single covariance matrix?

    > *Your answer…*

    **C1.2** When a measurement of a known landmark arrives, FastSLAM updates that landmark in every particle. Why does the computation time increase when we use more particles? What additional work is required when particles are resampled?

    Compare the execution times of FastSLAM and EKF-SLAM from your experiments. Which method was faster in your cases?

    > *Your answer…*

    ---

    **C2 — Effect of the number of particles**

    **C2.1** Using your C2 results, does increasing the number of particles improve the trajectory and map estimates? Compare the median ATE, aligned map RMSE and map rotation for both cases. Is the improvement consistent?

    > *Your answer…*

    **C2.2** How does the IQR change when you increase the number of particles? What does this tell you about the variability of FastSLAM between different runs? How does the execution time change?

    > *Your answer…*

    **C2.3** Compare FastSLAM with EKF-SLAM from Part A. Which gives better trajectory estimates in each case? Use the trajectory-error plots to identify where the largest differences occur.

    Why is it important to run FastSLAM with several random seeds instead of reporting only one execution?

    > *Your answer…*

    ---

    **C3 — Particle diversity and resampling**

    **C3.1** Look at the ancestry plots for both cases. How many different ancestors do the final particles have approximately 60 seconds before the end of the run? What happens to the number of distinct ancestors as you look further back in time?

    > *Your answer…*

    **C3.2** Suppose that all current particles share the same ancestor at an earlier time. What does this mean about the diversity of the past trajectories represented by the filter?

    Why could losing this diversity be a problem for landmarks that were mapped a long time ago?

    > *Your answer…*
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ---
    ## Part D — Consistency after a long observation gap (20 %)

    Case A has a first gap of about 48 s without landmark measurements (≈ 9–57 s). The provided experiment runs, on case A only:

    | run | α | Q | measurements |
    |---|---|---|---|
    | **baseline** | your α | your Q | real |
    | **k·α** | $k\,\alpha$ | your Q | real |
    | **alternative Q** | your α | $\operatorname{diag}(0.10^2,\,0.05^2)$ | real |
    | **synthetic** | your α | your Q | after 50 s every measurement is replaced by $h(x_\text{gt},m_\text{gt})+\varepsilon$, $\varepsilon\sim\mathcal N(0,Q)$ (`d_seeds` seeds) |
    | **FastSLAM** | your α and $k\,\alpha$ | your Q | real (`d_M` particles, `d_seeds` seeds each) |

    The synthetic run is a **simulation**: same trajectory, same landmarks and same measurement times, but synthetic independent, zero-mean Gaussian measurement noise with covariance $Q$ (while the motion and linearisation models may still be imperfect). Comparing it with the baseline separates what the **estimator** does from what the **real sensor errors** do.

    The figure shows the EKF heading error with the ±3σ band of the baseline and of the synthetic runs (gaps shaded), and the NIS $\nu^\top S^{-1}\nu$ of the updates of the landmarks mapped **before** the first gap ("old" landmarks), with the χ²₂ 99 % line (9.21). A large NIS means the filter finds an innovation surprising under its own model; by itself it does not identify the cause. The first table gives the heading error and its 3σ at four key instants; the second, the metrics of every run.

    No new code: choose $k$, $M$ and the number of seeds in the parameter cell and set `RUN_D = True`.
    """
    )
    return


@app.cell(hide_code=True)
def _(cases, landmark_label, long_gap_reobservations, mo, pd):
    # REFERENCE-ONLY: barcode identities are used here solely to analyse repeated sightings.
    _rows = []
    for _case, _run in cases.items():
        _events = long_gap_reobservations(_run, min_gap=30.0)
        if not _events:
            _rows.append({"case": _case, "landmark": "—", "reobserved at (s)": None,
                          "time since previous sighting (s)": None})
        for _event in _events:
            _rows.append({"case": _case, "landmark": landmark_label(_run, _event["barcode"]),
                          "reobserved at (s)": round(_event["time_s"], 1),
                          "time since previous sighting (s)": round(_event["gap_s"], 1)})
    mo.vstack([
        mo.md("**Long-gap landmark reobservations (reference diagnostic).** Each row is a sighting of a previously observed landmark after **at least 30 s without seeing that same landmark**. The interval is per landmark, not necessarily a period without *any* observations. This is a candidate link to older map information, **not proof of a spatial loop closure**. Barcode identities are used only for this analysis, never by the association filter in Part B."),
        mo.ui.table(pd.DataFrame(_rows), selection=None, page_size=15),
    ])
    return


@app.cell
def _():
    d_k = 5.0       # inflation factor of the process noise (k * alpha)
    d_M = 1000      # FastSLAM particles
    d_seeds = 5     # seeds of the synthetic EKF runs and of each FastSLAM setting (at least 5)
    RUN_D = False   # set True for the final export
    return RUN_D, d_M, d_k, d_seeds


@app.cell(hide_code=True)
def _():
    return


@app.cell(hide_code=True)
def _(
    RUN_D,
    align_map,
    cases,
    d_M,
    d_k,
    d_seeds,
    e2_Q,
    e2_alphas,
    evaluate,
    fastslam_summary,
    go,
    gt_at,
    h_landmark,
    landmark_label,
    make_subplots,
    map_table,
    measurement_gaps,
    mo,
    nees_position,
    np,
    pd,
    run_ekf_slam,
    run_fastslam,
):
    mo.stop(not RUN_D, mo.md("Part D paused: set RUN_D = True."))
    _r = cases["A"]
    mo.stop(run_ekf_slam(_r, e2_alphas, e2_Q) is None or run_fastslam(_r, 2, e2_alphas, e2_Q) is None,
            mo.md("⏳ Complete TODO 0 and C1 first."))
    _gaps = measurement_gaps(_r, 20.0)
    _k_alpha = tuple(d_k * _a for _a in e2_alphas)
    _Q_alt = np.diag([0.10**2, 0.05**2])

    def _synthetic(run, Q, seed, t_from=50.0):
        """Copy of `run` whose measurements after t_from seconds are regenerated from ground truth plus N(0, Q) noise."""
        _rng = np.random.default_rng(seed)
        _ev = []
        for _e in run["events"]:
            if _e[1] == 1 and _e[0] - run["t0"] > t_from:
                _z = h_landmark(gt_at(run["gt"], np.array([_e[0]]))[0], run["landmarks"][_e[2]]) + _rng.multivariate_normal(np.zeros(2), Q)
                _ev.append((_e[0], 1, _e[2], float(_z[0]), float(_z[1])))
            else:
                _ev.append(_e)
        return {**run, "events": _ev}

    _ekf = {"baseline": run_ekf_slam(_r, e2_alphas, e2_Q),
            f"{d_k:g}·α": run_ekf_slam(_r, _k_alpha, e2_Q),
            "alternative Q": run_ekf_slam(_r, e2_alphas, _Q_alt)}
    _synth = [run_ekf_slam(_synthetic(_r, e2_Q, _s), e2_alphas, e2_Q) for _s in range(d_seeds)]

    # heading error and 3-sigma on a common grid
    _T = _r["odom"][::10, 0]

    def _heading(res):
        _i = np.searchsorted(res["states"][:, 0], _T, side="right") - 1
        _e = np.degrees((res["states"][_i, 3] - gt_at(_r["gt"], _T)[:, 2] + np.pi) % (2 * np.pi) - np.pi)
        return _e, 3 * np.degrees(np.sqrt(np.maximum(res["robot_cov"][_i, 2, 2], 0.0)))

    _H = {_n: _heading(_res) for _n, _res in _ekf.items()}
    _Hs = np.array([_heading(_res)[0] for _res in _synth])
    _S3s = np.array([_heading(_res)[1] for _res in _synth])
    _x = _T - _r["t0"]

    # key instants: end of the first gap, 10 s later, just before the first re-observation of an old landmark, end
    _t_gap0, _t_gap1 = (_gaps[0] if _gaps else (np.inf, np.inf))
    _old = {_b for _b, _f in _ekf["baseline"]["first_seen"].items() if _f["t"] < _t_gap0}
    _t_old = min([_t for _t, _b, _ in _ekf["baseline"]["nis"] if _b in _old and _t > _t_gap1], default=np.nan)
    _instants = {"end of first gap": _t_gap1, "10 s after the gap": _t_gap1 + 10, "before old landmarks reappear": _t_old - 1, "end of run": _x[-1]}
    _idx = {_n: int(np.argmin(np.abs(_x - _v))) if np.isfinite(_v) else 0 for _n, _v in _instants.items()}
    _key = []
    for _n, (_e, _s) in _H.items():
        _key.append({"run": _n, **{f"{_kk} ({_x[_i]:.0f} s)": f"{int(round(_e[_i])):+d}° / ±{_s[_i]:.0f}°" for _kk, _i in _idx.items()}})
    _key.append({"run": f"synthetic (mean ± std of {d_seeds} seeds)",
                 **{f"{_kk} ({_x[_i]:.0f} s)": f"{int(round(_Hs[:, _i].mean())):+d}° ± {_Hs[:, _i].std():.0f}° / ±{_S3s[:, _i].mean():.0f}°" for _kk, _i in _idx.items()}})

    # NIS of the old landmarks during the first 10 s after they reappear (median / max)
    def _nis_old(results):
        _v = [_val for _res in results for _t, _b, _val in _res["nis"] if _b in _old and _t_old <= _t <= _t_old + 10.0]
        return f"{np.median(_v):.1f} / {np.max(_v):.1f}" if _v else "—"

    _nis_col = f"NIS old landmarks {_t_old:.0f}–{_t_old + 10:.0f} s (median / max)"
    for _row, _results in zip(_key, [[_res] for _res in _ekf.values()] + [_synth]):
        _row[_nis_col] = _nis_old(_results)

    _fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                         subplot_titles=("EKF-SLAM heading error (deg), ±3σ of baseline (blue) and synthetic (green)", "NIS of the updates of old landmarks (log scale)"))
    _col = {"baseline": "#1f77b4", f"{d_k:g}·α": "#d62728", "alternative Q": "#ff7f0e"}
    _e0, _s0 = _H["baseline"]
    _fig.add_trace(go.Scatter(x=np.concatenate([_x, _x[::-1]]), y=np.concatenate([_s0, -_s0[::-1]]), fill="toself", mode="none", fillcolor="rgba(31,119,180,0.15)", name="±3σ baseline"), row=1, col=1)
    _sm = _S3s.mean(0)
    _fig.add_trace(go.Scatter(x=_x, y=_sm, mode="lines", line=dict(color="#2ca02c", dash="dot", width=1), name="±3σ synthetic"), row=1, col=1)
    _fig.add_trace(go.Scatter(x=_x, y=-_sm, mode="lines", line=dict(color="#2ca02c", dash="dot", width=1), showlegend=False), row=1, col=1)
    for _n, (_e, _s) in _H.items():
        _fig.add_trace(go.Scatter(x=_x, y=_e, mode="lines", name=f"error, {_n}", line=dict(color=_col[_n])), row=1, col=1)
    _fig.add_trace(go.Scatter(x=_x, y=_Hs.mean(0), mode="lines", name="error, synthetic (mean)", line=dict(color="#2ca02c", dash="dash")), row=1, col=1)
    for _n, _res, _c in (("baseline", _ekf["baseline"], _col["baseline"]),
                         ("alternative Q", _ekf["alternative Q"], _col["alternative Q"]),
                         ("synthetic, seed 0", _synth[0], "#2ca02c")):
        _nis = np.array([(_t, _v) for _t, _b, _v in _res["nis"] if _b in _old]).reshape(-1, 2)
        _fig.add_trace(go.Scatter(x=_nis[:, 0], y=_nis[:, 1], mode="markers", marker=dict(size=4, color=_c), name=f"NIS old landmarks, {_n}"), row=2, col=1)
    for _a, _b in _gaps:
        for _row in (1, 2):
            _fig.add_vrect(x0=_a, x1=_b, fillcolor="gray", opacity=0.15, line_width=0, row=_row, col=1)
    _fig.add_hline(y=9.21, line_dash="dash", row=2, col=1)
    _fig.update_yaxes(type="log", row=2, col=1)
    _fig.update_xaxes(title_text="time since start (s)", row=2, col=1)
    _fig.update_layout(height=680, plot_bgcolor="white")

    def _ekf_row(name, res, seed=None):
        _df, _sm2 = map_table(_r, res, align_map, landmark_label)
        _nees = nees_position(res["states"], res["robot_cov"][:, :2, :2], _r["gt"], sample_times=_r["odom"][:, 0])
        return {"method": f"EKF-SLAM {name}", "seed": seed,
                **{_kk: round(_v, 3) for _kk, _v in evaluate(res["states"], _r["gt"], sample_times=_r["odom"][:, 0]).items()},
                "map RMSE aligned (m)": round(_sm2["map RMSE after rigid alignment (m)"], 3), "|map rotation| (deg)": round(abs(_sm2["alignment rotation (deg)"]), 1),
                "median position NEES": round(float(np.median(_nees)), 1), "% NEES < 11.83": round(100 * float(np.mean(_nees < 11.83)), 1)}

    _rows = [_ekf_row(_n, _res) for _n, _res in _ekf.items()] + [_ekf_row("synthetic", _res, _s) for _s, _res in enumerate(_synth)]
    for _n, _al in {"baseline": e2_alphas, f"{d_k:g}·α": _k_alpha}.items():
        for _sd in range(d_seeds):
            _rows.append({"method": f"FastSLAM M={d_M} {_n}", "seed": _sd, **fastslam_summary(_r, run_fastslam(_r, d_M, _al, e2_Q, seed=_sd))})
    d_table = pd.DataFrame(_rows)
    mo.vstack([
        mo.md(f"Gaps ≥ 20 s: {', '.join(f'{_a:.0f}–{_b:.0f} s' for _a, _b in _gaps) or 'none'} · old landmarks (mapped before {_t_gap0:.0f} s) first reappear at {_t_old:.0f} s"),
        mo.ui.plotly(_fig),
        mo.md("**Heading error / 3σ at key instants, and NIS of the old landmarks when they reappear**"),
        mo.ui.table(pd.DataFrame(_key), selection=None),
        mo.md("**All runs of Part D** (seed = None: deterministic run)"),
        mo.ui.table(d_table, selection=None, page_size=30),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r""" """)
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    #### ✍️ ANSWERS — Part D

    ### D1 — What happens when the robot loses and recovers landmark observations?

    Use the **heading-error plot** with its **3σ bounds**, the **key-instant table** and the **NIS results**, and compare the baseline EKF-SLAM run with the synthetic-measurement runs.

    **D1.1 — During the observation gap.** While the robot receives no landmark measurements, what happens to the actual heading error and to the heading uncertainty reported by the EKF (3σ)? Explain why the filter behaves this way when it can only use odometry.

    > *Your answer…*

    **D1.2 — When observations return.** After the gap, the robot may observe landmarks that were first added to the map **after the gap**, before it eventually sees a landmark mapped **before the gap**. Compare the baseline and the synthetic runs during this period:

    - Does the heading error decrease?
    - Does the reported heading uncertainty (3σ) decrease?
    - Do these two quantities change in the same way?
    - The synthetic measurements satisfy the filter's assumptions exactly. If the baseline and the synthetic runs behave differently during this period, what must be causing the change in the baseline?

    Think about the difference between observing a landmark whose estimated position depends on the robot's erroneous pose after the gap, and observing an older landmark that was mapped before that error accumulated. What does this suggest about the information provided by the two types of observations?

    > *Your answer…*

    **D1.3 — Re-observing older landmarks.** When the robot finally observes landmarks mapped before the gap, inspect the NIS values. Are these observations compatible with what the filter expected, or do they produce unusually large innovations? Explain what a large NIS tells us, and why it does not identify the exact cause of the disagreement by itself.

    > *Your answer…*

    ---
    ### D2 — Can we improve EKF-SLAM by changing the noise parameters?

    **D2.1 — Increasing motion uncertainty.** The motion-noise parameters $\alpha$ determine how much uncertainty the filter adds during prediction. In the k·α run they are multiplied by $k$, so the EKF assumes less reliable odometry. Compare the baseline and the k·α results using the heading error and its 3σ bounds, the NEES and the map rotation. Does increasing the assumed motion noise solve the problems observed in case A? Support your conclusion with the results.

    > *Your answer…*

    **D2.2 — Changing measurement uncertainty.** The alternative configuration uses

    $$
    \sigma_r = 0.10\ \text{m}, \qquad \sigma_\phi = 0.05\ \text{rad},
    $$

    so the filter assumes that range measurements are more accurate, and bearing measurements less accurate, than in the baseline. What changes in the estimated heading and map after the observation gap? Why can changing the assumed uncertainty of a **range** measurement also affect the estimated **heading**?

    *Hint: in EKF-SLAM the robot pose and the landmark positions are estimated jointly. Consider how the Kalman gain can use a measurement residual to correct different components of the state.*

    > *Your answer…*

    **D2.3 — Comparing EKF-SLAM and FastSLAM.** Compare the EKF-SLAM results with FastSLAM in case A:

    - Does FastSLAM estimate the heading more accurately?
    - Is its behaviour similar across random seeds?
    - Does it consistently outperform EKF-SLAM?
    - Does inflating α affect FastSLAM in the same way as EKF-SLAM? Relate your answer to the particle ancestry of Part C.

    Relate your observations to how the two algorithms represent uncertainty: one joint Gaussian in EKF-SLAM versus many **trajectory** hypotheses in FastSLAM.

    > *Your answer…*

    ---
    ### D3 — Could graph-based SLAM handle this differently?

    Our EKF-SLAM estimates the **current robot pose and the landmark map**. It stores previous estimated poses for evaluation and plotting, but it does not update them when new measurements arrive. A graph-based SLAM instead keeps past robot poses as variables connected by odometry and landmark observations, and estimates them jointly by minimising the disagreement between predicted and observed constraints. During the optimisation it can repeatedly recompute the linear approximations of these constraints around the improved estimates (**relinearisation**). Assume known correspondences; **no implementation is required**.

    **D3.1** How is repeatedly relinearising the measurement constraints different from the EKF-SLAM updates of Part A? Why might relinearisation help when the initial estimates contain large errors? Does it guarantee a correct solution?

    > *Your answer…*

    **D3.2** Suppose the robot observes an old landmark again after the gap. Could graph-based SLAM use this measurement to correct the robot poses estimated **during the gap**? Could our online EKF-SLAM change those past poses in the same way? Explain the difference.

    > *Your answer…*

    **D3.3** Name and explain one important problem that graph-based SLAM would still need to handle.

    > *Your answer…*

    ---
    ### D4 — What can we learn from late landmark re-observations?

    A landmark may leave the field of view and be observed again much later. Re-observing a previously mapped landmark can provide information about the robot's accumulated motion error, but a long time between observations does not automatically mean that the measurement provides a large correction. Use the **long-gap re-observation table**.

    **D4.1** Choose one re-observation event from case A and one from case B. For each, report the landmark, the time of the new observation and the time elapsed since its previous observation. If a case has no qualifying event, say so explicitly.

    > *Your answer…*

    **D4.2** Explain why observing an old landmark again may help correct accumulated drift. Why is the time since its last observation **not enough by itself** to predict how useful the measurement will be? Consider the uncertainty of the landmark estimate, the geometry of the observation and the other measurements available to the robot.

    > *Your answer…*

    ---
    **Final reflection** *(short; part of D)*. What do the experiments of Part D tell you about the difference between obtaining an accurate SLAM estimate and obtaining a reliable estimate of its uncertainty?

    > *Your answer…*
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ---
    ## Part E — Synthesis (10 %) · maximum 500 words

    Design a SLAM pipeline for a warehouse robot whose camera observes about 200 **unlabelled visual features per image** and that may map $10^4$–$10^5$ features. Based on your results in Parts B–D, compare EKF-SLAM, FastSLAM and a keyframe/graph-based alternative. Identify which problem would appear first (**data association, computational growth, consistency**) and propose concrete mitigations.

    For context: *keyframes* keep selected camera poses instead of every frame; a sparse graph or bundle-adjustment back-end refines them with relinearised constraints; *place recognition* proposes returns to previously visited places, and geometric verification rejects false loops. You may also draw on what you saw with `slam_toolbox` or Cartographer in the ROS 2 sessions, labelled as qualitative evidence. Cite at least **two numerical results from your own runs**, and distinguish evidence from extrapolation to large scale.

    > *Your answer…*

    ---
    ### ✔️ Before submitting
    - [ ] CHECK 0, B1 and C1 green · every ANSWER cell completed with your own numbers.
    - [ ] `RUN_B2`, `RUN_C_DEMO`, `RUN_C2` and `RUN_D` are True and their tables appear in E2.html.
    - [ ] Your α, Q, thresholds, particle counts and seeds are in the source.
    - [ ] Notebook runs from top to bottom · `.py` + `.html` in `E2_StudentCode_Surname.zip` · individual submission · deadline **Monday 14/12/2026, 23:59**.
    """
    )
    return


if __name__ == "__main__":
    app.run()
