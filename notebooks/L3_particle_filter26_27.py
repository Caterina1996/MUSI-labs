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
    # L3 — Particle Filter Foundations

    **11765 · Navigation and Environment Modelling in Mobile Robotics · MUSI** · Class exercise, Friday 16/10/2026

    **Formative** exercise (not graded). The functions you write here will be part of E1: keep your completed notebook.

    ### Learning objectives
    1. Represent motion uncertainty with samples and compare it with the EKF Gaussian.
    2. Compute particle weights from landmark measurements **in log space**, and measure weight degeneracy with $N_\text{eff}$.
    3. Implement the low-variance resampler and understand why it is preferred to multinomial resampling.
    4. Run Monte Carlo Localization (tracking) on real data and identify its weak points.

    ### Recommended reading
    Thrun et al., *Probabilistic Robotics*: Sec. 4.3 (particle filter, Table 4.3; low-variance sampler, Table 4.4), Sec. 5.3 (sampling the velocity model, Table 5.3), Sec. 8.3 (Monte Carlo Localization, Table 8.2).

    **🛠 TODO** code · **✅ CHECK** automatic feedback · **✍️ ANSWER** short answers with numbers from your runs.
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

    def evaluate(states, gt):
        """ATE, FTE and final heading error of an estimate [t, x, y, theta].
        Ground truth is interpolated at the estimate timestamps (the two streams
        have different rates, so comparing them index by index would be wrong)."""
        _g = gt_at(gt, states[:, 0])
        _e = np.hypot(states[:, 1] - _g[:, 0], states[:, 2] - _g[:, 1])
        _eth = (states[-1, 3] - _g[-1, 2] + np.pi) % (2 * np.pi) - np.pi
        return {
            "ATE (m)": float(np.sqrt(np.mean(_e**2))),
            "FTE (m)": float(_e[-1]),
            "final heading error (deg)": float(abs(np.degrees(_eth))),
        }

    def path_length(gt):
        return float(np.sum(np.hypot(np.diff(gt[:, 1]), np.diff(gt[:, 2]))))
    return evaluate, gt_at, load_run, path_length


@app.cell(hide_code=True)
def _(go, gt_at, make_subplots, np):
    # PROVIDED — plotting helpers
    _COLORS = ["#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#8c564b"]

    def plot_trajectories(run, estimates, title="Trajectories", ellipses=None):
        """estimates: dict name -> array [t, x, y, theta]; ellipses: list of (mean_xy, cov_2x2)."""
        _fig = go.Figure()
        _fig.add_trace(go.Scatter(x=run["gt"][:, 1], y=run["gt"][:, 2], mode="lines", name="ground truth", line=dict(color="#1f77b4", dash="dot")))
        _lx = [p[0] for p in run["landmarks"].values()]
        _ly = [p[1] for p in run["landmarks"].values()]
        _fig.add_trace(go.Scatter(x=_lx, y=_ly, mode="markers", name="landmarks", marker=dict(symbol="star", size=10, color="gold", line=dict(width=0))))
        for _i, (_name, _s) in enumerate(estimates.items()):
            if _s is not None:
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
    ### 🛠 TODO 0 — Paste your functions from L1 and L2

    Paste your implementations into the function bodies of the next cell, **keeping the names and signatures unchanged**:

    - from L1: `wrap_angle`, `motion_euler`;
    - from L2 (TODO 1): `G_jacobian`, `V_jacobian`, `control_noise`, `ekf_predict`.

    Part 1 uses them to compare the particle cloud with the EKF prediction. Dead reckoning is provided below: you do not need to paste it.
    """
    )
    return


@app.cell
def _(np):
    # TODO 0 — paste your L1 / L2 implementations (same names and signatures)

    # ---- from L1 ----
    def wrap_angle(a):
        return None

    def motion_euler(x, v, w, dt):
        return None

    # ---- from L2 (TODO 1) ----
    def G_jacobian(mu, v, w, dt):
        return None

    def V_jacobian(mu, v, w, dt):
        return None

    def control_noise(v, w, alphas):
        return None

    def ekf_predict(mu, Sigma, v, w, dt, alphas):
        return None, None

    return G_jacobian, V_jacobian, control_noise, ekf_predict, motion_euler, wrap_angle


@app.cell(hide_code=True)
def _(G_jacobian, V_jacobian, control_noise, ekf_predict, mo, motion_euler, np, wrap_angle):
    def _check():
        try:
            _mu = np.array([1.0, 2.0, 0.7])
            _a = (0.1, 0.01, 0.01, 0.1)
            _G = np.array([[1, 0, -0.3 * 0.05 * np.sin(0.7)], [0, 1, 0.3 * 0.05 * np.cos(0.7)], [0, 0, 1]])
            _V = np.array([[0.05 * np.cos(0.7), 0], [0.05 * np.sin(0.7), 0], [0, 0.05]])
            _M = np.diag([0.1 * 0.09 + 0.01 * 0.04, 0.01 * 0.09 + 0.1 * 0.04])
            _m, _S = ekf_predict(_mu, np.eye(3) * 0.01, 0.3, 0.2, 0.05, _a)
            _r = [
                ("wrap_angle", np.allclose(wrap_angle(np.array([3 * np.pi / 2, -3 * np.pi / 2, 0.1])), [-np.pi / 2, np.pi / 2, 0.1])),
                ("motion_euler", np.allclose(motion_euler(np.zeros(3), 1.0, np.pi / 2, 1.0), [1, 0, np.pi / 2])),
                ("G_jacobian", np.allclose(G_jacobian(_mu, 0.3, 0.2, 0.05), _G)),
                ("V_jacobian", np.allclose(V_jacobian(_mu, 0.3, 0.2, 0.05), _V)),
                ("control_noise", np.allclose(control_noise(0.3, 0.2, _a), _M)),
                ("ekf_predict (mean and covariance)", np.allclose(_m, motion_euler(_mu, 0.3, 0.2, 0.05)) and np.allclose(_S, _G @ (np.eye(3) * 0.01) @ _G.T + _V @ _M @ _V.T)),
            ]
        except Exception as _e:
            return f"⏳ not pasted yet ({type(_e).__name__})"
        return "<br>".join(f"{'✅' if _ok else '❌'} {_n}" for _n, _ok in _r)

    mo.md("**CHECK — L1 / L2 functions**<br>" + _check())
    return


@app.cell(hide_code=True)
def _(motion_euler, np):
    # PROVIDED — dead reckoning with your motion_euler (used in Part 4)
    def dead_reckoning(run):
        _x, _t, _u = run["x0"].copy(), run["t0"], (0.0, 0.0)
        _out = [[_t, *_x]]
        for _e in run["events"]:
            if _e[1] != 0:
                continue
            if _e[0] > _t:
                _x = motion_euler(_x, _u[0], _u[1], _e[0] - _t)
                if _x is None:
                    return None
                _t = _e[0]
                _out.append([_t, *_x])
            _u = (_e[2], _e[3])
        return np.array(_out)

    return (dead_reckoning,)


@app.cell(hide_code=True)
def _(mo):
    dataset_dd = mo.ui.dropdown([f"MRCLAM_Dataset{_j}" for _j in range(1, 10)], value="MRCLAM_Dataset1", label="Dataset")
    robot_dd = mo.ui.dropdown([f"Robot{_j}" for _j in range(1, 6)], value="Robot1", label="Robot")
    duration_sl = mo.ui.slider(30, 1200, step=30, value=300, label="Duration (s)", show_value=True)
    mo.hstack([dataset_dd, robot_dd, duration_sl], justify="start")
    return dataset_dd, duration_sl, robot_dd


@app.cell(hide_code=True)
def _(dataset_dd, duration_sl, load_run, mo, path_length, robot_dd):
    run = load_run(dataset_dd.value, robot_dd.value, duration_sl.value)
    mo.md(
        f"**{run['dataset']} / {run['robot']}** — {run['gt'][-1, 0] - run['gt'][0, 0]:.0f} s · "
        f"path {path_length(run['gt']):.1f} m · {len(run['odom'])} odometry commands · "
        f"{len(run['meas'])} landmark measurements · {run['n_robot_meas']} measurements of other robots (discarded)"
    )
    return (run,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    **Control-noise parameters** (recall L2)

    The motion model accounts for uncertainty in linear ($v$)
    and angular ($\omega$) velocities:

    $$
    \sigma_v^2 = \alpha_1 v^2 + \alpha_2 \omega^2
    $$

    $$
    \sigma_\omega^2 = \alpha_3 v^2 + \alpha_4 \omega^2
    $$

    - **$\alpha_1$:** Linear velocity noise caused by translation.
    - **$\alpha_2$:** Linear velocity noise caused by rotation.
    - **$\alpha_3$:** Angular velocity noise caused by translation.
    - **$\alpha_4$:** Angular velocity noise caused by rotation.

    **Try it:** How does increasing each parameter affect
    the spread of the particles?
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    l3_a1 = mo.ui.slider(0.0, 2.0, step=0.05, value=0.2, label="α1", show_value=True)
    l3_a2 = mo.ui.slider(0.0, 0.5, step=0.01, value=0.02, label="α2", show_value=True)
    l3_a3 = mo.ui.slider(0.0, 0.5, step=0.01, value=0.02, label="α3", show_value=True)
    l3_a4 = mo.ui.slider(0.0, 8.0, step=0.1, value=0.5, label="α4", show_value=True)
    mo.hstack([l3_a1, l3_a2, l3_a3, l3_a4], justify="start")
    return l3_a1, l3_a2, l3_a3, l3_a4


@app.cell(hide_code=True)
def _(l3_a1, l3_a2, l3_a3, l3_a4):
    l3_alphas = (l3_a1.value, l3_a2.value, l3_a3.value, l3_a4.value)
    return (l3_alphas,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ---
    ## 1 · Sampling the motion model (≈ 25 min)

    In L2, the EKF propagated a **mean pose** and its **covariance** to represent motion uncertainty.

    In a Particle Filter, we represent uncertainty using a set of $N$ particles, where each particle is a possible robot pose $(x,y,\theta)$.

    **Prediction:** For each particle, we generate a new pose by applying the motion model. All particles receive the same control input $(v,\omega)$, but we add independently sampled noise to the velocities.

    ### Step 1 — Sample noisy velocities

    For each particle $i$:

    $$
    \hat v^{[i]} \sim \mathcal N(v,\;\alpha_1v^2+\alpha_2\omega^2)
    $$

    $$
    \hat\omega^{[i]} \sim \mathcal N(\omega,\;\alpha_3v^2+\alpha_4\omega^2)
    $$

    Here, $\hat v^{[i]}$ and $\hat\omega^{[i]}$ are the sampled linear and angular velocities for particle $i$. The $\alpha$ parameters control the motion noise, as in L2.

    ### Step 2 — Predict the new pose (Euler integration)

    Recall the Euler motion model from L1/L2. For each particle:

    $$
    x_t^{[i]} = x_{t-1}^{[i]} + \hat v^{[i]}\Delta t\cos(\theta_{t-1}^{[i]})
    $$

    $$
    y_t^{[i]} = y_{t-1}^{[i]} + \hat v^{[i]}\Delta t\sin(\theta_{t-1}^{[i]})
    $$

    $$
    \theta_t^{[i]} = \operatorname{wrap}\left(\theta_{t-1}^{[i]}+\hat\omega^{[i]}\Delta t\right)
    $$

    The result is a new set of $N$ predicted particles, each with a slightly different pose.

    ### 🛠 TODO 1 — Implement the motion prediction

    Complete `sample_motion_model(P, v, w, dt, alphas, rng)`.

    **Inputs:**

    - `P`: NumPy array of shape `(N, 3)` containing the current particle poses $(x,y,\theta)$.
    - `v`, `w`: Linear and angular velocities from odometry.
    - `dt`: Time interval $\Delta t$ (seconds).
    - `alphas`: The four motion-noise parameters $(\alpha_1,\alpha_2,\alpha_3,\alpha_4)$.
    - `rng`: NumPy random number generator.

    **Expected output:** A new NumPy array of shape `(N, 3)` containing the predicted poses.

    **Implementation requirements:**

    1. Sample a noisy linear and angular velocity for each particle. Remember that `rng.normal()` expects the **standard deviation**, not the variance.
    2. Apply the Euler motion model to all particles.
    3. Normalize the headings using `wrap_angle()`.
    4. **Vectorise** the implementation using NumPy. Do not use a Python loop over particles.
    """
    )
    return


@app.cell
def _(np, wrap_angle):
    def sample_motion_model(P, v, w, dt, alphas, rng):
        return None

    return (sample_motion_model,)


@app.cell(hide_code=True)
def _(mo, np, sample_motion_model):
    def _check():
        try:
            # reference values computed here, independent of your TODO 0 functions
            _rng = np.random.default_rng(1)
            _x = np.array([1.0, 2.0, 0.5])
            _a = (0.1, 0.05, 0.05, 0.1)
            _mean = np.array([1.0 + 0.25 * np.cos(0.5), 2.0 + 0.25 * np.sin(0.5), 0.7])
            _V = np.array([[0.5 * np.cos(0.5), 0.0], [0.5 * np.sin(0.5), 0.0], [0.0, 0.5]])
            _M = np.diag([0.1 * 0.25 + 0.05 * 0.16, 0.05 * 0.25 + 0.1 * 0.16])
            _P = sample_motion_model(np.tile(_x, (40000, 1)), 0.5, 0.4, 0.5, _a, _rng)
            _r = [
                ("shape (N, 3)", _P.shape == (40000, 3)),
                ("sample mean ≈ motion model", np.allclose(_P.mean(0), _mean, atol=3e-3)),
                ("sample covariance ≈ V M Vᵀ (one step)", np.allclose(np.cov(_P.T), _V @ _M @ _V.T, atol=3e-4)),
                ("zero noise → deterministic model", np.allclose(sample_motion_model(np.tile(_x, (5, 1)), 0.5, 0.4, 0.5, (0, 0, 0, 0), _rng), _mean)),
            ]
        except Exception as _e:
            return f"⏳ not implemented yet ({type(_e).__name__})"
        return "<br>".join(f"{'✅' if _ok else '❌'} {_n}" for _n, _ok in _r)

    mo.md("**CHECK — sampling motion model**<br>" + _check())
    return


@app.cell(hide_code=True)
def _(mo):
    cloud_T = mo.ui.slider(1, 60, step=1, value=10, label="Propagate the first T seconds", show_value=True)
    cloud_T
    return (cloud_T,)


@app.cell(hide_code=True)
def _(cloud_T, ekf_predict, go, l3_alphas, mo, np, run, sample_motion_model, wrap_angle):

    # PROVIDED — Motion prediction only: Particle Filter vs. EKF
    # Same odometry and motion-noise parameters. No sensor correction.

    _rng = np.random.default_rng(0)
    _P = np.tile(run["x0"], (1000, 1))
    _mu = run["x0"].copy()
    _S = np.diag([1e-6] * 3)
    _t, _u, _ok = run["t0"], (0.0, 0.0), True

    # Save particle clouds at approximately 25%, 50%, 75%, 100% of T
    _checkpoints = np.linspace(0.0, float(cloud_T.value), 5)[1:]
    _next_cp = 0
    _snapshots = []
    _path = [[_t - run["t0"], _mu[0], _mu[1]]]

    # --------------------------------------------------------
    # 1. PROPAGATE PARTICLES AND EKF USING ODOMETRY ONLY
    # --------------------------------------------------------

    for _e in run["events"]:
        if _e[1] != 0:
            continue  # Ignore measurements

        if _e[0] - run["t0"] > cloud_T.value:
            break

        if _e[0] > _t:
            _dt = _e[0] - _t

            # Particle Filter prediction
            _P = sample_motion_model(
                _P, _u[0], _u[1], _dt, l3_alphas, _rng
            )
            if _P is None:
                _ok = False
                break

            # EKF prediction
            _mu, _S = ekf_predict(
                _mu, _S, _u[0], _u[1], _dt, l3_alphas
            )
            if _mu is None:
                _ok = False
                break

            _t = _e[0]
            _elapsed = _t - run["t0"]
            _path.append([_elapsed, _mu[0], _mu[1]])

            # Save intermediate prediction snapshots
            if (_next_cp < len(_checkpoints)
                    and _elapsed >= _checkpoints[_next_cp]):

                _snapshots.append((
                    _elapsed,
                    _P.copy(),
                    _mu.copy(),
                    _S.copy()
                ))

                while (_next_cp < len(_checkpoints) and
                       _elapsed >= _checkpoints[_next_cp]):
                    _next_cp += 1

        # The new command applies to the NEXT time interval
        _u = (_e[2], _e[3])

    if not _ok:
        _out = mo.md("⏳ Complete TODO 0 (L1/L2 functions) and TODO 1 first.")

    else:
        _elapsed = _t - run["t0"]

        # Always include the final prediction
        if not _snapshots or _snapshots[-1][0] < _elapsed - 1e-9:
            _snapshots.append((
                _elapsed, _P.copy(), _mu.copy(), _S.copy()
            ))

        # ----------------------------------------------------
        # 2. HELPER: EKF 3-SIGMA POSITION ELLIPSE
        # ----------------------------------------------------
        # For a 2D Gaussian, a 3-sigma ellipse contains
        # approximately 98.9% of the probability.

        _phi = np.linspace(0, 2 * np.pi, 100)
        _circle = np.stack([np.cos(_phi), np.sin(_phi)])

        def _ellipse_xy(Sigma):
            vals, vecs = np.linalg.eigh(Sigma[:2, :2])
            return vecs @ (
                3 * np.sqrt(np.maximum(vals, 0))[:, None]
                * _circle
            )

        # ----------------------------------------------------
        # 3. FIGURE 1 — EVOLUTION OF MOTION UNCERTAINTY
        # ----------------------------------------------------

        _fig_steps = go.Figure()
        _path_arr = np.asarray(_path)

        # EKF mean trajectory (equivalent to dead reckoning)
        _fig_steps.add_trace(go.Scatter(
            x=_path_arr[:, 1],
            y=_path_arr[:, 2],
            mode="lines",
            line=dict(color="gray", dash="dot", width=2),
            name="EKF mean (dead reckoning)"
        ))

        _colors = [
            "#636EFA", "#EF553B", "#00A98F",
            "#AB63FA", "#FFA15A"
        ]

        for _i, (_ts, _Ps, _mus, _Ss) in enumerate(_snapshots):
            _c = _colors[_i % len(_colors)]

            # Show 1 in 5 particles to avoid visual clutter
            _fig_steps.add_trace(go.Scatter(
                x=_Ps[::5, 0],
                y=_Ps[::5, 1],
                mode="markers",
                marker=dict(
                    color=_c, size=4, opacity=0.35
                ),
                name=f"t = {_ts:.1f} s",
                legendgroup=str(_i)
            ))

            # EKF uncertainty ellipse at the same time
            _el = _ellipse_xy(_Ss)

            _fig_steps.add_trace(go.Scatter(
                x=_mus[0] + _el[0],
                y=_mus[1] + _el[1],
                mode="lines",
                line=dict(color=_c, width=2),
                name=f"EKF 3σ at {_ts:.1f} s",
                legendgroup=str(_i),
                showlegend=False
            ))

        _fig_steps.update_layout(
            title="Evolution of motion uncertainty — prediction only",
            xaxis_title="x (m)",
            yaxis=dict(
                title="y (m)", scaleanchor="x", scaleratio=1
            ),
            legend_title="Checkpoint",
            height=500,
            plot_bgcolor="white"
        )

        # ----------------------------------------------------
        # 4. FIGURE 2 — FINAL PARTICLES VS. EKF
        # ----------------------------------------------------

        _fig_final = go.Figure()

        # All 1000 particles
        _fig_final.add_trace(go.Scatter(
            x=_P[:, 0],
            y=_P[:, 1],
            mode="markers",
            marker=dict(size=3, opacity=0.45),
            name="Particles (N=1000)"
        ))

        # Final EKF 3-sigma ellipse
        _el = _ellipse_xy(_S)

        _fig_final.add_trace(go.Scatter(
            x=_mu[0] + _el[0],
            y=_mu[1] + _el[1],
            mode="lines",
            line=dict(width=3),
            name="EKF 3σ ellipse"
        ))

        # Final EKF mean
        _fig_final.add_trace(go.Scatter(
            x=[_mu[0]],
            y=[_mu[1]],
            mode="markers",
            marker=dict(
                size=9, symbol="x", color="black"
            ),
            name="EKF mean"
        ))

        _fig_final.update_layout(
            title=f"Final uncertainty after {_elapsed:.1f} s",
            xaxis_title="x (m)",
            yaxis=dict(
                title="y (m)", scaleanchor="x", scaleratio=1
            ),
            height=500,
            plot_bgcolor="white"
        )

        # ----------------------------------------------------
        # 5. FINAL STATISTICS
        # ----------------------------------------------------

        # Circular heading statistics: -pi and +pi are adjacent
        _th_mean = np.arctan2(
            np.sin(_P[:, 2]).mean(),
            np.cos(_P[:, 2]).mean()
        )
        _th_std = np.std(wrap_angle(_P[:, 2] - _th_mean))
        _ekf_std = np.sqrt(np.maximum(np.diag(_S), 0))

        _stats = mo.md(
            f"**Mean position (m)** — "
            f"Particles: ({_P[:, 0].mean():.3f}, "
            f"{_P[:, 1].mean():.3f}); "
            f"EKF: ({_mu[0]:.3f}, {_mu[1]:.3f})\n\n"

            f"**Position std (m)** — "
            f"Particles: ({_P[:, 0].std():.3f}, "
            f"{_P[:, 1].std():.3f}); "
            f"EKF: ({_ekf_std[0]:.3f}, {_ekf_std[1]:.3f})\n\n"

            f"**Heading std (rad)** — "
            f"Particles: {_th_std:.3f}; "
            f"EKF: {_ekf_std[2]:.3f}\n\n"

            "*Prediction only: no landmark measurements are used. "
            "The EKF 3σ ellipse contains approximately 98.9% "
            "probability for a 2D Gaussian.*"
        )

        # ----------------------------------------------------
        # 6. DISPLAY RESULTS
        # ----------------------------------------------------

        _out = mo.vstack([
            mo.ui.plotly(_fig_steps),
            mo.ui.plotly(_fig_final),
            _stats
        ])

    _out
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    #### ✍️ ANSWER 1

    **1.1** Compare the cloud and the EKF ellipse for increasing $T$, first with the default α and then with a large one, e.g. α = (2, 0.5, 0.5, 8). Do they remain similar? Under what conditions would the Gaussian approximation become less accurate, and why?

    > *Your answer…*

    **1.2** Why must each particle receive its **own** noisy control, instead of adding the same noise to all of them?

    > *Your answer…*

    **1.3** What happens to the particle distribution when increasing $\alpha_1$ compared with increasing $\alpha_4$? Explain the difference in terms of translational and rotational uncertainty.

    > *Your answer…*
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 2 · Measurement update: Particle weights (≈ 30 min)

    In Part 1, we predicted the robot's possible poses using
    the motion model. Now, we use sensor measurements to
    determine **which particles are more likely to represent
    the robot's actual pose**.

    The idea is simple: a particle receives a higher weight
    if its predicted sensor measurement is close to the
    actual observation.

    ### Step 1 — Predict the measurement

    Recall the landmark observation model from L2.

    For each particle $i$ with pose $(x^{[i]},y^{[i]},\theta^{[i]})$
    and a known landmark at $(m_x,m_y)$, we predict:

    $$
    \hat r^{[i]} =
    \sqrt{(m_x-x^{[i]})^2+(m_y-y^{[i]})^2}
    $$

    $$
    \hat\phi^{[i]} =
    \operatorname{wrap}\left(
    \operatorname{atan2}(m_y-y^{[i]},m_x-x^{[i]})
    -\theta^{[i]}
    \right)
    $$

    Here, $\hat r^{[i]}$ is the predicted distance and
    $\hat\phi^{[i]}$ is the predicted relative bearing.

    We compare these predictions with the actual sensor
    measurement $z=(r,\phi)$:

    $$
    e_r^{[i]}=r-\hat r^{[i]}
    $$

    $$
    e_\phi^{[i]}=
    \operatorname{wrap}(\phi-\hat\phi^{[i]})
    $$

    ### Step 2 — Compute the measurement likelihood

    We assume independent Gaussian errors in range and
    bearing, with standard deviations $\sigma_r$ (metres)
    and $\sigma_\phi$ (radians).

    A particle with small measurement errors should receive
    a higher likelihood than a particle with large errors.

    Instead of computing the likelihood directly, we compute
    its logarithm:

    $$
    \log p(z\mid x^{[i]}) =
    -\frac12\left[
    \left(\frac{e_r^{[i]}}{\sigma_r}\right)^2+
    \left(\frac{e_\phi^{[i]}}{\sigma_\phi}\right)^2
    \right]
    -\log(2\pi\sigma_r\sigma_\phi)
    $$

    **Why log-likelihoods?**

    When multiple measurements are received, their
    likelihoods are multiplied. Repeated multiplication
    of very small numbers can cause numerical underflow.

    Using logarithms, we can add instead of multiply:

    $$
    \ell^{[i]} \leftarrow
    \ell^{[i]}+\log p(z\mid x^{[i]})
    $$

    where $\ell^{[i]}$ is the accumulated log-weight
    of particle $i$.

    ### Step 3 — Normalize the particle weights

    To obtain weights that sum to 1, we convert the
    accumulated log-weights back to normal weights:

    $$
    w^{[i]} =
    \frac{\exp(\ell^{[i]}-\ell_{\max})}
    {\sum_j\exp(\ell^{[j]}-\ell_{\max})}
    $$

    where $\ell_{\max}$ is the largest log-weight.

    Subtracting the maximum improves numerical stability
    without changing the relative weights.

    ### Step 4 — Measure particle degeneracy

    After several measurements, most of the total weight
    may become concentrated in only a few particles.
    This is called **weight degeneracy**.

    We quantify it using the effective sample size:

    $$
    N_{\mathrm{eff}} =
    \frac{1}{\sum_{i=1}^{N}(w^{[i]})^2}
    $$

    - $N_{\mathrm{eff}}\approx N$: weights are nearly uniform.
    - $N_{\mathrm{eff}}\approx 1$: almost all weight is concentrated
      in one particle.

    A low $N_{\mathrm{eff}}$ indicates that resampling may
    be needed (Part 3).

    ### 🛠 TODO 2 — Implement particle weighting

    Implement the following three functions using
    vectorized NumPy operations.

    **2A. `measurement_log_likelihood(P, z, landmark, sigma)`**

    Inputs:
    - `P`: `(N, 3)` array of particle poses $(x,y,\theta)$.
    - `z`: observed measurement `(range, bearing)`.
    - `landmark`: known landmark position `(m_x, m_y)`.
    - `sigma`: sensor noise standard deviations
      `(sigma_range, sigma_bearing)`.

    Tasks:
    1. Predict the range and bearing for every particle.
    2. Compute the measurement errors (wrap bearing errors).
    3. Evaluate the Gaussian log-likelihood.

    **Return:** A NumPy array of shape `(N,)` containing
    one log-likelihood per particle (not normalized weights).

    **2B. `normalize_log_weights(logw)`**

    Input:
    - `logw`: `(N,)` array of accumulated particle log-weights.

    Tasks:
    1. Subtract the maximum log-weight.
    2. Apply the exponential.
    3. Divide by the sum of the resulting values.

    **Return:** A NumPy array of shape `(N,)` containing
    normalized weights that sum to 1.

    **2C. `effective_sample_size(w)`**

    Input:
    - `w`: `(N,)` array of normalized particle weights.

    Task:
    - Calculate the effective sample size using the
      equation above.

    **Return:** A single numerical value between 1 and N.
    """
    )
    return


@app.cell
def _(np, wrap_angle):
    def measurement_log_likelihood(P, z, landmark, sigma):
        return None

    def normalize_log_weights(logw):
        return None

    def effective_sample_size(w):
        return None
    return (
        effective_sample_size,
        measurement_log_likelihood,
        normalize_log_weights,
    )


@app.cell(hide_code=True)
def _(
    effective_sample_size,
    measurement_log_likelihood,
    mo,
    normalize_log_weights,
    np,
):
    def _check():
        try:
            _P = np.array([[0.0, 0.0, 0.0], [2.0, 1.0, 0.5], [0.0, 0.0, np.pi]])
            _ll = measurement_log_likelihood(_P, (3.0, 0.0), (3.0, 0.0), (0.1, 0.05))
            _w = normalize_log_weights(np.array([-1000.0, -1001.0, -2000.0]))
            _r = [
                ("likelihood shape (N,) and correct particle most likely", np.shape(_ll) == (3,) and np.argmax(_ll) == 0),
                ("exact log-density at zero error", np.isclose(_ll[0], -np.log(2 * np.pi * 0.1 * 0.05))),
                ("bearing error wrapped (particle facing backwards is not 2π away)", np.isclose(measurement_log_likelihood(np.array([[0.0, 0.0, 0.01]]), (3.0, -np.pi + 0.01), (-3.0, 0.0), (0.1, 0.05))[0], -np.log(2 * np.pi * 0.1 * 0.05) - 0.5 * (0.02 / 0.05) ** 2, atol=1e-6)),
                ("normalisation stable for very negative log-weights", np.all(np.isfinite(_w)) and np.isclose(_w.sum(), 1) and np.isclose(_w[0] / _w[1], np.e)),
                ("N_eff: uniform weights → N, one particle → 1", np.isclose(effective_sample_size(np.ones(10) / 10), 10) and np.isclose(effective_sample_size(np.eye(10)[0]), 1)),
            ]
        except Exception as _e:
            return f"⏳ not implemented yet ({type(_e).__name__})"
        return "<br>".join(f"{'✅' if _ok else '❌'} {_n}" for _n, _ok in _r)

    mo.md("**CHECK — weights**<br>" + _check())
    return


@app.cell(hide_code=True)
def _(mo):
    one_k = mo.ui.slider(1, 3, step=1, value=1, label="Measurements k", show_value=True)
    one_unknown = mo.ui.checkbox(value=False, label="Unknown heading (θ uniform)")
    one_N = mo.ui.dropdown({"500": 500, "2 000": 2000, "5 000": 5000, "20 000": 20000}, value="2 000", label="Particles N")
    mo.hstack([one_k, one_unknown, one_N], justify="start")
    return one_N, one_k, one_unknown


@app.cell(hide_code=True)
def _(
    effective_sample_size,
    go,
    measurement_log_likelihood,
    mo,
    normalize_log_weights,
    np,
    one_N,
    one_k,
    one_unknown,
    run,
    wrap_angle,
):
    # PROVIDED — measurement update ONLY: particles neither move nor resample.
    _N = one_N.value
    _sigma = (0.30, 0.20)  # range [m], bearing [rad]; same noise for the sensor and the filter

    # separate generators: positions and observations do not change when the heading option changes,
    # and the first n particles are the same for any N (nested samples)
    _rng_xy, _rng_th, _rng_z = (np.random.default_rng(s) for s in (0, 1, 2))

    _L = np.asarray(list(run["landmarks"].values()), dtype=float)
    _x0 = np.asarray(run["x0"], dtype=float)
    _lo, _hi = _L.min(axis=0) - 0.5, _L.max(axis=0) + 0.5
    _xy = _rng_xy.uniform(_lo, _hi, size=(_N, 2))
    _th = _rng_th.uniform(-np.pi, np.pi, _N) if one_unknown.value else np.full(_N, _x0[2])
    _P = np.column_stack((_xy, _th))

    # three synthetic observations from the TRUE pose: near, medium and far landmark
    _order = np.argsort(np.hypot(*(_L - _x0[:2]).T))
    _sel = _L[_order[[0, len(_order) // 2, -1]]]
    _zs = []
    for _m in _sel:
        _dx, _dy = _m - _x0[:2]
        _zs.append((np.hypot(_dx, _dy) + _rng_z.normal(0.0, _sigma[0]),
                    wrap_angle(np.arctan2(_dy, _dx) - _x0[2] + _rng_z.normal(0.0, _sigma[1])), _m))

    _ready = (measurement_log_likelihood(_P[:1], _zs[0][:2], _zs[0][2], _sigma) is not None
              and normalize_log_weights(np.zeros(2)) is not None
              and effective_sample_size(np.ones(2) / 2) is not None)
    if not _ready:
        _out = mo.md("⏳ Implement TODO 2 first.")
    else:
        _k = one_k.value
        _logw = np.zeros(_N)
        for _r, _b, _m in _zs[:_k]:
            _logw += measurement_log_likelihood(_P, (_r, _b), _m, _sigma)
        _neff = effective_sample_size(normalize_log_weights(_logw))

        # Visualize a relative likelihood threshold, not the raw normalized
        # weights (which can be tiny). Color classification is for display only.
        _relative_logw = _logw - np.max(_logw)
        _high = _relative_logw >= -6.0

        _fig = go.Figure()
        _fig.add_trace(go.Scattergl(
            x=_P[:, 0], y=_P[:, 1], mode="markers",
            name="All candidate poses",
            marker=dict(color="#aeb6bf", size=3, opacity=0.24)
        ))
        _fig.add_trace(go.Scattergl(
            x=_P[_high, 0], y=_P[_high, 1], mode="markers",
            name="High likelihood",
            marker=dict(color="#ef8c27", size=6, opacity=0.9)
        ))
        _fig.add_trace(go.Scatter(
            x=_L[:, 0], y=_L[:, 1], mode="markers",
            name="Known landmarks",
            marker=dict(symbol="star", size=9, color="#64748b")
        ))
        _fig.add_trace(go.Scatter(
            x=_sel[:_k, 0], y=_sel[:_k, 1], mode="markers",
            name="Observed landmark(s)",
            marker=dict(symbol="circle-open", size=19,
                        color="#d23837", line=dict(width=2))
        ))
        _fig.add_trace(go.Scatter(
            x=[_x0[0]], y=[_x0[1]], mode="markers",
            name="Ground truth",
            marker=dict(symbol="x", size=13,
                        color="#111827", line=dict(width=3))
        ))

        # Keep a 1:1 metric aspect ratio without shrinking the plot area.
        _fig.update_layout(
            title=f"Experiment A — {_k} synthetic landmark observation(s)",
            width=780, height=610, template="plotly_white",
            xaxis=dict(title="x (m)", range=[_lo[0]-0.3, _hi[0]+0.3],
                       constrain="domain"),
            yaxis=dict(title="y (m)", range=[_lo[1]-0.3, _hi[1]+0.3],
                       scaleanchor="x", scaleratio=1, constrain="domain"),
            legend=dict(orientation="h", x=0.5, xanchor="center",
                        y=1.04, yanchor="bottom", font=dict(size=11)),
            margin=dict(l=55, r=30, t=110, b=55),
        )
        _out = mo.vstack([
            _fig,
            mo.md(
                f"**N_eff = {_neff:.0f} / {_N}** "
                f"({100 * _neff / _N:.1f}%) · "
                f"**High-likelihood particles: {np.count_nonzero(_high)}** · "
                f"heading **{'unknown' if one_unknown.value else 'known'}** · "
                f"measurements **{_k}**  \n\n"
                "*Grey = candidates; orange = log-likelihood within 6 of the maximum. "
                "Particles do not move: only weights change.*"
            )
        ])
    _out
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    #### ✍️ ANSWER 2

    **2.1 (known heading)** Describe the high-weight region after 1, 2 and 3 measurements. The first observed landmark is the nearest one and the third the farthest: compare the radial width (≈ σ_r) with the tangential width (≈ r·σ_φ) for each. Is the true pose exactly at the centre of the orange region? Why not?

    > *Your answer…*

    **2.2 (unknown heading)** Tick *unknown heading*. What does the region look like after one measurement, and why can the bearing no longer fix the direction? What changes with 2 and 3 measurements? Compare N_eff with the known-heading case.

    > *Your answer…*

    **2.3** With unknown heading and k = 3, compare N = 500, 2 000 and 20 000. Are there always particles close to the true pose? What does this tell you about the number of particles needed for global localization, where the state (x, y, θ) is unknown?

    > *Your answer…*

    **2.4** Orange particles have relative log-likelihood $\ell-\max\ell\geq -6$. What weight ratio does this threshold represent? Why do we accumulate log-weights and subtract $\max\ell$ before exponentiating, and what would happen after a few hundred measurements without it?

    > *Your answer…*
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 3 · Resampling (≈ 20 min)

    After a few measurements most particles have a negligible weight and only a few carry the belief (low $N_\text{eff}$). **Resampling** builds a new set of $N$ particles by copying each particle a number of times roughly proportional to its weight: likely particles are duplicated, unlikely ones disappear. After resampling all particles have the same weight again ($1/N$); `run_mcl` takes care of resetting the weights.

    We use the **low-variance (systematic) sampler** seen in class:

    1. Build the cumulative weights $c_i = w_1 + \dots + w_i$. They split $[0, 1)$ into $N$ consecutive intervals, interval $i$ has length $w_i$.
    2. Draw a **single** random number $r \sim U[0, 1/N)$.
    3. Place $N$ equally spaced pointers $U_m = r + m/N$, $m = 0, \dots, N-1$.
    4. For each pointer, select the particle whose interval contains it: the first $i$ with $c_i \ge U_m$.

    Example: $w = (\tfrac12, \tfrac14, \tfrac14, 0)$ gives intervals of length ½, ¼, ¼, 0. Whatever $r$ is, the four pointers (spaced ¼ apart) give copies $(2, 1, 1, 0)$.

    ### 🛠 TODO 3
    Implement `low_variance_resample(P, w, rng)`.

    - **Inputs:** `P`, array `(N, 3)` of particles `[x, y, θ]`; `w`, array `(N,)` of **normalised** weights (they sum to 1); `rng`, a NumPy random generator (use `rng.uniform`, so that results are reproducible).
    - **Output:** a **new** array `(N, 3)` containing the selected particles (copies of rows of `P`).
    - **Requirements:** use a single random number; no Python loop over particles (hint: `np.cumsum` and `np.searchsorted`). Because of floating-point rounding, the last cumulative weight can be slightly smaller than 1: make sure no index falls outside the array.

    *Reference: Thrun, Burgard & Fox, Probabilistic Robotics, Table 4.4.*
    """
    )
    return


@app.cell
def _(np):
    def low_variance_resample(P, w, rng):
        return None

    return (low_variance_resample,)


@app.cell(hide_code=True)
def _(low_variance_resample, mo, np):
    def _check():
        try:
            _P = np.arange(4.0)[:, None] * np.ones((1, 3))
            _ok_all = True
            for _s in range(20):
                _R = low_variance_resample(_P, np.array([0.5, 0.25, 0.25, 0.0]), np.random.default_rng(_s))
                _ok_all &= [int(np.sum(_R[:, 0] == _i)) for _i in range(4)] == [2, 1, 1, 0]
            _R2 = low_variance_resample(np.arange(1000.0)[:, None] * np.ones((1, 3)), np.ones(1000) / 1000, np.random.default_rng(0))
            _r = [
                ("weights (½, ¼, ¼, 0) → copies (2, 1, 1, 0) for every random draw", _ok_all),
                ("uniform weights keep every particle exactly once", len(np.unique(_R2[:, 0])) == 1000),
            ]
        except Exception as _e:
            return f"⏳ not implemented yet ({type(_e).__name__})"
        return "<br>".join(f"{'✅' if _ok else '❌'} {_n}" for _n, _ok in _r)

    mo.md("**CHECK — low-variance resampling**<br>" + _check())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""**Why low-variance?** The table compares *your* `low_variance_resample` with the naive alternative, **multinomial resampling** (the "roulette wheel" of T4, slides 15 and 46): N independent draws with probabilities $w$, implemented here in one line with `rng.choice(N, N, p=w)`. Both are run 2000 times on the same 10 weights, and we count how many copies each particle receives.""")
    return


@app.cell(hide_code=True)
def _(low_variance_resample, mo, np, pd):
    # PROVIDED — compares YOUR low-variance sampler with multinomial resampling (rng.choice), 2000 trials each
    _N = 10
    _w = np.array([0.30, 0.20, 0.15, 0.10, 0.08, 0.07, 0.05, 0.03, 0.01, 0.01])
    _P = np.arange(_N, dtype=float)[:, None] * np.ones((1, 3))
    _rng = np.random.default_rng(0)
    _out = mo.md("⏳ Implement TODO 3 first.")
    if low_variance_resample(_P, _w, _rng) is not None:
        _cm = np.array([np.bincount(_rng.choice(_N, _N, p=_w), minlength=_N) for _ in range(2000)])
        _cl = np.array([np.bincount(low_variance_resample(_P, _w, _rng)[:, 0].astype(int), minlength=_N) for _ in range(2000)])
        _reference = {
            "Particle": [f"P{i+1}" for i in range(_N)],
            "Weight": _w,
            "Expected copies (N*w)": _N * _w,
        }
        _tab_mult = pd.DataFrame({
            **_reference,
            "Mean copies": _cm.mean(0),
            "Std copies": _cm.std(0),
            "P(lost)": (_cm == 0).mean(0),
        }).round(3)
        _tab_systematic = pd.DataFrame({
            **_reference,
            "Mean copies": _cl.mean(0),
            "Std copies": _cl.std(0),
            "P(lost)": (_cl == 0).mean(0),
        }).round(3)
        _out = mo.vstack([
            mo.md("### Copies per particle over 2000 trials (N = 10)"),
            mo.md("**Method 1 — Multinomial resampling (provided)**"),
            _tab_mult,
            mo.md("**Method 2 — Low-variance resampling (your TODO 3)**"),
            _tab_systematic,
        ])
    _out
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    #### ✍️ ANSWER 3

    **3.1** Compare the multinomial and low-variance resampling methods. Both produce approximately the expected number of copies on average. How do they differ in terms of standard deviation and probability of losing particles? Why does this matter for a Particle Filter? Look at the probability of losing a particle with weight 0.10–0.15.

    > *Your answer…*

    **3.2** Should we resample after every measurement? What quantity do we use to decide? How can the effective sample size ($N_{\mathrm{eff}}$) help us decide when to resample?

    > *Your answer…*
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ---
    ## 4 · Monte Carlo Localization on MRCLAM (≈ 25 min)

    The provided `run_mcl` assembles your functions into MCL for **position tracking**: all particles start at the known initial pose. The estimate is the weighted mean (circular mean for $\theta$), recorded at the same odometry instants as dead reckoning so that both are evaluated on the same times.
    """
    )
    return


@app.cell(hide_code=True)
def _(
    effective_sample_size,
    low_variance_resample,
    measurement_log_likelihood,
    normalize_log_weights,
    np,
    sample_motion_model,
):
    # PROVIDED — MCL (tracking). resample: "neff" (when N_eff < N/2) or "always" (after every measurement)
    def run_mcl(run, N, alphas, sigma, resample="neff", seed=0):
        if normalize_log_weights(np.zeros(2)) is None or effective_sample_size(np.ones(2) / 2) is None:
            return None, None
        _rng = np.random.default_rng(seed)
        _P = np.tile(run["x0"], (N, 1))
        _logw = np.zeros(N)
        _t, _u = run["t0"], (0.0, 0.0)
        _out, _neff = [[_t, *run["x0"]]], []
        _nres = 0

        def _mean(w):
            return np.array([w @ _P[:, 0], w @ _P[:, 1], np.arctan2(w @ np.sin(_P[:, 2]), w @ np.cos(_P[:, 2]))])

        for _e in run["events"]:
            if _e[0] > _t:
                _P = sample_motion_model(_P, _u[0], _u[1], _e[0] - _t, alphas, _rng)
                if _P is None:
                    return None, None
                _t = _e[0]
            if _e[1] == 0:
                _u = (_e[2], _e[3])
                if _t > _out[-1][0]:  # one estimate per odometry instant, as dead reckoning
                    _out.append([_t, *_mean(normalize_log_weights(_logw))])
                continue
            _ll = measurement_log_likelihood(_P, (_e[3], _e[4]), run["landmarks"][_e[2]], sigma)
            if _ll is None:
                return None, None
            _logw = _logw + _ll
            _w = normalize_log_weights(_logw)
            _ne = effective_sample_size(_w)
            _neff.append([_t - run["t0"], _ne])
            if resample == "always" or _ne < N / 2:
                _P = low_variance_resample(_P, _w, _rng)
                if _P is None:
                    return None, None
                _logw = np.zeros(N)
                _nres += 1
        return np.array(_out), {"neff": np.asarray(_neff).reshape(-1, 2), "resamplings": _nres}
    return (run_mcl,)


@app.cell(hide_code=True)
def _(mo):
    mcl_N = mo.ui.slider(20, 3000, step=10, value=500, label="N particles", show_value=True)
    mcl_sr = mo.ui.slider(0.02, 1.0, step=0.01, value=0.3, label="σ_r (m)", show_value=True)
    mcl_sp = mo.ui.slider(0.02, 1.0, step=0.01, value=0.3, label="σ_φ (rad)", show_value=True)
    mcl_policy = mo.ui.dropdown({"when N_eff < N/2": "neff", "after every measurement": "always"}, value="when N_eff < N/2", label="Resample")
    mcl_btn = mo.ui.run_button(label="Run MCL")
    mo.vstack([mo.hstack([mcl_N, mcl_sr, mcl_sp, mcl_policy]), mcl_btn])
    return mcl_N, mcl_btn, mcl_policy, mcl_sp, mcl_sr


@app.cell(hide_code=True)
def _(
    dead_reckoning,
    evaluate,
    go,
    l3_alphas,
    mcl_N,
    mcl_btn,
    mcl_policy,
    mcl_sp,
    mcl_sr,
    mo,
    pd,
    plot_errors,
    plot_trajectories,
    run,
    run_mcl,
    time,
):
    mo.stop(not mcl_btn.value, mo.md("Choose the settings and press **Run MCL** (≈ 1–5 s)."))
    _t0 = time.time()
    _s, _i = run_mcl(run, mcl_N.value, l3_alphas, (mcl_sr.value, mcl_sp.value), mcl_policy.value)
    if _s is None:
        _out = mo.md("⏳ Implement TODOs 1–3 first.")
    else:
        _dr = dead_reckoning(run)
        _tab = pd.DataFrame(
            [
                {"method": "dead reckoning", **evaluate(_dr, run["gt"])},
                {"method": f"MCL N={mcl_N.value} ({mcl_policy.value})", **evaluate(_s, run["gt"]), "resamplings": _i["resamplings"], "time (s)": round(time.time() - _t0, 1)},
            ]
        ).round(3)
        _fn = go.Figure(go.Scatter(x=_i["neff"][:, 0], y=_i["neff"][:, 1], mode="lines"))
        _fn.add_hline(y=mcl_N.value / 2, line_dash="dash")
        _fn.update_layout(title="N_eff after each measurement", xaxis_title="time (s)", height=300, plot_bgcolor="white")
        _est = {"DR": _dr, "MCL": _s}
        _out = mo.vstack([_tab, mo.ui.plotly(plot_trajectories(run, _est, title="DR vs MCL")), mo.ui.plotly(plot_errors(run, _est)), mo.ui.plotly(_fn)])
    _out
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    #### ✍️ ANSWER 4

    **4.1** Compare MCL with dead reckoning (ATE, FTE, heading). Then compare the two resampling policies with the same $N$: which one is better, and how many resampling steps does each perform?

    > *Your answer…*

    **4.2** Change $N$ (e.g. 20, 100, 500, 2000). How do accuracy and run time change? What happens with very few particles?

    > *Your answer…*



    ---
    **Keep this notebook.** In E1 you will paste `sample_motion_model`, `measurement_log_likelihood`, `normalize_log_weights`, `effective_sample_size` and `low_variance_resample`.
    """
    )
    return


if __name__ == "__main__":
    app.run()
