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
    # L4 — EKF-SLAM with Known Correspondences

    **11765 · Navigation and Environment Modelling in Mobile Robotics · MUSI** · Class exercise, Friday 30/10/2026

    **Formative** exercise (not graded). The four functions you write here (`slam_predict`, `h_slam`/`H_slam`, `add_landmark`, `slam_update`) are reused in **E2**: keep your completed notebook.

    ### Learning objectives
    1. Build the joint state of robot pose and map and propagate it through the motion model **without touching the map**.
    2. Write the range–bearing model and its (sparse) Jacobian with respect to the **whole** state.
    3. Add a newly observed landmark to the state with a **correctly initialised covariance** (inverse observation model).
    4. Run EKF-SLAM on real data and evaluate both the trajectory and the map against ground truth.
    5. Explain, with your own numbers, the structure of the SLAM posterior: correlations, monotonic convergence of the map, the effect of losing the measurements and re-observing old landmarks, and the computational cost.

    ### Recommended reading
    Thrun, Burgard & Fox, *Probabilistic Robotics*: Ch. 10, Sec. 10.1–10.2 (EKF-SLAM with known correspondences, Table 10.1) · Siegwart, Nourbakhsh & Scaramuzza, *Introduction to Autonomous Mobile Robots*, 2nd ed.: Sec. 5.8 (SLAM) · H. Durrant-Whyte & T. Bailey, "Simultaneous localization and mapping: Part I", *IEEE Robotics & Automation Magazine* 13(2), 2006 · M. W. M. G. Dissanayake *et al.*, "A solution to the simultaneous localization and map building (SLAM) problem", *IEEE Trans. Robotics and Automation* 17(3), 2001.

    ### How to work
    - **🛠 TODO** cells: write code. **✅ CHECK** cells: automatic feedback on your code. **✍️ ANSWER** cells: double-click and write a short answer **with numbers from your own runs**.
    - In class (2 h) the priority is Parts 0–3 with every ✅ CHECK green. Parts 4–6 contain no new code, only provided experiments and questions: complete them at home before the E2 session (Friday 06/11), which starts from them.
    - Ground truth is used only for the **initial pose** and for **evaluation**. In SLAM this also applies to the landmark positions: the filter never reads `run["landmarks"]`.
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

    def evaluate(states, gt):
        """ATE, FTE and final heading error of an estimate [t, x, y, theta].
        ATE (Absolute Trajectory Error): RMS position error over the whole trajectory.
        FTE (Final Trajectory Error): position error at the last instant.
        Ground truth is interpolated at the estimate timestamps."""
        _g = gt_at(gt, states[:, 0])
        _e = np.hypot(states[:, 1] - _g[:, 0], states[:, 2] - _g[:, 1])
        _eth = (states[-1, 3] - _g[-1, 2] + np.pi) % (2 * np.pi) - np.pi
        return {
            "ATE (m)": float(np.sqrt(np.mean(_e**2))),
            "FTE (m)": float(_e[-1]),
            "final heading error (deg)": float(abs(np.degrees(_eth))),
        }

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

        _p = _vec @ (
            k
            * np.sqrt(np.maximum(_lam, 0))[:, None]
            * np.stack([np.cos(_phi), np.sin(_phi)])
        )

        return mean[0] + _p[0], mean[1] + _p[1]


    def subsample_for_plot(data, max_points=1500):
        """
        Keep at most max_points for visualization only.

        This does NOT modify the data used by the filter or by the evaluation
        metrics; it only reduces the number of points sent to Plotly/Marimo.
        """
        if data is None or len(data) <= max_points:
            return data

        # Evenly spaced indices, preserving both the first and last point.
        _idx = np.linspace(0, len(data) - 1, max_points, dtype=int)
        return data[_idx]


    def plot_slam(run, res, title="EKF-SLAM", extra=None):
        """
        Ground truth trajectory + landmarks (★), SLAM trajectory and SLAM map
        (✕ with 3σ ellipses).

        A thin dotted line joins each estimated landmark to its ground-truth
        position.

        extra: dict name -> array [t, x, y, theta] of other trajectories
               (e.g. dead reckoning).

        Long trajectories are subsampled for plotting only.
        """
        _fig = go.Figure()

        # Ground-truth trajectory — subsampled only for visualization
        _gt = subsample_for_plot(run["gt"])

        _fig.add_trace(
            go.Scatter(
                x=_gt[:, 1],
                y=_gt[:, 2],
                mode="lines",
                name="ground truth",
                line=dict(color="#1f77b4", dash="dot"),
            )
        )

        # Ground-truth landmarks
        _bcs = list(run["landmarks"].keys())

        _fig.add_trace(
            go.Scatter(
                x=[run["landmarks"][b][0] for b in _bcs],
                y=[run["landmarks"][b][1] for b in _bcs],
                mode="markers+text",
                name="landmarks (ground truth)",
                text=[landmark_label(run, b) for b in _bcs],
                textposition="top center",
                textfont=dict(size=11, color="#555555"),
                hovertext=[
                    f"<b>{landmark_label(run, b)}</b> "
                    f"(barcode {int(b)}) — ground truth"
                    for b in _bcs
                ],
                hoverinfo="text",
                marker=dict(
                    symbol="star",
                    size=11,
                    color="gold",
                    line=dict(width=0.5, color="#8a6d00"),
                ),
            )
        )

        # Optional extra trajectories, e.g. dead reckoning
        for _i, (_name, _s) in enumerate((extra or {}).items()):
            if _s is None:
                continue

            _s_plot = subsample_for_plot(_s)

            _fig.add_trace(
                go.Scatter(
                    x=_s_plot[:, 1],
                    y=_s_plot[:, 2],
                    mode="lines",
                    name=_name,
                    line=dict(
                        color=_COLORS[(_i + 1) % len(_COLORS)],
                        width=1,
                    ),
                )
            )

        # EKF-SLAM result
        if res is not None:
            _s_plot = subsample_for_plot(res["states"])

            _fig.add_trace(
                go.Scatter(
                    x=_s_plot[:, 1],
                    y=_s_plot[:, 2],
                    mode="lines",
                    name="SLAM trajectory",
                    line=dict(color=_COLORS[0]),
                )
            )

            _mu = res["mu"]
            _Sig = res["Sigma"]

            _ex, _ey, _lab = [], [], []

            for _j, _b in enumerate(res["ids"]):
                _m = _mu[3 + 2 * _j : 5 + 2 * _j]

                _ex.append(_m[0])
                _ey.append(_m[1])
                _lab.append(landmark_label(run, _b))

                # 3σ landmark uncertainty ellipse
                _P = _Sig[
                    3 + 2 * _j : 5 + 2 * _j,
                    3 + 2 * _j : 5 + 2 * _j,
                ]

                _x, _y = ellipse_xy(_m, _P)

                _fig.add_trace(
                    go.Scatter(
                        x=_x,
                        y=_y,
                        mode="lines",
                        line=dict(color="#d62728", width=1),
                        showlegend=False,
                        hoverinfo="skip",
                    )
                )

                # Line from estimated landmark to ground truth
                if _b in run["landmarks"]:
                    _g = run["landmarks"][_b]

                    _fig.add_trace(
                        go.Scatter(
                            x=[_m[0], _g[0]],
                            y=[_m[1], _g[1]],
                            mode="lines",
                            line=dict(
                                color="gray",
                                width=1,
                                dash="dot",
                            ),
                            showlegend=False,
                            hoverinfo="skip",
                        )
                    )

            # Estimated landmarks
            _fig.add_trace(
                go.Scatter(
                    x=_ex,
                    y=_ey,
                    mode="markers+text",
                    name="SLAM map (3σ)",
                    text=_lab,
                    textposition="bottom center",
                    textfont=dict(size=10, color="#d62728"),
                    marker=dict(
                        symbol="x",
                        size=9,
                        color="#d62728",
                    ),
                    hovertext=[
                        f"<b>{_label}</b> — estimate"
                        for _label in _lab
                    ],
                    hoverinfo="text",
                )
            )

        _fig.update_layout(
            title=title,
            height=560,
            plot_bgcolor="white",
            xaxis=dict(
                title="x (m)",
                constrain="domain",
            ),
            yaxis=dict(
                title="y (m)",
                scaleanchor="x",
                scaleratio=1,
            ),
        )

        return _fig


    def plot_errors(run, estimates):
        """
        Position and heading error vs time for several estimates
        [t, x, y, theta].

        Trajectories are subsampled for plotting only.
        """
        _fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            subplot_titles=(
                "Position error (m)",
                "Heading error (deg)",
            ),
        )

        for _i, (_name, _s) in enumerate(estimates.items()):
            if _s is None:
                continue

            # Only reduce points for visualization
            _s_plot = subsample_for_plot(_s)

            # Ground truth interpolated exactly at the plotted timestamps
            _g = gt_at(run["gt"], _s_plot[:, 0])

            _t = _s_plot[:, 0] - run["t0"]

            # Position error
            _pos_err = np.hypot(
                _s_plot[:, 1] - _g[:, 0],
                _s_plot[:, 2] - _g[:, 1],
            )

            _fig.add_trace(
                go.Scatter(
                    x=_t,
                    y=_pos_err,
                    mode="lines",
                    name=_name,
                    line=dict(
                        color=_COLORS[_i % len(_COLORS)]
                    ),
                ),
                row=1,
                col=1,
            )

            # Heading error, wrapped to [-pi, pi)
            _heading_err = (
                _s_plot[:, 3]
                - _g[:, 2]
                + np.pi
            ) % (2 * np.pi) - np.pi

            _fig.add_trace(
                go.Scatter(
                    x=_t,
                    y=np.degrees(_heading_err),
                    mode="lines",
                    name=_name,
                    showlegend=False,
                    line=dict(
                        color=_COLORS[_i % len(_COLORS)]
                    ),
                ),
                row=2,
                col=1,
            )

        _fig.update_xaxes(
            title_text="time (s)",
            row=2,
            col=1,
        )

        _fig.update_layout(
            height=480,
            plot_bgcolor="white",
        )

        return _fig
    return landmark_label, plot_errors, plot_slam


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Provided functions from L1–L3

    `wrap_angle`, `motion_euler`, `G_jacobian`, `V_jacobian` and `control_noise` are the solutions already distributed with L3 (Euler model, Thrun's $\alpha$ convention $M=\operatorname{diag}(\alpha_1v^2+\alpha_2\omega^2,\ \alpha_3v^2+\alpha_4\omega^2)$). `num_jacobian` (central finite differences) is used **only by the CHECK cells**.
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
    ## 0 · Your measurement model from L2

    ### 🛠 TODO 0 — `h_landmark(mu, landmark)` and `H_landmark(mu, landmark)`
    Paste your tested measurement model from **L2**, keeping the same names and signatures.

    **Inputs (both functions)**
    - `mu`: robot pose `[x, y, theta]`, shape `(3,)`; position in metres, heading in radians.
    - `landmark`: known landmark position `[mx, my]`, shape `(2,)`, in metres.

    **Returns**
    - `h_landmark`: expected measurement `[range, bearing]`, shape `(2,)`; range in metres and bearing in radians, wrapped to $[-\pi,\pi)$.
    - `H_landmark`: measurement Jacobian with respect to the **robot pose**, shape `(2, 3)`; its rows correspond to range and bearing.
    """
    )
    return


@app.cell
def _():
    def h_landmark(mu, landmark):
        return None

    def H_landmark(mu, landmark):
        return None
    return H_landmark, h_landmark


@app.cell(hide_code=True)
def _(H_landmark, h_landmark, mo, np, num_jacobian):
    def _check():
        try:
            _mu = np.array([1.0, -0.5, 0.4])
            _lm = (3.0, 1.0)
            _r = [
                ("h_landmark", np.allclose(h_landmark(_mu, _lm), [np.hypot(2.0, 1.5), np.arctan2(1.5, 2.0) - 0.4])),
                ("h_landmark wraps the bearing", abs(h_landmark(np.array([0.0, 0.0, np.pi - 0.1]), (-1.0, -0.1))[1]) < np.pi),
                ("H_landmark = numerical Jacobian", np.allclose(H_landmark(_mu, _lm), num_jacobian(lambda x: h_landmark(x, _lm), _mu), atol=1e-6)),
            ]
        except Exception as _e:
            return f"⏳ not implemented yet ({type(_e).__name__})"
        return "<br>".join(f"{'✅' if _ok else '❌'} {_n}" for _n, _ok in _r)

    mo.md("**✅ CHECK 0**<br>" + _check())
    return


@app.cell(hide_code=True)
def _(mo):
    # Dataset1 is reserved for E1 and Dataset9 (barriers) is not used in L4/E2.
    dataset_dd = mo.ui.dropdown([f"MRCLAM_Dataset{_j}" for _j in range(2, 9)], value="MRCLAM_Dataset4", label="Dataset")
    robot_dd = mo.ui.dropdown([f"Robot{_j}" for _j in range(1, 6)], value="Robot3", label="Robot")
    duration_sl = mo.ui.slider(30, 1200, step=30, value=300, label="Duration (s)", show_value=True)
    mo.hstack([dataset_dd, robot_dd, duration_sl], justify="start")
    return dataset_dd, duration_sl, robot_dd


@app.cell(hide_code=True)
def _(dataset_dd, duration_sl, load_run, mo, path_length, robot_dd):
    run = load_run(dataset_dd.value, robot_dd.value, duration_sl.value)
    mo.md(
        f"**{run['dataset']} / {run['robot']}** — {run['gt'][-1, 0] - run['gt'][0, 0]:.0f} s · "
        f"path {path_length(run['gt']):.1f} m · {len(run['odom'])} odometry commands · "
        f"{len(run['meas'])} landmark measurements of {len(set(run['meas'][:, 1]))} different landmarks · "
        f"{run['n_robot_meas']} measurements of other robots (discarded)"
    )
    return (run,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ---
    ## 1 · The SLAM state and the prediction step

    In EKF-SLAM the state $\mathbf x$ contains the robot pose $\mathbf x_r$ **and** the map $\mathbf m$:

    $$
    \mathbf x=\big(\underbrace{x,\ y,\ \theta}_{\mathbf x_r},\ \underbrace{m_{0,x},\ m_{0,y},\ \dots,\ m_{n-1,x},\ m_{n-1,y}}_{\mathbf m}\big)^\top\in\mathbb R^{3+2n},
    \qquad
    \Sigma=\begin{pmatrix}\Sigma_{rr}&\Sigma_{rm}\\ \Sigma_{mr}&\Sigma_{mm}\end{pmatrix}.
    $$

    In the code, `mu` is the mean of $\mathbf x$ and `mu[:3]` the mean of $\mathbf x_r$. Landmark $j$ occupies entries `3+2j` and `4+2j`. The state **grows**: a landmark gets the next free slot the first time it is observed (Part 2), so slot $j$ is *not* the barcode; the driver keeps the barcode → slot table.

    **Prediction.** Landmarks are static, so only the robot moves: $\bar{\mathbf x}=\big(g(\mathbf x_r,u),\,\mathbf m\big)$. The Jacobian of the full transition is $\operatorname{blockdiag}(G,I_{2n})$ and the control noise enters only through the robot, so

    $$
    \bar\Sigma_{rr}=G\,\Sigma_{rr}\,G^\top+V M V^\top,\qquad
    \bar\Sigma_{rm}=G\,\Sigma_{rm},\qquad
    \bar\Sigma_{mr}=\bar\Sigma_{rm}^\top,\qquad
    \bar\Sigma_{mm}=\Sigma_{mm}.
    $$

    The robot keeps its correlations with the map: they are transported by $G$, and no noise is added to them because the control noise is independent of the map.

    ### 🛠 TODO 1 — `slam_predict(mu, Sigma, v, w, dt, alphas)`

    **Inputs**
    - `mu`: current joint robot–map mean, shape `(3+2n,)`; the first three entries are `[x, y, theta]`.
    - `Sigma`: current joint covariance, shape `(3+2n, 3+2n)`; $n$ is the number of mapped landmarks (possibly zero).
    - `v`, `w`: linear velocity (m/s) and angular velocity (rad/s).
    - `dt`: time elapsed since the previous event, in seconds.
    - `alphas`: four motion-noise coefficients $(\alpha_1,\alpha_2,\alpha_3,\alpha_4)$ used by `control_noise`.

    **Returns**
    - `mu_bar`: predicted joint mean, same shape as `mu`; the robot moves, while landmark positions stay unchanged.
    - `Sigma_bar`: predicted joint covariance, same shape as `Sigma`; robot uncertainty and robot–map correlations are propagated.

    **Implementation:** use `motion_euler`, `G_jacobian`, `V_jacobian` and `control_noise`, evaluated at the **previous** robot pose. Apply the block equations above without constructing the full-state Jacobian. Do not modify the inputs.
    """
    )
    return


@app.function
def slam_predict(mu, Sigma, v, w, dt, alphas):
    return None, None


@app.cell(hide_code=True)
def _(G_jacobian, V_jacobian, control_noise, mo, motion_euler, np):
    def _check():
        try:
            _rng = np.random.default_rng(0)
            _n = 3 + 2 * 4
            _A = _rng.normal(size=(_n, _n))
            _Sigma = 0.01 * _A @ _A.T + 0.01 * np.eye(_n)
            _mu = np.concatenate([[1.0, 2.0, 0.7], _rng.uniform(-3, 3, _n - 3)])
            _mu0, _Sigma0 = _mu.copy(), _Sigma.copy()
            _u, _dt, _a = (0.3, 0.2), 0.1, (0.1, 0.01, 0.01, 0.1)
            _mb, _Sb = slam_predict(_mu, _Sigma, _u[0], _u[1], _dt, _a)
            # reference: full-size Jacobian and noise mapped with F_x (Thrun, Table 10.1)
            _Fx = np.zeros((3, _n))
            _Fx[:, :3] = np.eye(3)
            _Gf = np.eye(_n) + _Fx.T @ (G_jacobian(_mu[:3], *_u, _dt) - np.eye(3)) @ _Fx
            _V = V_jacobian(_mu[:3], *_u, _dt)
            _Sref = _Gf @ _Sigma @ _Gf.T + _Fx.T @ _V @ control_noise(*_u, _a) @ _V.T @ _Fx
            _r = [
                ("shapes (3+2n,) and (3+2n, 3+2n)", np.shape(_mb) == (_n,) and np.shape(_Sb) == (_n, _n)),
                ("robot mean = motion model; landmarks unchanged", np.allclose(_mb[:3], motion_euler(_mu[:3], *_u, _dt)) and np.allclose(_mb[3:], _mu[3:])),
                ("covariance = full-matrix formula (Thrun, Table 10.1)", np.allclose(_Sb, _Sref, atol=1e-12)),
                ("map block Σ_mm unchanged", np.allclose(_Sb[3:, 3:], _Sigma[3:, 3:])),
                ("inputs not modified", np.array_equal(_mu, _mu0) and np.array_equal(_Sigma, _Sigma0)),
            ]
        except Exception as _e:
            return f"⏳ not implemented yet ({type(_e).__name__})"
        return "<br>".join(f"{'✅' if _ok else '❌'} {_n}" for _n, _ok in _r)

    mo.md("**✅ CHECK 1**<br>" + _check())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ---
    ## 2 · Measurements of mapped landmarks and new landmarks

    ### 2.1 The measurement model on the full state

    The range–bearing model of L2 is unchanged, but the landmark is now **part of the state** instead of a known constant. For landmark slot $j$, with $\Delta x=m_{j,x}-x$, $\Delta y=m_{j,y}-y$, $q=\Delta x^2+\Delta y^2$:

    $$
    \hat{\mathbf z}=h(\mathbf x_r,\mathbf m_j)=\begin{pmatrix}\sqrt q\\ \operatorname{atan2}(\Delta y,\Delta x)-\theta\end{pmatrix}.
    $$

    The Jacobian with respect to the whole state is $2\times(3+2n)$: one row per measured quantity ($\hat r$, $\hat\phi$), one column per state variable. The measurement depends only on the robot and on landmark $j$, so all other columns are zero:

    $$
    H_j=\big[\,H_r\;\big|\;0\cdots0\;\big|\;H_{m_j}\;\big|\;0\cdots0\,\big],\qquad
    H_r=\begin{pmatrix}-\dfrac{\Delta x}{\sqrt q}&-\dfrac{\Delta y}{\sqrt q}&0\\[6pt] \dfrac{\Delta y}{q}&-\dfrac{\Delta x}{q}&-1\end{pmatrix},\qquad
    H_{m_j}=\begin{pmatrix}\dfrac{\Delta x}{\sqrt q}&\dfrac{\Delta y}{\sqrt q}\\[6pt] -\dfrac{\Delta y}{q}&\dfrac{\Delta x}{q}\end{pmatrix}.
    $$

    $H_r$ is the L2 Jacobian. Since the measurement depends on $\mathbf m_j-(x,y)$, moving the landmark has the opposite effect of moving the robot: $H_{m_j}=-H_r[:,0{:}2]$. For a state with three landmarks and an observation of slot 1:

    | | $x$ | $y$ | $\theta$ | $m_{0,x}$ | $m_{0,y}$ | $m_{1,x}$ | $m_{1,y}$ | $m_{2,x}$ | $m_{2,y}$ |
    |---|---|---|---|---|---|---|---|---|---|
    | $\hat r$ | ✱ | ✱ | 0 | 0 | 0 | ✱ | ✱ | 0 | 0 |
    | $\hat\phi$ | ✱ | ✱ | −1 | 0 | 0 | ✱ | ✱ | 0 | 0 |

    A zero column does **not** mean that the other landmarks cannot be corrected: they are corrected through their correlations in $\bar\Sigma$ (Part 3).

    ### 🛠 TODO 2 — `h_slam(mu, j)` and `H_slam(mu, j)`

    **Inputs (both functions)**
    - `mu`: joint robot–map mean, shape `(3+2n,)`, containing the robot pose followed by all mapped landmark positions.
    - `j`: **zero-based landmark slot** in `mu` (not the barcode); landmark `j` is stored at `mu[3+2*j : 5+2*j]`.

    **Returns**
    - `h_slam`: expected `[range, bearing]` to landmark `j`, shape `(2,)`; metres and wrapped radians.
    - `H_slam`: measurement Jacobian with respect to the **entire joint state**, shape `(2, 3+2n)`; only the robot and landmark `j` columns may be non-zero.

    Reuse your L2 `h_landmark` and `H_landmark` functions.
    """
    )
    return


@app.cell
def _():
    def h_slam(mu, j):
        return None

    def H_slam(mu, j):
        return None
    return H_slam, h_slam


@app.cell(hide_code=True)
def _(H_slam, h_slam, mo, np, num_jacobian):
    def _check():
        try:
            _mu = np.array([1.0, -0.5, 0.4, 3.0, 1.0, -1.0, 2.0, 0.5, -2.5])
            _J = num_jacobian(lambda y: h_slam(y, 1), _mu)
            _H = H_slam(_mu, 1)
            _r = [
                ("h_slam uses the landmark stored in slot j", np.allclose(h_slam(_mu, 1), [np.hypot(-2.0, 2.5), np.arctan2(2.5, -2.0) - 0.4])),
                ("H_slam shape (2, 3+2n)", np.shape(_H) == (2, _mu.size)),
                ("H_slam = numerical Jacobian", np.allclose(_H, _J, atol=1e-6)),
                ("zeros in the columns of the other landmarks", np.allclose(_H[:, 3:5], 0) and np.allclose(_H[:, 7:9], 0)),
            ]
        except Exception as _e:
            return f"⏳ not implemented yet ({type(_e).__name__})"
        return "<br>".join(f"{'✅' if _ok else '❌'} {_n}" for _n, _ok in _r)

    mo.md("**✅ CHECK 2**<br>" + _check())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### 2.2 A landmark seen for the first time: state augmentation

    The first observation $\mathbf z=(r,\phi)$ of a landmark is not used to correct anything: it **creates** the landmark. From the current robot pose, the inverse observation model gives its position

    $$
    \mathbf m=f(\mathbf x_r,\mathbf z)=\begin{pmatrix}x+r\cos(\theta+\phi)\\ y+r\sin(\theta+\phi)\end{pmatrix},
    $$

    which is appended to the mean: $\mathbf x_{\text{new}}=(\mathbf x,\ \mathbf m)$. The covariance grows by two rows and two columns. $\mathbf m$ depends on the robot pose (uncertain, and correlated with the whole map) and on the measurement (noise $Q$, independent of everything else). Linearising $f$ with

    $$
    G_r=\frac{\partial f}{\partial\mathbf x_r}\ (2\times3),\qquad G_z=\frac{\partial f}{\partial\mathbf z}\ (2\times2),
    $$

    (not to be confused with the motion Jacobian $G$ of Part 1) the augmented covariance must contain:
    - the **new landmark block**, with the robot uncertainty mapped through $G_r$ plus the measurement uncertainty mapped through $G_z$;
    - the **cross-covariance** between the new landmark and every existing variable, which comes only from the robot: use the robot rows $\Sigma[0{:}3,\,:]$ and $G_r$;
    - the old covariance, unchanged.

    ### 🛠 TODO 3 — `add_landmark(mu, Sigma, z, Q)`

    **Inputs**
    - `mu`, `Sigma`: current joint mean and covariance, with shapes `(3+2n,)` and `(3+2n, 3+2n)`.
    - `z`: first observation `[range, bearing]` of a **new** landmark, shape `(2,)`; metres and radians.
    - `Q`: range–bearing measurement-noise covariance, shape `(2, 2)`.

    **Returns**
    - `mu_new`: joint mean after **appending** the new landmark position, shape `(3+2n+2,)`.
    - `Sigma_new`: augmented covariance, shape `(3+2n+2, 3+2n+2)`, including the new landmark's uncertainty and its correlations with the existing state.

    **Implementation:** derive $G_r$ and $G_z$ from the inverse observation model above. Preserve the existing state and covariance, keep the result symmetric, and do not modify the inputs. The driver calls `add_landmark` **instead of** `slam_update` on a first observation (see question 3.1).
    """
    )
    return


@app.function
def add_landmark(mu, Sigma, z, Q):
    return None, None


@app.cell(hide_code=True)
def _(mo, np, num_jacobian):
    def _check():
        try:
            _rng = np.random.default_rng(1)
            _mu = np.array([0.5, -0.3, 0.7, 2.0, 1.0, -1.0, 3.0])
            _A = _rng.normal(size=(7, 7))
            _Sigma = 0.01 * _A @ _A.T + 0.01 * np.eye(7)
            _Q = np.diag([0.1, 0.05]) ** 2
            _z = np.array([2.0, 0.3])
            _mu0, _Sigma0 = _mu.copy(), _Sigma.copy()
            _mn, _Sn = add_landmark(_mu, _Sigma, _z, _Q)

            def _inv(xz):
                return np.array([xz[0] + xz[3] * np.cos(xz[2] + xz[4]), xz[1] + xz[3] * np.sin(xz[2] + xz[4])])

            _J = num_jacobian(_inv, np.concatenate([_mu[:3], _z]))
            _Gr, _Gz = _J[:, :3], _J[:, 3:]
            _r = [
                ("shapes (3+2n+2,) and square", np.shape(_mn) == (9,) and np.shape(_Sn) == (9, 9)),
                ("new mean = inverse observation model; old entries kept", np.allclose(_mn[-2:], _inv(np.concatenate([_mu[:3], _z]))) and np.allclose(_mn[:7], _mu)),
                ("old covariance block unchanged", np.allclose(_Sn[:7, :7], _Sigma)),
                ("new landmark block G_r Σ_rr G_rᵀ + G_z Q G_zᵀ", np.allclose(_Sn[7:, 7:], _Gr @ _Sigma[:3, :3] @ _Gr.T + _Gz @ _Q @ _Gz.T, atol=1e-8)),
                ("cross-covariance G_r Σ[0:3, :] (and symmetric)", np.allclose(_Sn[7:, :7], _Gr @ _Sigma[:3, :], atol=1e-8) and np.allclose(_Sn, _Sn.T)),
                ("inputs not modified", np.array_equal(_mu, _mu0) and np.array_equal(_Sigma, _Sigma0)),
            ]
        except Exception as _e:
            return f"⏳ not implemented yet ({type(_e).__name__})"
        return "<br>".join(f"{'✅' if _ok else '❌'} {_n}" for _n, _ok in _r)

    mo.md("**✅ CHECK 3**<br>" + _check())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ---
    ## 3 · The correction step and a complete run

    A new observation $\mathbf z=(r,\phi)$ of a landmark **already in the state** (slot $j$) corrects the predicted belief $(\bar\mu,\bar\Sigma)$ exactly as in L2, but on the full state:

    | Step | Expression | Size |
    |---|---|---|
    | expected measurement | $\hat{\mathbf z}=h(\bar\mu,j)$ | $2$ |
    | innovation (wrap the bearing) | $\nu=\mathbf z-\hat{\mathbf z}$ | $2$ |
    | Jacobian | $H=H_j(\bar\mu)$ | $2\times(3+2n)$ |
    | innovation covariance | $S=H\bar\Sigma H^\top+Q$ | $2\times2$ |
    | Kalman gain | $K=\bar\Sigma H^\top S^{-1}$ | $(3+2n)\times2$ |
    | mean (wrap $\theta$) | $\mu=\bar\mu+K\nu$ | $3+2n$ |
    | covariance | $\Sigma=(I-KH)\bar\Sigma=\bar\Sigma-KSK^\top$ | $(3+2n)\times(3+2n)$ |

    **Why one observation moves the whole map.** $H$ has non-zero columns only for the robot and landmark $j$, but $K=\bar\Sigma H^\top S^{-1}$ multiplies it by the **full** covariance:

    $$
    \bar\Sigma H^\top=\bar\Sigma_{:,r}\,H_r^\top+\bar\Sigma_{:,m_j}\,H_{m_j}^\top .
    $$

    Row $i$ of $K$ is non-zero whenever variable $i$ is correlated with the robot or with $\mathbf m_j$: $H$ says which variables the measurement depends on, $\bar\Sigma$ says which variables are correlated with them, and $K$ combines both. This is the difference between localization (L2) and SLAM.

    The two expressions of the covariance are algebraically equal (because $KS=\bar\Sigma H^\top$) but do not have the same cost: Part 6 measures it.

    ### 🛠 TODO 4 — `slam_update(mu_bar, Sigma_bar, z, j, Q)`

    **Inputs**
    - `mu_bar`, `Sigma_bar`: predicted joint mean and covariance **before the measurement**, shapes `(3+2n,)` and `(3+2n, 3+2n)`.
    - `z`: observed `[range, bearing]` of a landmark **already in the map**, shape `(2,)` (metres, radians).
    - `j`: zero-based slot of that landmark in the joint state (not the barcode).
    - `Q`: range–bearing measurement-noise covariance, shape `(2, 2)`.

    **Returns**
    - `mu`: corrected joint mean, shape `(3+2n,)`.
    - `Sigma`: corrected joint covariance, shape `(3+2n, 3+2n)`.
    - `nu`: innovation (observed minus expected measurement), shape `(2,)`; wrap its bearing component.
    - `S`: innovation covariance, shape `(2, 2)`; uncertainty of the predicted measurement difference.
    - `K`: Kalman gain, shape `(3+2n, 2)`; distributes the correction across the joint state.

    **Implementation:** use `h_slam` and `H_slam` and the equations in the table above. Wrap the corrected robot heading, keep `Sigma` symmetric, and do not modify the inputs.
    """
    )
    return


@app.function
def slam_update(mu_bar, Sigma_bar, z, j, Q):
    return None, None, None, None, None


@app.cell(hide_code=True)
def _(H_slam, mo, np):
    def _check():
        try:
            _rng = np.random.default_rng(2)
            _mu = np.array([0.5, -0.3, 0.7, 2.0, 1.0, -1.0, 3.0])
            _A = _rng.normal(size=(7, 7))
            _Sigma = 0.01 * _A @ _A.T + 0.01 * np.eye(7)
            _Q = np.diag([0.1, 0.05]) ** 2
            _z = (2.2, 0.1)
            _mu0, _Sigma0 = _mu.copy(), _Sigma.copy()
            _m1, _S1, _nu, _Sin, _K = slam_update(_mu, _Sigma, _z, 1, _Q)
            _H = H_slam(_mu, 1)
            # a measurement whose bearing is close to ±π: the innovation must be wrapped
            _mw = np.array([0.0, 0.0, 0.0, -3.0, -0.05])
            _, _, _nuw, _, _ = slam_update(_mw, np.eye(5) * 0.01, (3.0, np.pi - 0.01), 0, _Q)
            # Check that the covariance remains valid and the inputs were preserved.
            _r = [
                ("S = H Σ̄ Hᵀ + Q and K = Σ̄ Hᵀ S⁻¹", np.allclose(_Sin, _H @ _Sigma @ _H.T + _Q) and np.allclose(_K, _Sigma @ _H.T @ np.linalg.inv(_Sin))),
                ("μ = μ̄ + Kν and Σ = (I − KH) Σ̄", np.allclose(_m1, _mu + _K @ _nu) and np.allclose(_S1, (np.eye(7) - _K @ _H) @ _Sigma, atol=1e-10)),
                ("bearing innovation wrapped", abs(_nuw[1]) < 0.5),
                ("updated covariance symmetric and positive semidefinite", np.allclose(_S1, _S1.T, atol=1e-10) and np.linalg.eigvalsh(_S1).min() > -1e-10),
                ("inputs not modified", np.array_equal(_mu, _mu0) and np.array_equal(_Sigma, _Sigma0)),
            ]
        except Exception as _e:
            return f"⏳ not implemented yet ({type(_e).__name__})"
        return "<br>".join(f"{'✅' if _ok else '❌'} {_n}" for _n, _ok in _r)

    mo.md("**✅ CHECK 4**<br>" + _check())
    return


@app.cell(hide_code=True)
def _(np, time):
    # PROVIDED — EKF-SLAM with known correspondences over the event stream.
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
        _map_history = []
        _tic = time.perf_counter()
        for _e in run["events"]:
            if _e[0] > _t:
                _mu, _Sigma = slam_predict(_mu, _Sigma, _u[0], _u[1], _e[0] - _t, alphas)
                _t = _e[0]
            if _e[1] == 0:
                _u = (_e[2], _e[3])
                # Store the estimated map at every odometry instant (used by the map-error plot of Part 5).
                # Ground truth is NOT used here.
                _map_history.append((
                    _t - run["t0"],
                    tuple(_ids),
                    _mu[3:].reshape(-1, 2).copy(),
                ))
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
                _info["updates"] += 1
            _Sigma = 0.5 * (_Sigma + _Sigma.T)
            if decouple:
                _D = np.zeros_like(_Sigma)
                _D[:3, :3] = _Sigma[:3, :3]
                for _k in range(len(_ids)):
                    _D[3 + 2 * _k: 5 + 2 * _k, 3 + 2 * _k: 5 + 2 * _k] = _Sigma[3 + 2 * _k: 5 + 2 * _k, 3 + 2 * _k: 5 + 2 * _k]
                _Sigma = _D
            _nobs[_b] = _nobs.get(_b, 0) + 1
            # map snapshot right after the measurement (no delay in the map-error plot)
            _map_history.append((_t - run["t0"], tuple(_ids), _mu[3:].reshape(-1, 2).copy()))
            _d = np.diag(_Sigma)[3:].reshape(-1, 2).sum(1)
            for _k, _bb in enumerate(_ids):
                _lm_std.setdefault(_bb, []).append((_t - run["t0"], float(np.sqrt(_d[_k]))))
            _states.append([_t, *_mu[:3]])
            _rcov.append(_Sigma[:3, :3].copy())
        _info["run time (s)"] = round(time.perf_counter() - _tic, 2)
        return dict(
            states=np.array(_states),
            robot_cov=np.array(_rcov),
            mu=_mu,
            Sigma=_Sigma,
            ids=_ids,
            lm_std=_lm_std,
            first_seen=_first,
            n_obs=_nobs,
            info=_info,
            map_history=_map_history,
        )

    def map_table(run, res, align_fn, landmark_label_fn):
        """Per-landmark evaluation of a SLAM map against the ground-truth map."""
        import pandas as _pd
        _bcs = [_b for _b in res["ids"] if _b in run["landmarks"]]
        if not _bcs:
            _columns = ["landmark", "barcode", "slot", "first seen (s)", "observations",
                        "error (m)", "3σ radius (m)", "Mahalanobis² (χ²₂: 99 % < 9.21)",
                        "error after alignment (m)"]
            _summary = {"map RMSE (m)": float("nan"),
                        "map RMSE after rigid alignment (m)": float("nan"),
                        "alignment rotation (deg)": float("nan"),
                        "alignment translation (m)": float("nan")}
            return _pd.DataFrame(columns=_columns), _summary
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
    ### 3.1 Run EKF-SLAM

    The robot starts at the ground-truth initial pose with (almost) zero uncertainty: this **defines the map frame**. The map starts empty. The filter uses two noise models:

    | Parameter | Enters | Meaning | Increasing it… |
    |---|---|---|---|
    | $\alpha_1$, $\alpha_2$ | $M_{11}=\alpha_1v^2+\alpha_2\omega^2$ (prediction) | speed error caused by translation / rotation | the robot position uncertainty grows faster |
    | $\alpha_3$, $\alpha_4$ | $M_{22}=\alpha_3v^2+\alpha_4\omega^2$ (prediction) | turn-rate error caused by translation / rotation | the heading uncertainty grows faster |
    | $\sigma_r$ | $Q=\operatorname{diag}(\sigma_r^2,\sigma_\phi^2)$ (correction and initialisation) | range noise | landmarks less precise radially; range measurements correct less |
    | $\sigma_\phi$ | $Q$ | bearing noise | landmarks less precise tangentially (more so when far); bearings correct less |

    $\alpha$ adds uncertainty only to the robot; it reaches the map through the initialisation of new landmarks and through the corrections. $Q$ weighs every correction and also sets part of the initial uncertainty of each new landmark. These parameters should describe the real errors: making them artificially large or small does not make the filter better, and can make it inconsistent.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    a1_sl = mo.ui.slider(0.0, 2.0, step=0.05, value=0.2, label="α1", show_value=True)
    a2_sl = mo.ui.slider(0.0, 0.5, step=0.01, value=0.02, label="α2", show_value=True)
    a3_sl = mo.ui.slider(0.0, 0.5, step=0.01, value=0.02, label="α3", show_value=True)
    a4_sl = mo.ui.slider(0.0, 8.0, step=0.1, value=0.5, label="α4", show_value=True)
    sr_sl = mo.ui.slider(0.01, 1.0, step=0.01, value=0.1, label="σ_r (m)", show_value=True)
    sp_sl = mo.ui.slider(0.01, 0.5, step=0.01, value=0.05, label="σ_φ (rad)", show_value=True)
    mo.vstack([
        mo.md("**Process noise** $\\alpha$ · **measurement noise** $Q$"),
        mo.hstack([a1_sl, a2_sl, a3_sl, a4_sl], justify="start"),
        mo.hstack([sr_sl, sp_sl], justify="start"),
    ])
    return a1_sl, a2_sl, a3_sl, a4_sl, sp_sl, sr_sl


@app.cell(hide_code=True)
def _(a1_sl, a2_sl, a3_sl, a4_sl, np, sp_sl, sr_sl):
    alphas = (a1_sl.value, a2_sl.value, a3_sl.value, a4_sl.value)
    Q = np.diag([sr_sl.value**2, sp_sl.value**2])
    return Q, alphas


@app.cell(hide_code=True)
def _(Q, alphas, dead_reckoning, evaluate, mo, np, pd, run, run_ekf_slam):
    slam_res = run_ekf_slam(run, alphas, Q)
    slam_dr = dead_reckoning(run)
    if slam_res is None:
        _out = mo.md("⏳ Implement TODO 1–4 first.")
    else:
        _out = mo.vstack([
            mo.ui.table(pd.DataFrame([
                {"method": "dead reckoning", **evaluate(slam_dr, run["gt"])},
                {"method": "EKF-SLAM", **evaluate(slam_res["states"][np.isin(slam_res["states"][:, 0], slam_dr[:, 0])], run["gt"])},
            ]).round(3), selection=None),
            mo.md("Both rows are evaluated at the same instants (the odometry timestamps). " + " · ".join(f"{_k}: **{_v}**" for _k, _v in slam_res["info"].items()) + f" · state dimension at the end: **{slam_res['mu'].size}**"),
        ])
    _out
    return slam_dr, slam_res


@app.cell(hide_code=True)
def _(mo, plot_errors, plot_slam, run, slam_dr, slam_res):
    mo.stop(slam_res is None)
    mo.vstack([
        mo.ui.plotly(plot_slam(run, slam_res, title=f"EKF-SLAM — {run['dataset']} / {run['robot']}", extra={"dead reckoning": slam_dr})),
        mo.ui.plotly(plot_errors(run, {"EKF-SLAM": slam_res["states"], "dead reckoning": slam_dr})),
    ])
    return


@app.cell(hide_code=True)
def _(align_map, landmark_label, map_table, mo, run, slam_res):
    mo.stop(slam_res is None)
    slam_map_df, slam_map_summary = map_table(run, slam_res, align_map, landmark_label)
    mo.vstack([
        mo.md("**Map evaluation** (ground truth used only here). *Error after alignment*: error remaining after the rigid rotation + translation that best fits the estimated map to the true one."),
        mo.ui.table(slam_map_df, selection=None, page_size=20),
        mo.md(" · ".join(f"{_k}: **{_v:.3f}**" for _k, _v in slam_map_summary.items())),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    #### ✍️ ANSWERS — Part 3

    **3.1** After `add_landmark`, the driver does **not** call `slam_update` with the same measurement. Why not? What would happen to the landmark mean and covariance if the same observation were processed twice?

    > *Your answer…*


    **3.2 — Map accuracy and consistency** (default parameters)
    1. Compare *map RMSE* with *map RMSE after rigid alignment* and the alignment rotation. What kind of error dominates the estimated map?
    2. How many landmarks have Mahalanobis² below 9.21? Is the uncertainty reported by the filter compatible with the errors it actually makes?
    3. Compare one landmark first observed near the beginning with one first observed much later: final error, reported uncertainty, number of observations. Does an early first sighting guarantee a small final error? Why can an already-mapped landmark still move during later corrections?

    > *Your answer…*
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ---
    ## 4 · Structure of the SLAM posterior

    Dissanayake *et al.* (2001) proved three properties of the **linear-Gaussian** SLAM problem (see also Thrun, Sec. 10.2.4, and Durrant-Whyte & Bailey 2006):

    1. the determinant of the covariance of any landmark (or set of landmarks) **never increases** when a measurement is processed;
    2. in the limit, the landmark estimates become **fully correlated**: the relative map becomes known exactly;
    3. in the limit, the covariance of any landmark is bounded below by the uncertainty the **robot** had when the **first** landmark was observed: the absolute map can never be more accurate than the initial pose.

    The figures below test these statements on the real (nonlinear) data. The table compares the correct filter with a deliberately wrong one that **throws away the correlations** after every measurement (`decouple=True`).
    """
    )
    return


@app.cell(hide_code=True)
def _(go, landmark_label, make_subplots, mo, np, run, slam_res):
    mo.stop(slam_res is None or not slam_res["ids"], mo.md("No mapped landmarks to analyse yet; choose a longer run."))

    import pandas as _pd

    # Visualization settings
    _MAX_POINTS_PER_LM = 300
    _COOBS_WINDOW_S = 2.0
    _REOBS_GAP_S = 30.0

    # Distinct landmark colours (reused in both panels)
    _cols = [
        "#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4",
        "#008b8b", "#b22222", "#bf00bf", "#807000", "#4682b4",
        "#8b4513", "#800080", "#006400", "#ff1493", "#000075",
    ]


    # ============================================================
    # 1 — CORRELATION MATRIX
    # ============================================================

    _Sig = slam_res["Sigma"]

    # rho_ij = Sigma_ij / sqrt(Sigma_ii * Sigma_jj)
    _std_state = np.sqrt(np.maximum(np.diag(_Sig), 1e-15))
    _rho = np.abs(_Sig / np.outer(_std_state, _std_state))

    # Label the components of the SLAM state x
    _names = ["x_r", "y_r", "θ_r"] + [
        f"{landmark_label(run, _b)}.{_c}"
        for _b in slam_res["ids"]
        for _c in ("x", "y")
    ]

    _heat = go.Figure(
        go.Heatmap(
            z=_rho,
            x=_names,
            y=_names,
            colorscale="Viridis",
            zmin=0,
            zmax=1,
            xgap=1,
            ygap=1,
            colorbar=dict(title="|ρ|"),
            hovertemplate=(
                "%{y} vs %{x}<br>"
                "|ρ| = %{z:.2f}<extra></extra>"
            ),
        )
    )

    _heat.update_layout(
        title="Absolute correlation between components of SLAM state x",
        width=710,
        height=700,
        margin=dict(l=90, r=40, t=75, b=110),
        xaxis=dict(
            title="Components of x (columns)",
            tickangle=-90,
            tickfont=dict(size=9),
            automargin=True,
        ),
        yaxis=dict(
            title="Components of x (rows)",
            autorange="reversed",
            tickfont=dict(size=9),
            automargin=True,
        ),
    )


    # ============================================================
    # 2 — UNCERTAINTY HISTORY: TWO SEPARATE PANELS
    # ============================================================

    _fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.68, 0.32],
        vertical_spacing=0.14,
        subplot_titles=(
            "Landmark position uncertainty over time",
            "Robot position uncertainty at each landmark's first sighting",
        ),
    )

    _ratios = []
    _incr = 0

    for _k, _b in enumerate(slam_res["ids"]):

        _h = np.asarray(slam_res["lm_std"][_b])
        _f = slam_res["first_seen"][_b]

        _c = _cols[_k % len(_cols)]
        _lab = landmark_label(run, _b)

        # Reduce plotting size only; keep all original data for metrics.
        _idx = np.linspace(
            0,
            len(_h) - 1,
            min(len(_h), _MAX_POINTS_PER_LM),
            dtype=int,
        )
        _hp = _h[_idx]

        # TOP: landmark position uncertainty history
        _fig.add_trace(
            go.Scatter(
                x=_hp[:, 0],
                y=_hp[:, 1],
                mode="lines",
                name=_lab,
                legendgroup=_lab,
                line=dict(color=_c, width=1.8),
                hovertemplate=(
                    f"{_lab}<br>"
                    "t = %{x:.1f} s<br>"
                    "landmark std = %{y:.3f} m"
                    "<extra></extra>"
                ),
            ),
            row=1,
            col=1,
        )

        # BOTTOM: robot position uncertainty at first sighting
        _fig.add_trace(
            go.Scatter(
                x=[_f["t"]],
                y=[_f["robot std (m)"]],
                mode="markers",
                name=_lab,
                legendgroup=_lab,
                showlegend=False,
                marker=dict(
                    size=12,
                    color=_c,
                    symbol="circle",
                    line=dict(color="black", width=0.5),
                ),
                hovertemplate=(
                    f"{_lab}<br>"
                    "first seen at t = %{x:.1f} s<br>"
                    "robot std = %{y:.3f} m"
                    "<extra></extra>"
                ),
            ),
            row=2,
            col=1,
        )

        # Summary statistics, using all original samples
        _ratios.append(
            _h[-1, 1] / max(_f["robot std (m)"], 1e-9)
        )

        _incr += int(
            np.any(np.diff(_h[:, 1]) > 1e-9)
        )


    # Top panel
    _fig.update_yaxes(
        type="log",
        title_text="√tr(Σ_jj) (m)",
        showgrid=True,
        gridcolor="#e5e7eb",
        row=1,
        col=1,
    )

    # Bottom panel
    _fig.update_yaxes(
        type="log",
        title_text="Robot √(Σxx + Σyy) (m)",
        showgrid=True,
        gridcolor="#e5e7eb",
        row=2,
        col=1,
    )

    _fig.update_xaxes(
        showgrid=True,
        gridcolor="#e5e7eb",
    )

    _fig.update_xaxes(
        title_text="Time since start (s)",
        row=2,
        col=1,
    )

    _fig.update_layout(
        title="Uncertainty: landmarks and robot at first sighting",
        height=680,
        width=950,
        plot_bgcolor="white",
        margin=dict(l=80, r=190, t=100, b=70),
        legend=dict(
            title="Landmarks",
            x=1.02,
            xanchor="left",
            y=1,
            yanchor="top",
            bgcolor="rgba(255,255,255,0.95)",
            bordercolor="#d0d0d0",
            borderwidth=1,
            font=dict(size=10),
        ),
    )


    # ============================================================
    # 3 — LONG-GAP RE-OBSERVATIONS
    # ============================================================

    # A long gap between sightings suggests a possible revisit.
    # It is a loop-closure CANDIDATE, not definitive proof.

    _meas = np.asarray(run["meas"])
    _reobs = []

    for _b in slam_res["ids"]:

        _times = np.sort(
            _meas[_meas[:, 1] == _b, 0]
        )

        _gaps = np.diff(_times)

        for _idx in np.flatnonzero(
            _gaps >= _REOBS_GAP_S
        ):
            _reobs.append({
                "landmark": landmark_label(run, _b),
                "previous sighting (s)": round(
                    float(_times[_idx] - run["t0"]), 1
                ),
                "re-observed (s)": round(
                    float(_times[_idx + 1] - run["t0"]), 1
                ),
                "gap (s)": round(float(_gaps[_idx]), 1),
            })

    # Show the eight longest gaps
    _reobs = sorted(
        _reobs,
        key=lambda d: -d["gap (s)"],
    )[:8]


    # ============================================================
    # 4 — CO-OBSERVATIONS AND LANDMARK CORRELATIONS
    # ============================================================

    # Two landmarks are called "co-observed" here if they appear
    # in at least one common 2-second time bin.
    # This is an approximate temporal definition.

    _ids = list(slam_res["ids"])

    _seen_bins = {
        int(_b): set()
        for _b in _ids
    }

    for _m in _meas:
        _b = int(_m[1])

        if _b in _seen_bins:
            _bin = int(
                np.floor(
                    (_m[0] - run["t0"]) / _COOBS_WINDOW_S
                )
            )
            _seen_bins[_b].add(_bin)


    _pairs = []

    for _i in range(len(_ids)):
        for _j in range(_i + 1, len(_ids)):

            _bi = _ids[_i]
            _bj = _ids[_j]

            # Number of time bins shared by both landmarks
            _shared = len(
                _seen_bins[int(_bi)]
                & _seen_bins[int(_bj)]
            )

            # Mean absolute correlation between the two
            # 2D landmark position blocks (2 x 2 entries).
            _pcorr = float(np.mean(
                _rho[
                    3 + 2*_i : 5 + 2*_i,
                    3 + 2*_j : 5 + 2*_j,
                ]
            ))

            _pairs.append({
                "pair": (
                    f"{landmark_label(run, _bi)} – "
                    f"{landmark_label(run, _bj)}"
                ),
                "shared 2 s bins": _shared,
                "mean |ρ|": round(_pcorr, 3),
            })


    # Most frequently co-observed landmark pairs
    _together = sorted(
        (
            _p for _p in _pairs
            if _p["shared 2 s bins"] > 0
        ),
        key=lambda p: (
            p["shared 2 s bins"],
            p["mean |ρ|"],
        ),
        reverse=True,
    )[:5]

    # Strongly correlated pairs never co-observed
    _apart = sorted(
        (
            _p for _p in _pairs
            if _p["shared 2 s bins"] == 0
        ),
        key=lambda p: p["mean |ρ|"],
        reverse=True,
    )[:5]

    _pairs_show = (
        [
            {"type": "co-observed", **_p}
            for _p in _together
        ]
        +
        [
            {"type": "not co-observed", **_p}
            for _p in _apart
        ]
    )

    _inter_corr = [
        _p["mean |ρ|"]
        for _p in _pairs
    ]


    # ============================================================
    # 5 — DISPLAY
    # ============================================================

    mo.vstack([
        mo.md(
            r"""
            ### Correlation matrix

            For two state variables $x_i$ and $x_j$, the
            correlation coefficient is

            $$
    \rho_{ij} =
            \frac{\Sigma_{ij}}
            {\sqrt{\Sigma_{ii}\Sigma_{jj}}}.
    $$

            The heatmap displays $|\rho_{ij}|$:

            - **Close to 1:** strong correlation (positive or negative).
            - **Close to 0:** weak linear correlation.


            """
        ),

        mo.hstack([
            mo.ui.plotly(_heat),

            mo.md(
                f"**Mean inter-landmark |ρ|:** "
                f"{np.mean(_inter_corr):.2f}\n\n"
                f"**Landmarks whose std ever increased after a "
                f"measurement:** {_incr} / {len(_ids)}\n\n"
                f"**Final landmark std / robot std at first sighting:**\n\n"
                f"- Min: {min(_ratios):.2f}\n"
                f"- Median: {np.median(_ratios):.2f}"
            ),
        ], widths=[3, 2]),

        mo.md(
            r"""
            ### Uncertainty over time

            **Top:** estimated landmark position uncertainty
            $\sqrt{\operatorname{tr}(\Sigma_{jj})}$.

            **Bottom:** robot position uncertainty at the instant
            each landmark was first observed.

            These are different quantities. The uncertainty of
            the robot at first sighting is not necessarily a
            lower bound for the landmark's final uncertainty.
            """
        ),

        mo.ui.plotly(_fig),

        mo.md(
            f"""
            ### Long-gap landmark re-observations

            The table shows landmarks re-observed after at least
            **{_REOBS_GAP_S:.0f} seconds** without a measurement
            of that same landmark.

            These are **potential loop-closure opportunities**.
            A loop closure occurs when the robot recognises a
            previously visited area and uses that information
            to constrain accumulated drift.

            A long time gap alone does not guarantee that
            the robot has completed a loop.
            """
        ),

        (
            mo.ui.table(
                _pd.DataFrame(_reobs),
                selection=None,
                page_size=8,
            )
            if _reobs
            else mo.md(
                "No long-gap re-observations found "
                "for this run and threshold."
            )
        ),

        mo.md(
            r"""
            ### Co-observation and correlation

            The table compares landmark pairs that were observed
            within the same **2-second time bin** with pairs that
            were never observed in the same bin.

            The correlation score is the mean absolute value of
            the four entries of their $2\times2$ correlation
            block.

            **Key question:** Can two landmarks be strongly
            correlated even if they have never been observed
            together?
            """
        ),

        (
            mo.ui.table(
                _pd.DataFrame(_pairs_show),
                selection=None,
                page_size=10,
            )
            if _pairs_show
            else mo.md("Not enough landmark pairs to compare.")
        ),
    ])
    return


@app.cell(hide_code=True)
def _(
    Q,
    align_map,
    alphas,
    evaluate,
    landmark_label,
    map_table,
    mo,
    np,
    pd,
    run,
    run_ekf_slam,
    slam_res,
):
    mo.stop(slam_res is None or not slam_res["ids"], mo.md("No mapped landmarks to compare yet; choose a longer run."))
    _rows = []
    for _name, _res in [("full covariance (correct)", slam_res), ("correlations dropped (wrong)", run_ekf_slam(run, alphas, Q, decouple=True))]:
        _df, _sm = map_table(run, _res, align_map, landmark_label)
        _m2 = _df["Mahalanobis² (χ²₂: 99 % < 9.21)"].to_numpy()
        _rows.append({
            "filter": _name, **{_k: round(_v, 3) for _k, _v in evaluate(_res["states"], run["gt"]).items()},
            "map RMSE (m)": round(_sm["map RMSE (m)"], 3),
            "median landmark std (m)": round(float(np.median([np.array(_res["lm_std"][_b])[-1, 1] for _b in _res["ids"]])), 3),
            "median Mahalanobis²": round(float(np.median(_m2)), 1),
            "landmarks with Mahalanobis² < 9.21": f"{int(np.sum(_m2 < 9.21))} / {len(_m2)}",
        })
    mo.ui.table(pd.DataFrame(_rows), selection=None)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    #### ✍️ ANSWERS — Part 4

    **4.1** Describe the correlation matrix. Which blocks are strongly correlated? Can two landmarks that were **not observed in the same 2-second time bin** still be correlated? Explain how.

    > *Your answer…*

    **4.2** Do your results agree with property 1? The circles mark the robot position std at the moment each landmark was first observed: is that value a lower bound for the final std of the landmark? Explain why or why not, and what the actual bound of property 3 is in this run. Why is property 3 a statement about the **absolute** map, not the relative one?

    > *Your answer…*

    **4.3** What happens when correlations are dropped? Compare landmark uncertainty and Mahalanobis². Does `decouple=True` actually reduce the computational complexity **of this implementation**? Why is discarding correlations not a valid way to simplify EKF-SLAM?

    > *Your answer…*
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ---
    ## 5 · Losing the landmarks and seeing them again

    The sliders define a time window in which all landmark measurements are **ignored** (as if the camera failed or the robot drove through a region without landmarks). Odometry is still used. Watch the robot error and its $3\sigma$ radius $3\sqrt{\lambda_{\max}(\Sigma_{xy})}$ during the window and, especially, when the robot **re-observes a landmark that was mapped before the window** (dashed line).
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    bo_start_sl = mo.ui.slider(0, 270, step=5, value=120, label="blackout start (s)", show_value=True)
    bo_len_sl = mo.ui.slider(0, 150, step=5, value=60, label="blackout length (s)", show_value=True)
    mo.hstack([bo_start_sl, bo_len_sl], justify="start")
    return bo_len_sl, bo_start_sl


@app.cell(hide_code=True)
def _(
    Q,
    align_map,
    alphas,
    bo_len_sl,
    bo_start_sl,
    evaluate,
    go,
    gt_at,
    landmark_label,
    make_subplots,
    map_table,
    mo,
    np,
    pd,
    run,
    run_ekf_slam,
    slam_res,
):
    # PROVIDED — map error evolution (ground truth used for evaluation only)

    def map_rmse_over_time(run, res, times, other_res=None):
        """
        Compute map RMSE over time using a dynamically growing landmark set.

        If other_res is provided, evaluate only landmarks already mapped
        in BOTH runs at each timestamp, ensuring a fair comparison.

        Ground truth is used for evaluation only.

        Returns:
            errors: map RMSE at each time
            counts: number of landmarks included at each time
        """
        _times = np.asarray(times, dtype=float)
        _errors = np.full(len(_times), np.nan)
        _counts = np.zeros(len(_times), dtype=int)

        _history = res["map_history"]
        _history_times = np.array([_h[0] for _h in _history])

        if other_res is not None:
            _other_history = other_res["map_history"]
            _other_times = np.array([_h[0] for _h in _other_history])

        for _k, _t in enumerate(_times):

            # Latest available map estimate
            _idx = np.searchsorted(
                _history_times, _t, side="right"
            ) - 1

            if _idx < 0:
                continue

            _, _ids, _map = _history[_idx]

            # Landmarks currently known
            _slots = {
                _b: _j for _j, _b in enumerate(_ids)
            }

            _barcodes = [
                _b for _b in _ids
                if _b in run["landmarks"]
            ]

            # For comparisons, keep only landmarks known in BOTH runs
            if other_res is not None:
                _other_idx = np.searchsorted(
                    _other_times, _t, side="right"
                ) - 1

                if _other_idx < 0:
                    continue

                _other_ids = set(
                    _other_history[_other_idx][1]
                )

                _barcodes = [
                    _b for _b in _barcodes
                    if _b in _other_ids
                ]

            if not _barcodes:
                continue

            _est = np.array([
                _map[_slots[_b]]
                for _b in _barcodes
            ])

            _ref = np.array([
                run["landmarks"][_b]
                for _b in _barcodes
            ])

            _errors[_k] = np.sqrt(
                np.mean(
                    np.sum((_est - _ref) ** 2, axis=1)
                )
            )

            _counts[_k] = len(_barcodes)

        return _errors, _counts


    mo.stop(slam_res is None)

    _win = (bo_start_sl.value, bo_start_sl.value + bo_len_sl.value)
    slam_bo = run_ekf_slam(run, alphas, Q, blackout=_win)
    mo.stop(slam_bo is None)

    # ------------------------------------------------------------
    # 1) Robot error + 3σ, with and without blackout
    # ------------------------------------------------------------
    _fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        subplot_titles=(
            "Position error and 3σ radius (m)",
            "Heading error and 3σ (deg)",
        ),
    )

    _rows = []

    for _name, _res, _c in [
        ("no blackout", slam_res, "#1f77b4"),
        ("with blackout", slam_bo, "#d62728"),
    ]:
        _s = _res["states"]
        _g = gt_at(run["gt"], _s[:, 0])
        _t = _s[:, 0] - run["t0"]

        # Robot position error
        _e = np.hypot(_s[:, 1] - _g[:, 0], _s[:, 2] - _g[:, 1])

        # 3σ position radius from the 2x2 robot covariance block
        _r3 = 3 * np.sqrt(np.linalg.eigvalsh(_res["robot_cov"][:, :2, :2])[:, -1])

        # Absolute heading error in degrees
        _eh = np.degrees(
            np.abs((_s[:, 3] - _g[:, 2] + np.pi) % (2 * np.pi) - np.pi)
        )

        # 3σ heading uncertainty in degrees
        _h3 = 3 * np.degrees(np.sqrt(np.maximum(_res["robot_cov"][:, 2, 2], 0.0)))

        # Plot every 10th point to keep the figure lighter
        _k = slice(None, None, 10)

        _fig.add_trace(
            go.Scatter(
                x=_t[_k], y=_e[_k],
                mode="lines",
                name=f"error — {_name}",
                line=dict(color=_c),
            ),
            row=1, col=1,
        )
        _fig.add_trace(
            go.Scatter(
                x=_t[_k], y=_r3[_k],
                mode="lines",
                name=f"3σ — {_name}",
                line=dict(color=_c, dash="dot"),
            ),
            row=1, col=1,
        )
        _fig.add_trace(
            go.Scatter(
                x=_t[_k], y=_eh[_k],
                mode="lines",
                showlegend=False,
                line=dict(color=_c),
            ),
            row=2, col=1,
        )
        _fig.add_trace(
            go.Scatter(
                x=_t[_k], y=_h3[_k],
                mode="lines",
                showlegend=False,
                line=dict(color=_c, dash="dot"),
            ),
            row=2, col=1,
        )

        # Summary table for this run
        _df, _sm = map_table(run, _res, align_map, landmark_label)
        _m2 = _df["Mahalanobis² (χ²₂: 99 % < 9.21)"].to_numpy()

        _rows.append({
            "run": _name,
            **{
                _kk: round(_v, 3)
                for _kk, _v in evaluate(_s, run["gt"]).items()
            },
            "landmarks": len(_res["ids"]),
            "map RMSE (m)": round(_sm["map RMSE (m)"], 3),
            "aligned map RMSE (m)": round(
                _sm["map RMSE after rigid alignment (m)"], 3
            ),
            "alignment rotation (deg)": round(
                _sm["alignment rotation (deg)"], 2
            ),
            "consistent landmarks": f"{int(np.sum(_m2 < 9.21))}/{len(_m2)}",
        })

    # ------------------------------------------------------------
    # 2) First re-observation after the blackout of an "old" landmark
    # ------------------------------------------------------------
    _old = [
        _b for _b, _f in slam_bo["first_seen"].items()
        if _f["t"] < _win[0]
    ]

    _m = run["meas"]
    _tm = _m[:, 0] - run["t0"]

    _sel = (_tm >= _win[1]) & np.isin(_m[:, 1], _old)
    _t_reobs = float(_tm[_sel][0]) if _sel.any() else None

    for _row in (1, 2):
        if _win[1] > _win[0]:
            _fig.add_vrect(
                x0=_win[0],
                x1=_win[1],
                fillcolor="gray",
                opacity=0.15,
                line_width=0,
                row=_row,
                col=1,
            )
        if _t_reobs is not None:
            _fig.add_vline(
                x=_t_reobs,
                line_dash="dash",
                line_color="black",
                row=_row,
                col=1,
            )

    _fig.update_xaxes(title_text="time since start (s)", row=2, col=1)
    _fig.update_yaxes(showgrid=True, gridcolor="#e5e7eb")
    _fig.update_xaxes(showgrid=True, gridcolor="#e5e7eb")
    _fig.update_layout(
        height=560,
        plot_bgcolor="white",
    )

    # ------------------------------------------------------------
    # 3) Dynamic map RMSE over time
    # ------------------------------------------------------------

    _t_grid = np.arange(
        0.0,
        run["odom"][-1, 0] - run["t0"] + 1e-9,
        1.0,
    )

    _fig_map = go.Figure()

    for _name, _res, _other, _color in [
        ("no blackout", slam_res, slam_bo, "#1f77b4"),
        ("with blackout", slam_bo, slam_res, "#d62728"),
    ]:

        _errors, _counts = map_rmse_over_time(
            run,
            _res,
            _t_grid,
            other_res=_other,
        )

        _fig_map.add_trace(
            go.Scatter(
                x=_t_grid,
                y=_errors,
                customdata=_counts,
                mode="lines",
                name=_name,
                line=dict(
                    color=_color,
                    width=2,
                    shape="hv",
                ),
                hovertemplate=(
                    "t = %{x:.1f} s<br>"
                    "Map RMSE = %{y:.3f} m<br>"
                    "Landmarks = %{customdata}"
                    "<extra>%{fullData.name}</extra>"
                ),
            )
        )

    # Blackout interval
    if _win[1] > _win[0]:
        _fig_map.add_vrect(
            x0=_win[0],
            x1=_win[1],
            fillcolor="gray",
            opacity=0.15,
            line_width=0,
        )

    # First re-observation after blackout
    if _t_reobs is not None:
        _fig_map.add_vline(
            x=_t_reobs,
            line_dash="dash",
            line_color="black",
        )

    _fig_map.update_layout(
        title="Map RMSE over time — dynamically mapped landmarks",
        xaxis_title="Time since start (s)",
        yaxis_title="Map RMSE (m)",
        height=400,
        plot_bgcolor="white",
        legend=dict(orientation="h", y=1.15),
    )

    _fig_map.update_xaxes(
        showgrid=True,
        gridcolor="#e5e7eb",
    )

    _fig_map.update_yaxes(
        showgrid=True,
        gridcolor="#e5e7eb",
    )
    # ------------------------------------------------------------
    # 4) Landmark observation timeline (also useful for Part 5.3)
    # ------------------------------------------------------------
    _last_s = max(5.0, run["odom"][-1, 0] - run["t0"])
    _bins = np.arange(0.0, _last_s + 5.01, 5.0)
    _obs_counts, _edges = np.histogram(run["meas"][:, 0] - run["t0"], bins=_bins)
    _fig_obs = go.Figure(go.Bar(
        x=_edges[:-1] + 2.5, y=_obs_counts,
        width=4.5, marker_color="#64748b", name="measurements",
    ))
    _fig_obs.update_layout(
        title="Landmark observations over time (5 s bins)",
        xaxis_title="time since start (s)",
        yaxis_title="measurements per 5 s",
        height=260, plot_bgcolor="white", showlegend=False,
        margin=dict(t=55, b=40),
    )
    _fig_obs.update_xaxes(showgrid=True, gridcolor="#e5e7eb")
    _fig_obs.update_yaxes(showgrid=True, gridcolor="#e5e7eb")

    # ------------------------------------------------------------
    # 5) Display
    # ------------------------------------------------------------
    mo.vstack([
        mo.ui.table(pd.DataFrame(_rows), selection=None),
        mo.md(
            f"""
            **Blackout:** {_win[0]:.0f}–{_win[1]:.0f} s  
            **Measurements ignored:** {slam_bo['info']['ignored (blackout)']}  
            """
            + (
                f"**First re-observation of an old landmark:** t = {_t_reobs:.1f} s"
                if _t_reobs is not None
                else "**First re-observation of an old landmark:** none after the blackout window"
            )
        ),
        mo.ui.plotly(_fig),
        mo.md(
            f"""
            ### Map error over time

            At each instant the RMSE is computed over the landmarks already
            mapped **in both runs**, so the two curves compare the same
            landmarks (the hover shows how many). The curves jump when a new
            landmark enters the comparison.

            During the blackout the landmark means are not updated, because
            all measurements are ignored. When observations resume, the map
            can be corrected through the robot–map and landmark–landmark
            correlations stored in the covariance.
            """
        ),
        mo.ui.plotly(_fig_map),
        mo.md("### Landmark observation timeline\n\nCounts of real landmark measurements in 5-second bins. Use this to identify long periods without observations, particularly in Part 5.3."),
        mo.ui.plotly(_fig_obs),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    #### ✍️ ANSWERS — Part 5

    **5.1** During the blackout, how do position/heading errors and their displayed 3σ indicators evolve? Do the errors seem compatible with the reported uncertainties? Why is a single run and a 3σ curve only an indicative consistency check (especially in 2D)?

    > *Your answer…*


    **5.2 — Re-observation and map correction**
    1. What happens to the robot errors and to their 3σ bounds at the first re-observation of a landmark mapped before the blackout (dashed line)?
    2. Use the map-error plot: does the map change during the blackout? What happens when observations resume? Does a reduction of the uncertainty always come with a reduction of the actual error?
    3. Explain with $K=\bar\Sigma H^\top S^{-1}$ how the observation of one landmark can correct other landmarks.

    > *Your answer…*


    **5.3** Without blackout, select **Dataset6 / Robot2** (300 s). Inspect the **landmark observation timeline**, the heading error and 3σ, and the map table (raw vs aligned RMSE, rotation). Describe what happens in the first minute and compare the EKF heading error with the dead-reckoning heading error at the same times. This case will be further analysed in E2.

    > *Your answer…*
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ---
    ## 6 · Computational cost

    So far, we have evaluated how accurately EKF-SLAM estimates the robot pose and the landmark map. But **can EKF-SLAM scale to a large environment with thousands of landmarks?**

    In EKF-SLAM, the state contains the robot pose $(x,y,\theta)$ and the 2D positions of $n$ landmarks. Therefore, the state dimension is $D=3+2n$, and the covariance matrix has size $D\times D$.

    As the map grows, some computations become more expensive.

    **Computational experiment.** The button below measures the execution times of **your implementations** of `slam_predict` and `slam_update`, using synthetic states with different numbers of landmarks. This allows us to study how computation time increases with map size without running an entire robot dataset.

    We approximate the relationship between execution time and the number of landmarks as

    $$
    t(n)\propto n^p.
    $$

    The exponent $p$ is estimated from the slope of the **log–log plot**. To interpret it, imagine doubling the number of landmarks:

    - $p\approx1$: execution time approximately doubles.
    - $p\approx2$: execution time increases by a factor of four.
    - $p\approx3$: execution time increases by a factor of eight.

    **Why synthetic data?** The MRCLAM datasets contain only a small number of landmarks, so they cannot show how computational cost grows for large maps. Here we generate artificial SLAM states with different numbers of landmarks and covariance matrices of the corresponding sizes. We then time your actual `slam_predict` and `slam_update`. This lets us vary the map size while keeping the experiment controlled. We measure the cost of the EKF operations, not the runtime of a complete robotic system.

    **Your task:** Compare the measured exponents with the theoretical computational cost of EKF-SLAM prediction and update. Then use your measurements to assess whether the algorithm would be practical for a large visual SLAM system.

    Remember that **theoretical complexity and measured runtime are not necessarily identical**: implementation details such as copying matrices or using dense matrix operations also matter.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    cost_btn = mo.ui.run_button(label="Run timing (≈ 10 s)")
    cost_btn
    return (cost_btn,)


@app.cell(hide_code=True)
def _(cost_btn, go, mo, np, slam_res, time):
    mo.stop(slam_res is None, mo.md("⏳ Implement TODO 1–4 first."))
    mo.stop(not cost_btn.value, mo.md("Press the button to run the timing."))
    _ns = [10, 25, 50, 100, 200, 400]
    _rng = np.random.default_rng(0)
    _Q = np.diag([0.1, 0.05]) ** 2
    _tp, _tu = [], []
    for _n in _ns:
        _N = 3 + 2 * _n
        _A = _rng.normal(size=(_N, _N))
        _Sig = _A @ _A.T / _N + np.eye(_N)
        _mu = np.concatenate([[0.0, 0.0, 0.0], _rng.uniform(-5, 5, 2 * _n)])
        _rep = max(3, int(2000 / _n))
        _t0 = time.perf_counter()
        for _ in range(_rep):
            slam_predict(_mu, _Sig, 0.2, 0.1, 0.015, (0.2, 0.02, 0.02, 0.5))
        _t1 = time.perf_counter()
        for _ in range(_rep):
            slam_update(_mu, _Sig, (2.0, 0.1), _n // 2, _Q)
        _t2 = time.perf_counter()
        _tp.append((_t1 - _t0) / _rep * 1e3)
        _tu.append((_t2 - _t1) / _rep * 1e3)
    _pp = np.polyfit(np.log(_ns[-3:]), np.log(_tp[-3:]), 1)[0]
    _pu = np.polyfit(np.log(_ns[-3:]), np.log(_tu[-3:]), 1)[0]
    _fig = go.Figure()
    _fig.add_trace(go.Scatter(x=_ns, y=_tp, mode="lines+markers", name=f"slam_predict (slope {_pp:.1f})"))
    _fig.add_trace(go.Scatter(x=_ns, y=_tu, mode="lines+markers", name=f"slam_update (slope {_pu:.1f})"))
    _fig.update_xaxes(type="log", title_text="number of landmarks n")
    _fig.update_yaxes(type="log", title_text="time per call (ms)")
    _fig.update_layout(height=420, plot_bgcolor="white", title="Cost per call (slopes fitted on the three largest n)")
    mo.vstack([mo.ui.plotly(_fig), mo.md(f"Empirical exponent: predict **{_pp:.2f}**, update **{_pu:.2f}**.")])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    #### ✍️ ANSWERS — Part 6

    ### 6.1 — Computational complexity of EKF-SLAM

    1. Report the measured exponents $p$ for `slam_predict` and `slam_update`. What do these values mean when the number of landmarks doubles?
    2. What are the **theoretical complexities** of EKF-SLAM prediction and update as a function of $n$? Explain why they differ.
    3. Does your measured prediction exponent agree with the theoretical complexity? If not, identify an implementation detail that could explain the difference.

    > *Your answer…*


    ### 6.2 — Can EKF-SLAM scale to visual SLAM?

    Consider a visual SLAM system with approximately $10^4$ mapped landmarks, processing tens of landmark observations per frame at 30 Hz.

    1. Use your measured update time and its scaling exponent to estimate the computational cost at $n=10^4$.
    2. Estimate the memory required for the full covariance matrix, assuming 64-bit floating-point numbers. Is this approach practical in real time?
    3. Are there alternative SLAM systems to avoid the computational cost of dense EKF-SLAM?

    > *Your answer…*
    """
    )
    return


if __name__ == "__main__":
    app.run()
