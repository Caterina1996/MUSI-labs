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
    # L2 — Gaussian Prediction and EKF Localization

    **11765 · Navigation and Environment Modelling in Mobile Robotics · MUSI** · Class exercise, Friday 09/10/2026

    **Formative** exercise (not graded). The functions you write here are reused in **E1**: keep your completed notebook.

    ### Learning objectives
    1. Propagate Gaussian uncertainty through the motion model ($G$, $V$, $M$) and judge whether it is **consistent** with the real error.
    2. Implement the range–bearing landmark model and its Jacobian.
    3. Implement the EKF correction and interpret the innovation and the Kalman gain.
    4. Diagnose why a correct EKF can perform badly on real data, and protect it with **Mahalanobis validation gating**.

    ### Recommended reading
    Thrun, Burgard & Fox, *Probabilistic Robotics*: Sec. 3.3 (EKF) and Sec. 7.4 (EKF localization, Table 7.2) · Siegwart & Nourbakhsh, *Introduction to Autonomous Mobile Robots*: Kalman-filter localization in Ch. 5, including measurement validation with the Mahalanobis distance.

    ### How to work
    - **🛠 TODO** cells: write code. **✅ CHECK** cells: automatic feedback on your code. **✍️ ANSWER** cells: double-click and write a short answer **with numbers from your own runs**.
    - All parts (1–5) and all ANSWER cells are part of the exercise. In class, the priority is to get every ✅ CHECK green; finish the answers at home.
    - Ground truth is used only for the initial pose and for analysis/evaluation, **never inside the filter**.
    """
    )
    return


@app.cell(hide_code=True)
def _():
    import os

    import numpy as np
    import pandas as pd
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    return go, make_subplots, np, os, pd


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
          landmarks     {barcode: (x, y)}           known map
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

    def landmark_label(run, barcode):
        """'L<subject>' if the subject number is known, otherwise the barcode."""
        _ids = run.get("landmark_ids", {})
        return f"L{_ids[barcode]}" if barcode in _ids else f"{int(barcode)}"

    def plot_trajectories(run, estimates, title="Trajectories", ellipses=None, label_landmarks=True):
        """estimates: dict name -> array [t, x, y, theta]; ellipses: list of (mean_xy, cov_2x2)."""
        _fig = go.Figure()
        _fig.add_trace(go.Scatter(x=run["gt"][:, 1], y=run["gt"][:, 2], mode="lines", name="ground truth", line=dict(color="#1f77b4", dash="dot")))
        _bcs = list(run["landmarks"].keys())
        _lx = [run["landmarks"][b][0] for b in _bcs]
        _ly = [run["landmarks"][b][1] for b in _bcs]
        _lab = [landmark_label(run, b) for b in _bcs]
        _hover = [f"<b>{l}</b><br>barcode {int(b)}<br>x = {x:.2f} m, y = {y:.2f} m" for l, b, x, y in zip(_lab, _bcs, _lx, _ly)]
        _fig.add_trace(go.Scatter(
            x=_lx, y=_ly, mode="markers+text" if label_landmarks else "markers", name="landmarks",
            text=_lab, textposition="top center", textfont=dict(size=11, color="#555555"),
            hovertext=_hover, hoverinfo="text",
            marker=dict(symbol="star", size=11, color="gold", line=dict(width=0.5, color="#8a6d00")),
        ))
        for _i, (_name, _s) in enumerate(estimates.items()):
            if _s is not None:
                _fig.add_trace(go.Scatter(x=_s[:, 1], y=_s[:, 2], mode="lines", name=_name, line=dict(color=_COLORS[_i % 5])))
        if ellipses:
            _phi = np.linspace(0, 2 * np.pi, 60)
            _c = np.stack([np.cos(_phi), np.sin(_phi)])
            for _m, _Sigma_xy in ellipses:
                _lam, _vec = np.linalg.eigh(_Sigma_xy)
                _p = _vec @ (3 * np.sqrt(np.maximum(_lam, 0))[:, None] * _c)
                _fig.add_trace(go.Scatter(x=_m[0] + _p[0], y=_m[1] + _p[1], mode="lines", line=dict(color="gray", width=1), showlegend=False, hoverinfo="skip"))
        _fig.update_layout(title=title, height=520, plot_bgcolor="white", xaxis_title="x (m)", yaxis_title="y (m)", yaxis=dict(scaleanchor="x", scaleratio=1))
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
    return landmark_label, plot_errors, plot_trajectories


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Provided helpers from L1

    `wrap_angle` and `motion_euler` are the L1 solutions. `num_jacobian` computes a **numerical** Jacobian with central finite differences,

    $$
    \frac{\partial f}{\partial x_i}\approx\frac{f(x+\varepsilon e_i)-f(x-\varepsilon e_i)}{2\varepsilon},
    $$

    and is used **only by the CHECK cells** to verify your analytical Jacobians. It is not part of the EKF.
    """
    )
    return


@app.cell(hide_code=True)
def _(np):
    # PROVIDED — solutions of L1 and a numerical Jacobian for the CHECK cells
    def wrap_angle(a):
        return (a + np.pi) % (2 * np.pi) - np.pi

    def motion_euler(x, v, w, dt):
        return np.array([x[0] + v * dt * np.cos(x[2]),
                         x[1] + v * dt * np.sin(x[2]),
                         wrap_angle(x[2] + w * dt)])

    def num_jacobian(f, x, eps=1e-6):
        """Central finite differences. Used only by the CHECK cells, not by the EKF."""
        _x = np.asarray(x, float)
        _f0 = np.asarray(f(_x))
        _J = np.zeros((_f0.size, _x.size))
        for _i in range(_x.size):
            _d = np.zeros_like(_x)
            _d[_i] = eps
            _J[:, _i] = (np.asarray(f(_x + _d)) - np.asarray(f(_x - _d))) / (2 * eps)
        return _J
    return motion_euler, num_jacobian, wrap_angle


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
def _(mo, plot_trajectories, run):
    mo.ui.plotly(plot_trajectories(run, {}, title="Ground-truth trajectory and landmarks (hover a landmark for its barcode)"))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ---
    ## 1 · Gaussian prediction (≈ 30 min)

    The EKF represents the belief as a Gaussian $\mathcal N(\mu,\Sigma)$. The prediction step propagates both the **estimate** and its **uncertainty** (Thrun et al., Table 7.2), here with the **Euler** velocity model of L1 as $g$:

    $$
    \bar\mu_t=g(\mu_{t-1},u_t),\qquad
    \bar\Sigma_t=G_t\Sigma_{t-1}G_t^\top+V_tM_tV_t^\top,
    $$

    $$
    G_t=\frac{\partial g}{\partial x}\Big|_{\mu_{t-1}},\qquad
    V_t=\frac{\partial g}{\partial u}\Big|_{\mu_{t-1}},\qquad
    M_t=\begin{pmatrix}\alpha_1v^2+\alpha_2\omega^2&0\\0&\alpha_3v^2+\alpha_4\omega^2\end{pmatrix}.
    $$

    | Symbol | Size | Meaning |
    |---|---:|---|
    | $\mu_{t-1}$, $\Sigma_{t-1}$ | $3$, $3\times3$ | previous pose estimate $(x,y,\theta)$ and its covariance |
    | $u_t=(v,\omega)$ | $2$ | applied control: linear and angular velocity |
    | $G_t$ | $3\times3$ | Jacobian w.r.t. the **state**: how a previous pose error propagates (state error → state error) |
    | $V_t$ | $3\times2$ | Jacobian w.r.t. the **control**: how a control error becomes a pose error (control error → state error) |
    | $M_t$ | $2\times2$ | covariance of the control noise in $(v,\omega)$ space |
    | $V_tM_tV_t^\top$ | $3\times3$ | control noise expressed in state space: the **process noise** $R_t$ of the EKF algorithm |
    | $\bar\mu_t$, $\bar\Sigma_t$ | $3$, $3\times3$ | predicted pose and its covariance |

    The predicted covariance has two sources:

    $$
    \bar\Sigma_t=\underbrace{G_t\Sigma_{t-1}G_t^\top}_{\text{uncertainty already present}}+\underbrace{V_tM_tV_t^\top}_{\text{new uncertainty from noisy controls}}.
    $$

    For the Euler model used here,

    $$
    g(x,u)=\begin{bmatrix}x+v\Delta t\cos\theta\\ y+v\Delta t\sin\theta\\ \theta+\omega\Delta t\end{bmatrix},
    $$

    so $G_t$ is obtained by differentiating $g$ w.r.t. $(x,y,\theta)$ and $V_t$ by differentiating it w.r.t. $(v,\omega)$.

    ⚠️ The Jacobian $G_t$ of the **exact (circular-arc)** model in Thrun et al. is different from the one you need here. The CHECK compares your $G$ and $V$ with finite differences of the **Euler** model.

    ### 🛠 TODO 1
    Implement `G_jacobian`, `V_jacobian`, `control_noise` and `ekf_predict` (returns `(mu_bar, Sigma_bar)`). Derive $G$ and $V$ by hand first.
    """
    )
    return


@app.cell
def _():
    def G_jacobian(mu, v, w, dt):
        return None

    def V_jacobian(mu, v, w, dt):
        return None
    # Mt
    def control_noise(v, w, alphas):
        return None

    def ekf_predict(mu, Sigma, v, w, dt, alphas):
        return None, None
    return G_jacobian, V_jacobian, control_noise, ekf_predict


@app.cell(hide_code=True)
def _(
    G_jacobian,
    V_jacobian,
    control_noise,
    ekf_predict,
    mo,
    motion_euler,
    np,
    num_jacobian,
):
    def _check():
        _mu = np.array([1.0, 2.0, 0.7])
        _v, _w, _dt = 0.3, 0.2, 0.05
        try:
            _r = []
            _r.append(("G matches finite differences", np.allclose(G_jacobian(_mu, _v, _w, _dt), num_jacobian(lambda x: motion_euler(x, _v, _w, _dt), _mu), atol=1e-6)))
            _r.append(("V matches finite differences", np.allclose(V_jacobian(_mu, _v, _w, _dt), num_jacobian(lambda u: motion_euler(_mu, u[0], u[1], _dt), [_v, _w]), atol=1e-6)))
            _r.append(("M = diag(α1v²+α2ω², α3v²+α4ω²)", np.allclose(control_noise(_v, _w, (1, 2, 3, 4)), np.diag([0.09 + 0.08, 0.27 + 0.16]))))
            _mu_bar, _Sigma_bar = ekf_predict(_mu, np.eye(3) * 0.01, _v, _w, _dt, (0.1, 0.01, 0.01, 0.1))
            _r.append(("predicted mean = motion model", np.allclose(_mu_bar, motion_euler(_mu, _v, _w, _dt))))
            _r.append(("predicted Σ̄ symmetric, positive definite, larger trace", np.allclose(_Sigma_bar, _Sigma_bar.T) and np.all(np.linalg.eigvalsh(_Sigma_bar) > 0) and np.trace(_Sigma_bar) > 0.03))
        except Exception as _e:
            return f"⏳ not implemented yet ({type(_e).__name__})"
        return "<br>".join(f"{'✅' if _ok else '❌'} {_n}" for _n, _ok in _r)

    mo.md("**CHECK — prediction**<br>" + _check())
    return


@app.cell(hide_code=True)
def _(mo):
    a1_sl = mo.ui.slider(0.0, 2.0, step=0.05, value=0.2, label="α1", show_value=True)
    a2_sl = mo.ui.slider(0.0, 0.5, step=0.01, value=0.02, label="α2", show_value=True)
    a3_sl = mo.ui.slider(0.0, 0.5, step=0.01, value=0.02, label="α3", show_value=True)
    a4_sl = mo.ui.slider(0.0, 8.0, step=0.1, value=0.5, label="α4", show_value=True)
    mo.vstack([mo.md("**Control-noise parameters** (used in the whole notebook)"), mo.hstack([a1_sl, a2_sl, a3_sl, a4_sl], justify="start")])
    return a1_sl, a2_sl, a3_sl, a4_sl


@app.cell(hide_code=True)
def _(a1_sl, a2_sl, a3_sl, a4_sl):
    alphas = (a1_sl.value, a2_sl.value, a3_sl.value, a4_sl.value)
    return (alphas,)


@app.cell(hide_code=True)
def _(alphas, ekf_predict, gt_at, np, run):
    # PROVIDED — prediction only (no measurements): mean = dead reckoning, covariance grows.
    def predict_only(run, alphas):
        _mu = run["x0"].copy()
        _Sigma = np.diag([1e-6] * 3)
        _t = run["t0"]
        _u = (0.0, 0.0)
        _st, _cov = [[_t, *_mu]], [_Sigma]
        for _e in run["events"]:
            if _e[1] != 0:
                continue
            _dt = _e[0] - _t
            if _dt > 0:
                _mu, _Sigma = ekf_predict(_mu, _Sigma, _u[0], _u[1], _dt, alphas)
                if _mu is None:
                    return None, None
                _t = _e[0]
                _st.append([_t, *_mu])
                _cov.append(_Sigma)
            _u = (_e[2], _e[3])
        return np.array(_st), np.array(_cov)

    pred_states, pred_covs = predict_only(run, alphas)
    nees = None
    if pred_states is not None:
        _g = gt_at(run["gt"], pred_states[:, 0])
        _e = pred_states[:, 1:3] - _g[:, :2]
        _idx = np.arange(200, len(pred_states), 25)
        nees = np.array([[pred_states[_i, 0] - run["t0"], _e[_i] @ np.linalg.solve(pred_covs[_i][:2, :2], _e[_i])] for _i in _idx])
    return nees, pred_covs, pred_states


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Is the predicted uncertainty consistent with the real error? (NEES)

    The prediction gives not only a position but also a covariance. The $2\times2$ position covariance $\Sigma_{xy}$ is drawn as a **3σ ellipse**: the set of errors with $e^\top\Sigma_{xy}^{-1}e\le 9$, where $e=(x-x^{gt},\,y-y^{gt})$ is the real position error.

    The **Normalized Estimation Error Squared**

    $$
    \mathrm{NEES}=e^\top\Sigma_{xy}^{-1}e
    $$

    is the squared Mahalanobis distance of the real error. It answers: *is the actual error of the size predicted by the covariance?*

    - If the filter is consistent, $\mathrm{NEES}\sim\chi^2_2$: its mean is **2**, and it stays below **11.83** (the 99.7 % $\chi^2_2$ bound) about 99.7 % of the time.
    - A persistently large NEES means the filter is **overconfident**: its ellipses are too small for the errors that actually occur.
    - In 2-D the 3σ ellipse corresponds to $\mathrm{NEES}=9$ and contains only about 98.9 % of the probability, not 99.7 % as in 1-D.

    In the figure, for each selected time: **○** predicted position, **×** ground-truth position, **ellipse** predicted 3σ uncertainty, **dotted line** real error.
    """
    )
    return


@app.cell(hide_code=True)
def _(go, gt_at, mo, nees, np, plot_trajectories, pred_covs, pred_states, run):
    if pred_states is None:
        _out = mo.md("⏳ Implement TODO 1 first.")
    else:
        _idx = np.linspace(len(pred_states) // 10, len(pred_states) - 1, 6).astype(int)
        _gt_sel = gt_at(run["gt"], pred_states[_idx, 0])
        _f1 = plot_trajectories(run, {"prediction (= DR)": pred_states}, title="Prediction only: estimate, ground truth and 3σ uncertainty")
        _cols = ["#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A", "#19D3F3"]
        _phi = np.linspace(0, 2 * np.pi, 80)
        _circ = np.stack([np.cos(_phi), np.sin(_phi)])
        for _k, _i in enumerate(_idx):
            _mean = pred_states[_i, 1:3]
            _gtp = _gt_sel[_k, :2]
            _lam, _vec = np.linalg.eigh(pred_covs[_i][:2, :2])
            _el = _vec @ (3 * np.sqrt(np.maximum(_lam, 0))[:, None] * _circ)
            _c = _cols[_k]
            _f1.add_trace(go.Scatter(x=_mean[0] + _el[0], y=_mean[1] + _el[1], mode="lines", line=dict(color=_c, width=2), name=f"t = {pred_states[_i, 0] - run['t0']:.0f} s"))
            _f1.add_trace(go.Scatter(x=[_mean[0]], y=[_mean[1]], mode="markers", marker=dict(color=_c, size=10, symbol="circle"), showlegend=False))
            _f1.add_trace(go.Scatter(x=[_gtp[0]], y=[_gtp[1]], mode="markers", marker=dict(color=_c, size=11, symbol="x"), showlegend=False))
            _f1.add_trace(go.Scatter(x=[_mean[0], _gtp[0]], y=[_mean[1], _gtp[1]], mode="lines", line=dict(color=_c, width=1, dash="dot"), showlegend=False))
        _f1.update_layout(height=650, legend_title="Selected times")
        _f2 = go.Figure(go.Scatter(x=nees[:, 0], y=nees[:, 1], mode="lines", name="NEES"))
        _f2.add_hline(y=2, line_dash="dot", annotation_text="expected mean = 2")
        _f2.add_hline(y=11.83, line_dash="dash", annotation_text="χ²₂ 99.7 % = 11.83")
        _f2.update_layout(title="NEES on (x, y)", xaxis_title="time (s)", yaxis_title="NEES", yaxis_type="log", height=420, plot_bgcolor="white")
        _out = mo.vstack([
            mo.ui.plotly(_f1),
            mo.ui.plotly(_f2),
            mo.md(f"Mean NEES = **{nees[:, 1].mean():.1f}** (ideal ≈ 2) · below the 99.7 % χ² bound **{100 * np.mean(nees[:, 1] < 11.83):.0f} %** of the time (ideal ≈ 99.7 %)"),
        ])
    _out
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    #### ✍️ ANSWER 1

    **1.1** Explain in words what $G$, $V$ and $M$ represent. Why are $G$ and $V$ evaluated at $\mu_{t-1}$?

    > *Your answer…*

    **1.2** How do the ellipses evolve along the trajectory? Which direction grows fastest and why?

    > *Your answer…*

    **1.3** Tune $\alpha$ so that Robot1 (Dataset1) is roughly consistent (mean NEES ≈ 2), then switch to Robot5 **without changing them**. What happens? Can a zero-mean noise model be consistent for both? Relate your answer to L1 (systematic errors).

    > *Your answer…*
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ---
    ## 2 · Landmark measurement model (≈ 20 min)

    For a known landmark $m=(m_x,m_y)$ the range–bearing sensor model is

    $$
    z=h(x,m)+\delta,\qquad \delta\sim\mathcal N(0,Q),\qquad
    h(x,m)=\begin{bmatrix}r\\ \phi\end{bmatrix}=\begin{bmatrix}\sqrt{\delta_x^2+\delta_y^2}\\ \operatorname{atan2}(\delta_y,\delta_x)-\theta\end{bmatrix},
    \qquad \delta_x=m_x-x,\;\delta_y=m_y-y,
    $$

    with the bearing $\phi$ wrapped to $[-\pi,\pi)$. Since the true pose is unknown, the EKF evaluates $h$ at the predicted mean: $\hat z=h(\bar\mu,m)$ is the **predicted measurement**, what the sensor should observe if the predicted pose were correct.

    Because $h$ is nonlinear ($\sqrt{\cdot}$, $\operatorname{atan2}$), the EKF linearises it around $\bar\mu$: $h(x)\approx h(\bar\mu)+H(x-\bar\mu)$, with the **measurement Jacobian**

    $$
    H=\frac{\partial h}{\partial x}\Big|_{\bar\mu}=
    \begin{bmatrix}
    -\delta_x/r & -\delta_y/r & 0\\
    \delta_y/q & -\delta_x/q & -1
    \end{bmatrix},
    \qquad q=\delta_x^2+\delta_y^2,\quad r=\sqrt q .
    $$

    $H$ is $2\times3$: it maps a small state error $(x,y,\theta)$ into a small measurement error $(r,\phi)$, $\delta z\approx H\,\delta x$. The first row describes the effect on the **range**, the second on the **bearing**. In Part 3 it is used to bring the pose uncertainty into measurement space, $H\bar\Sigma H^\top$.

    ### 🛠 TODO 2
    - `h_landmark(mu, landmark)`: `mu = [x, y, theta]`, `landmark = [m_x, m_y]` → returns `np.array([r, phi])` with `phi` wrapped.
    - `H_landmark(mu, landmark)` → returns the $2\times3$ Jacobian above.
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
            _mu = np.array([1.0, 1.0, 0.3])
            _l = (4.0, 5.0)
            _r = [
                ("h at a known configuration", np.allclose(h_landmark(np.array([0.0, 0.0, 0.0]), (3.0, 4.0)), [5.0, np.arctan2(4, 3)])),
                ("H matches finite differences", np.allclose(H_landmark(_mu, _l), num_jacobian(lambda x: h_landmark(x, _l), _mu), atol=1e-6)),
                ("bearing wrapped (landmark behind the robot)", abs(h_landmark(np.array([0.0, 0.0, np.pi - 0.1]), (-1.0, -0.2))[1]) <= np.pi),
            ]
        except Exception as _e:
            return f"⏳ not implemented yet ({type(_e).__name__})"
        return "<br>".join(f"{'✅' if _ok else '❌'} {_n}" for _n, _ok in _r)

    mo.md("**CHECK — measurement model**<br>" + _check())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### How good are the measurements?

    For **analysis only**, we use the ground-truth pose to compute what each landmark measurement should have been, $z_{\text{expected}}=h(x^{gt},m)$, and the **residual**

    $$
    e_z=z_{\text{measured}}-z_{\text{expected}} .
    $$

    A residual of 0 means perfect agreement; $e_r=-2$ m means the measured range was 2 m shorter than expected. If the EKF assumption $\delta\sim\mathcal N(0,Q)$ were a good description of the sensor, the residuals would form a single bell centred at 0. The histograms use a **log scale** so that rare large residuals remain visible. The second figure shows **when** each residual occurred and **which landmark** produced it; the table summarises the residuals per landmark.
    """
    )
    return


@app.cell(hide_code=True)
def _(
    go,
    gt_at,
    h_landmark,
    landmark_label,
    make_subplots,
    mo,
    np,
    run,
    wrap_angle,
):
    # PROVIDED — empirical measurement residuals (ground truth used ONLY for analysis)
    _m = run["meas"]
    _out = mo.md("⏳ Implement TODO 2 first.")
    if len(_m) and h_landmark(run["x0"], (0.0, 0.0)) is not None:
        _X = gt_at(run["gt"], _m[:, 0])
        _res = np.array([np.array([_z[2], _z[3]]) - h_landmark(_x, run["landmarks"][_z[1]]) for _z, _x in zip(_m, _X)])
        _res[:, 1] = wrap_angle(_res[:, 1])
        _time = _m[:, 0] - run["t0"]

        _hist = make_subplots(rows=1, cols=2, subplot_titles=("Range residual (m)", "Bearing residual (rad)"))
        _hist.add_trace(go.Histogram(x=_res[:, 0], nbinsx=120, name="range"), row=1, col=1)
        _hist.add_trace(go.Histogram(x=_res[:, 1], nbinsx=120, name="bearing"), row=1, col=2)
        _hist.add_vline(x=0, line_dash="dash", row=1, col=1)
        _hist.add_vline(x=0, line_dash="dash", row=1, col=2)
        _hist.update_xaxes(title_text="e_r = r_measured − r_expected (m)", row=1, col=1)
        _hist.update_xaxes(title_text="e_φ = φ_measured − φ_expected (rad)", row=1, col=2)
        _hist.update_yaxes(title_text="count (log scale)", type="log")
        _hist.update_layout(title="Residual distribution", height=380, showlegend=False, plot_bgcolor="white")

        _cols = ["#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A", "#19D3F3", "#FF6692", "#B6E880",
                 "#FF97FF", "#FECB52", "#2E91E5", "#E15F99", "#1CA71C", "#FB0D0D", "#DA16FF"]
        _tf = make_subplots(rows=2, cols=1, shared_xaxes=True, subplot_titles=("Range residual vs time", "Bearing residual vs time"))
        for _k, _b in enumerate(sorted(set(_m[:, 1]))):
            _mask = _m[:, 1] == _b
            _lab = landmark_label(run, _b)
            for _row, _unit in [(1, "m"), (2, "rad")]:
                _tf.add_trace(go.Scatter(
                    x=_time[_mask], y=_res[_mask, _row - 1], mode="markers", name=_lab, legendgroup=_lab, showlegend=_row == 1,
                    marker=dict(size=5, color=_cols[_k % len(_cols)]),
                    hovertemplate=f"{_lab} (barcode {int(_b)})<br>t = %{{x:.1f}} s<br>residual = %{{y:.3f}} {_unit}<extra></extra>",
                ), row=_row, col=1)
        _tf.add_hline(y=0, line_dash="dash", row=1, col=1)
        _tf.add_hline(y=0, line_dash="dash", row=2, col=1)
        _tf.update_yaxes(title_text="range res. (m)", row=1, col=1)
        _tf.update_yaxes(title_text="bearing res. (rad)", row=2, col=1)
        _tf.update_xaxes(title_text="time since start (s)", row=2, col=1)
        _tf.update_layout(title="When, and from which landmark, do the residuals come?", height=600, plot_bgcolor="white", legend_title="Landmark")

        _out = mo.vstack([
            mo.ui.plotly(_hist),
            mo.md(
                f"Range residual: median |·| = **{np.median(abs(_res[:, 0])):.3f} m**, std = **{_res[:, 0].std():.2f} m**, "
                f"**{100 * np.mean(abs(_res[:, 0]) > 1):.1f} %** larger than 1 m · bearing residual: median |·| = **{np.median(abs(_res[:, 1])):.3f} rad**"
            ),
            mo.ui.plotly(_tf),
        ])
    _out
    return


@app.cell(hide_code=True)
def _(gt_at, h_landmark, landmark_label, mo, np, pd, run, wrap_angle):
    # PROVIDED — residuals grouped by landmark (ground truth used ONLY for analysis)
    _m = run["meas"]
    _out = mo.md("⏳ Implement TODO 2 first.")
    if len(_m) and h_landmark(run["x0"], (0.0, 0.0)) is not None:
        _X = gt_at(run["gt"], _m[:, 0])
        _res = np.array([np.array([_z[2], _z[3]]) - h_landmark(_x, run["landmarks"][_z[1]]) for _z, _x in zip(_m, _X)])
        _res[:, 1] = wrap_angle(_res[:, 1])
        _bad = (np.abs(_res[:, 0]) > 1.0) | (np.abs(_res[:, 1]) > 0.5)
        _rows = []
        for _b in sorted(set(_m[:, 1])):
            _k = _m[:, 1] == _b
            _rows.append({
                "landmark": landmark_label(run, _b),
                "barcode": int(_b),
                "map position (x, y)": f"({run['landmarks'][_b][0]:.2f}, {run['landmarks'][_b][1]:.2f})",
                "n meas": int(_k.sum()),
                "% large": round(100 * _bad[_k].mean(), 1),
                "median range res. (m)": round(float(np.median(_res[_k, 0])), 2),
                "median bearing res. (rad)": round(float(np.median(_res[_k, 1])), 2),
            })
        _out = mo.vstack([
            mo.md("**Residuals per landmark** — large = |range res.| > 1 m or |bearing res.| > 0.5 rad"),
            pd.DataFrame(_rows).sort_values("% large", ascending=False),
        ])
    _out
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    #### ✍️ ANSWER 2

    **2.1** Why must the bearing *innovation* $z_\phi-\hat z_\phi$ be wrapped? Give a numerical example near $\pm\pi$ of what goes wrong otherwise.

    > *Your answer…*

    **2.2** Look at the residual histograms (log scale). Is a single Gaussian a good model of the measurement error? What would a Gaussian-based filter do with the large residuals?

    > *Your answer…*

    **2.3** Look at the per-landmark table and at the residual-vs-time plot. Where do the large residuals come from? Which explanation fits best: random sensor noise, occlusions, or an error in the map or in the measurement–landmark association? Formulate a hypothesis that you could test.

    > *Your answer…*
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ---
    ## 3 · EKF correction (≈ 30 min)

    After the prediction the belief is $\mathcal N(\bar\mu,\bar\Sigma)$. A range–bearing measurement $z=(r,\phi)$ of a known landmark corrects both the pose and its uncertainty in four steps (Thrun et al., Table 7.2):

    1. **Innovation** — compare the real and the predicted measurement: $\nu=z-h(\bar\mu,m)$, with the bearing component wrapped to $[-\pi,\pi)$.
    2. **Innovation covariance** — how uncertain the innovation is:
       $$S=H\bar\Sigma H^\top+Q,\qquad Q=\operatorname{diag}(\sigma_r^2,\sigma_\phi^2).$$
       $H\bar\Sigma H^\top$ is the uncertainty due to the robot pose, $Q$ the uncertainty of the sensor itself.
    3. **Kalman gain** — how strongly the measurement corrects the prediction; it maps a measurement-space error $(r,\phi)$ into a state correction $(x,y,\theta)$:
       $$K=\bar\Sigma H^\top S^{-1}.$$
    4. **Correction** — $\mu=\bar\mu+K\nu$ (wrap $\theta$), and $\Sigma=(I-KH)\bar\Sigma$. The measurement reduces the uncertainty in the directions it observes.

    `ekf_update(mu_bar, Sigma_bar, z, landmark, Q)`:

    | Input | Shape | Meaning | | Output | Shape | Meaning |
    |---|---:|---|---|---|---:|---|
    | `mu_bar` | `(3,)` | predicted pose $\bar\mu$ | | `mu` | `(3,)` | corrected pose $\mu$ |
    | `Sigma_bar` | `(3,3)` | predicted covariance $\bar\Sigma$ | | `Sigma` | `(3,3)` | corrected covariance $\Sigma$ |
    | `z` | `(2,)` | measured $[r,\phi]$ | | `nu` | `(2,)` | innovation |
    | `landmark` | `(2,)` | $[m_x,m_y]$ | | `S` | `(2,2)` | innovation covariance |
    | `Q` | `(2,2)` | measurement-noise covariance | | `K` | `(3,2)` | Kalman gain |

    Naming convention in the code: `Sigma` is always a **state** covariance ($\Sigma$, $\bar\Sigma$); `S` is only the **innovation** covariance.

    ### 🛠 TODO 3
    Implement `ekf_update(mu_bar, Sigma_bar, z, landmark, Q)` returning `(mu, Sigma, nu, S, K)`. Use your `h_landmark` and `H_landmark`. $S$ is only $2\times2$, so `np.linalg.inv(S)` is fine.
    """
    )
    return


@app.function
def ekf_update(mu_bar, Sigma_bar, z, landmark, Q):
    return None, None, None, None, None


@app.cell(hide_code=True)
def _(mo, np):
    def _check():
        try:
            _mu_bar = np.array([0.0, 0.0, np.pi - 0.02])
            _Sigma_bar = np.diag([0.1, 0.1, 0.05])
            _Q = np.diag([0.1, 0.05]) ** 2
            _l = (3.0, 0.1)  # expected bearing ≈ -3.088 rad; measured +3.132 rad (same direction)
            _mu, _Sigma, _nu, _S, _K = ekf_update(_mu_bar, _Sigma_bar, (3.0, np.pi - 0.01), _l, _Q)
            _r = [
                ("shapes: μ (3,), Σ (3,3), ν (2,), S (2,2), K (3,2)", _mu.shape == (3,) and _Sigma.shape == (3, 3) and _nu.shape == (2,) and _S.shape == (2, 2) and _K.shape == (3, 2)),
                ("innovation wrapped near ±π", abs(_nu[1]) < 0.5),
                ("heading stays wrapped", -np.pi <= _mu[2] < np.pi),
                ("uncertainty decreases (trace Σ < trace Σ̄)", np.trace(_Sigma) < np.trace(_Sigma_bar)),
                ("Σ symmetric", np.allclose(_Sigma, _Sigma.T, atol=1e-10)),
            ]
        except Exception as _e:
            return f"⏳ not implemented yet ({type(_e).__name__})"
        return "<br>".join(f"{'✅' if _ok else '❌'} {_n}" for _n, _ok in _r)

    mo.md("**CHECK — EKF update**<br>" + _check())
    return


@app.cell(hide_code=True)
def _(ekf_predict, np):
    # PROVIDED — EKF over the event stream. `gate(nu, S)` returns True to accept a measurement.
    def run_ekf(run, alphas, Q, gate=None):
        _mu = run["x0"].copy()
        _Sigma = np.diag([1e-6] * 3)
        _t = run["t0"]
        _u = (0.0, 0.0)
        _out = [[_t, *_mu]]
        _info = {"accepted": 0, "rejected": 0, "first_update": None}
        for _e in run["events"]:
            _dt = _e[0] - _t
            if _dt > 0:
                _mu, _Sigma = ekf_predict(_mu, _Sigma, _u[0], _u[1], _dt, alphas)
                if _mu is None:
                    return None, None
                _t = _e[0]
            if _e[1] == 0:
                _u = (_e[2], _e[3])
                _out.append([_t, *_mu])
                continue
            _res = ekf_update(_mu, _Sigma, (_e[3], _e[4]), run["landmarks"][_e[2]], Q)  # (mu, Sigma, nu, S, K)
            if _res[0] is None:
                return None, None
            if gate is not None and not gate(_res[2], _res[3]):
                _info["rejected"] += 1
                continue
            if _info["first_update"] is None:
                _info["first_update"] = dict(z=(_e[3], _e[4]), nu=_res[2], S=_res[3], K=_res[4])
            _mu, _Sigma = _res[0], _res[1]
            _info["accepted"] += 1
            _out.append([_t, *_mu])
        return np.array(_out), _info
    return (run_ekf,)


@app.cell(hide_code=True)
def _(mo):
    sr_sl = mo.ui.slider(0.01, 2.0, step=0.01, value=0.3, label="σ_r (m)", show_value=True)
    sp_sl = mo.ui.slider(0.01, 1.0, step=0.01, value=0.3, label="σ_φ (rad)", show_value=True)
    q_scale_dd = mo.ui.dropdown({"× 1": 1.0, "× 10": 10.0, "× 100": 100.0, "× 1000": 1000.0}, value="× 1", label="scale both σ")
    mo.vstack([
        mo.md(r"**Measurement noise** $Q=\mathrm{diag}(\sigma_r^2,\sigma_\phi^2)$ — the scale multiplies both standard deviations (to explore very large values)."),
        mo.hstack([sr_sl, sp_sl, q_scale_dd], justify="start"),
    ])
    return q_scale_dd, sp_sl, sr_sl


@app.cell(hide_code=True)
def _(np, q_scale_dd, sp_sl, sr_sl):
    Q = np.diag([(q_scale_dd.value * sr_sl.value) ** 2, (q_scale_dd.value * sp_sl.value) ** 2])
    return (Q,)


@app.cell(hide_code=True)
def _(Q, alphas, evaluate, mo, np, pd, plot_errors, pred_states, run, run_ekf):
    ekf_plain, info_plain = run_ekf(run, alphas, Q)
    if ekf_plain is None or pred_states is None:
        _out = mo.md("⏳ Implement TODOs 1–3 first.")
    else:
        _f = info_plain["first_update"]
        _fmt = lambda a: np.array2string(np.asarray(a), precision=4, suppress_small=True)
        _out = mo.vstack([
            pd.DataFrame([
                {"method": "dead reckoning", **evaluate(pred_states, run["gt"])},
                {"method": "EKF (no gating: every measurement is used)", **evaluate(ekf_plain, run["gt"])},
            ]).round(3),
            mo.md(f"**First real update** — z = `{_fmt(_f['z'])}`, innovation ν = `{_fmt(_f['nu'])}`<br>S = `{_fmt(_f['S'])}`<br>K = `{_fmt(_f['K'])}`"),
            mo.ui.plotly(plot_errors(run, {"DR": pred_states, "EKF no gating": ekf_plain})),
        ])
    _out
    return (ekf_plain,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    #### ✍️ ANSWER 3

    **3.1** Using $K=\bar\Sigma H^\top S^{-1}$ (not only a plot): what happens to the correction when $\sigma_r,\sigma_\phi$ are very large? And when $\alpha$ is very large? Check it with the sliders and the σ scale. How large must σ be before the EKF behaves like dead reckoning, and why is it so large?

    > *Your answer…*

    **3.2** With the default settings, is the EKF clearly better than dead reckoning? Look at the table and the error plot. If not, formulate a hypothesis using what you saw in Part 2.

    > *Your answer…*
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ---
    ## 4 · Rejecting incompatible measurements: validation gating (≈ 25 min)

    If the filter is correct, the innovation satisfies $\nu\sim\mathcal N(0,S)$, so its squared **Mahalanobis distance**

    $$
    d^2=\nu^\top S^{-1}\nu
    $$

    follows a $\chi^2$ distribution with 2 degrees of freedom (the same idea as the NEES of Part 1, now applied to the innovation, which can be computed online without ground truth). A measurement with $d^2$ above a threshold (5.99 → 95 %, 9.21 → 99 %, 13.82 → 99.9 %) is very unlikely under the model, and it is **rejected before the update**. This is called a **validation gate** (Siegwart & Nourbakhsh, Ch. 5). "No gating" means that every measurement is used.

    ### 🛠 TODO 4
    Implement `accept_measurement(nu, S, threshold)` → `True` if $d^2\le$ threshold, `False` otherwise.
    """
    )
    return


@app.function
def accept_measurement(nu, S, threshold):
    return None


@app.cell(hide_code=True)
def _(mo, np):
    try:
        _S = np.diag([0.04, 0.01])
        if accept_measurement(np.array([0.2, 0.1]), _S, 9.21) is None:
            raise NotImplementedError
        _ok = bool(accept_measurement(np.array([0.2, 0.1]), _S, 9.21)) and not bool(accept_measurement(np.array([1.0, 0.0]), _S, 9.21))
        _msg = f"{'✅' if _ok else '❌'} **CHECK:** d² = 2 accepted, d² = 25 rejected (threshold 9.21)."
    except Exception as _e:
        _msg = f"⏳ **CHECK:** not implemented yet ({type(_e).__name__})."
    mo.md(_msg)
    return


@app.cell(hide_code=True)
def _(mo):
    gate_dd = mo.ui.dropdown({"95 % (5.99)": 5.99, "99 % (9.21)": 9.21, "99.9 % (13.82)": 13.82}, value="99 % (9.21)", label="Gate threshold")
    gate_dd
    return (gate_dd,)


@app.cell(hide_code=True)
def _(
    Q,
    alphas,
    ekf_plain,
    evaluate,
    gate_dd,
    mo,
    np,
    pd,
    plot_errors,
    plot_trajectories,
    pred_states,
    run,
    run_ekf,
):
    ekf_gated, info_gated = run_ekf(run, alphas, Q, gate=lambda nu, S: accept_measurement(nu, S, gate_dd.value))
    if ekf_gated is None or ekf_plain is None or accept_measurement(np.zeros(2), np.eye(2), 1.0) is None:
        _out = mo.md("⏳ Implement TODOs 1–4 first.")
    else:
        _tab = pd.DataFrame([
            {"method": "dead reckoning", **evaluate(pred_states, run["gt"])},
            {"method": "EKF (no gating)", **evaluate(ekf_plain, run["gt"])},
            {"method": f"EKF (gate {gate_dd.value})", **evaluate(ekf_gated, run["gt"]), "rejected": info_gated["rejected"], "accepted": info_gated["accepted"]},
        ]).round(3)
        _est = {"DR": pred_states, "EKF no gating": ekf_plain, "EKF gated": ekf_gated}
        _out = mo.vstack([_tab, mo.ui.plotly(plot_trajectories(run, _est, title="DR vs EKF")), mo.ui.plotly(plot_errors(run, _est))])
    _out
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    #### ✍️ ANSWER 4

    **4.1** Compare ATE/FTE with and without gating, and the number of rejected measurements. Does it confirm your hypothesis of 3.2? Compare the number of rejected measurements with what you found in 2.3.

    > *Your answer…*

    **4.2** Gating can fail. Set α = (0.05, 0.005, 0.005, 0.05), σ_r = σ_φ = 0.1 and run Robot5 of Dataset1. What happens to the number of rejected measurements and to the error? Explain the mechanism (hint: what does an overconfident $\bar\Sigma$ do to $S$ and to $d^2$?).

    > *Your answer…*
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ---
    ## 5 · Sensitivity across robots

    With the current sliders as **baseline** (and gating at the selected threshold), the button runs Robot1 and Robot5 of the selected dataset with (i) the baseline, (ii) process noise × 10 (all α), (iii) measurement noise × 3 in σ. It can be completed at home.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    sens_btn = mo.ui.run_button(label="Run sensitivity experiment")
    sens_btn
    return (sens_btn,)


@app.cell(hide_code=True)
def _(
    Q,
    alphas,
    dataset_dd,
    duration_sl,
    evaluate,
    gate_dd,
    load_run,
    mo,
    pd,
    run_ekf,
    sens_btn,
):
    mo.stop(not sens_btn.value, mo.md("Press the button to run (≈ 10–20 s)."))
    _rows = []
    for _rb in ["Robot1", "Robot5"]:
        _r = load_run(dataset_dd.value, _rb, duration_sl.value)
        for _name, _a, _Q in [("baseline", alphas, Q), ("α × 10", tuple(10 * a for a in alphas), Q), ("σ × 3", alphas, 9 * Q)]:
            _s, _i = run_ekf(_r, _a, _Q, gate=lambda nu, S: accept_measurement(nu, S, gate_dd.value))
            if _s is None:
                break
            _rows.append({"robot": _rb, "config": _name, **{k: round(v, 3) for k, v in evaluate(_s, _r["gt"]).items()}, "rejected": _i["rejected"]})
    pd.DataFrame(_rows) if _rows else mo.md("⏳ Implement TODOs 1–4 first.")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    #### ✍️ ANSWER 5

    **5.1** Do the same parameters work equally well for both robots? Find one change that improves one metric (or one robot) but worsens another.

    > *Your answer…*

    **5.2** Ground truth was used for ATE/FTE, for the residual analysis and for the initial pose. Is it used anywhere inside the EKF updates? Why must it not be?

    > *Your answer…*

    ---
    **Keep this notebook.** In E1 you will paste `G_jacobian`, `V_jacobian`, `control_noise`, `ekf_predict`, `h_landmark`, `H_landmark`, `ekf_update` and `accept_measurement`.
    """
    )
    return


if __name__ == "__main__":
    app.run()
