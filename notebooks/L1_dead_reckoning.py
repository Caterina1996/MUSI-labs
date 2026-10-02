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
    # L1 — Motion Models and Dead Reckoning

    **11765 · Navigation and Environment Modelling in Mobile Robotics · MUSI** · Class exercise, Friday 02/10/2026

    This is a **formative** exercise: it is not graded, but the functions you write here (`wrap_angle`, `motion_euler`, `motion_exact`, `position_errors`) are reused in **E1**. Keep your completed notebook.

    ### Learning objectives
    1. Distinguish the data a robot has online from the ground truth used only for evaluation.
    2. Implement the velocity motion model with Euler and exact (circular-arc) integration.
    3. Quantify dead-reckoning drift with ATE, FTE and heading error, aligning estimates and ground truth **in time**.
    4. Compare robots, datasets and sampling intervals through controlled (one-factor-at-a-time) experiments.

    ### Before the session
    Thrun, Burgard & Fox, *Probabilistic Robotics*, Sec. 5.3 (velocity motion model) · Siegwart & Nourbakhsh, *Introduction to Autonomous Mobile Robots*, Ch. 5 (odometry error model: systematic vs. non-systematic errors).

    ### How to work
    **🛠 TODO** cells: write code · **✅ CHECK** cells: automatic feedback · **✍️ ANSWER** cells: double-click and write a short answer **with numbers from your runs**.
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
            duration_s=duration_s,
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
    ## 1 · Explore the MRCLAM data (≈ 20 min)

    UTIAS MRCLAM (Leung et al., 2011): 5 iRobot Create robots in a 15 m × 8 m room with 15 cylindrical landmarks identified by barcodes.

    | File | Content | Available online on the robot? |
    |---|---|---|
    | `RobotN_Odometry.dat` | `[t, v, ω]` — velocity **commands** | yes |
    | `RobotN_Measurement.dat` | `[t, barcode, range, bearing]` (camera) | yes |
    | `RobotN_Groundtruth.dat` | `[t, x, y, θ]` (Vicon motion capture) | **no — evaluation only** |
    | `Landmark_Groundtruth.dat` | landmark positions | only if a map is given |

    The controls below drive the whole notebook. The provided `load_run` already sorts odometry and measurements by time and discards measurements of other robots.
    """
    )
    return


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
    mo.ui.plotly(plot_trajectories(run, {}, title="Ground-truth trajectory and landmarks"))
    return


@app.cell(hide_code=True)
def _(load_run, mo, path_length, pd):
    # PROVIDED — the three cases of the Task 1 experiment (300 s each)
    _rows = []
    for _ds, _rb in [("MRCLAM_Dataset1", "Robot1"), ("MRCLAM_Dataset1", "Robot5"), ("MRCLAM_Dataset3", "Robot1")]:
        _r = load_run(_ds, _rb, 300)
        _dur = _r["gt"][-1, 0] - _r["gt"][0, 0]
        _rows.append(
            {
                "case": f"{_ds[-8:]} / {_rb}",
                "path (m)": round(path_length(_r["gt"]), 1),
                "landmark meas. / min": round(60 * len(_r["meas"]) / _dur, 1),
                "mean |ω| cmd (rad/s)": round(float(abs(_r["odom"][:, 2]).mean()), 3),
            }
        )
    overview_df = pd.DataFrame(_rows)
    mo.vstack([mo.md("**Task 1 experiment** — same robot vs. different robot, same dataset vs. different dataset"), overview_df])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### 🛠 TODO 1 — sampling rates
    Using `run["odom"]` and `run["gt"]`, compute the mean sampling frequency (Hz) of the odometry commands and of the ground truth.
    """
    )
    return


@app.cell
def _(np, run):
    f_odom = None  # Hz
    f_gt = None  # Hz
    return f_gt, f_odom


@app.cell(hide_code=True)
def _(f_gt, f_odom, mo, np, run):
    # Validate values independently; merely displaying them is not a CHECK.
    _ref_odom = 1.0 / np.mean(np.diff(run["odom"][:, 0]))
    _ref_gt = 1.0 / np.mean(np.diff(run["gt"][:, 0]))
    if f_odom is None or f_gt is None:
        _msg = "⏳ **CHECK:** not implemented yet."
    else:
        try:
            _ok = (np.isfinite(f_odom) and np.isfinite(f_gt)
                   and np.isclose(f_odom, _ref_odom, rtol=0.01)
                   and np.isclose(f_gt, _ref_gt, rtol=0.01))
            _msg = (f"{'✅' if _ok else '❌'} **CHECK:** "
                    f"odometry = {f_odom:.2f} Hz · ground truth = {f_gt:.2f} Hz. "
                    "The two streams are **not** synchronised.")
        except (TypeError, ValueError):
            _msg = "❌ **CHECK:** return two numerical frequencies in Hz."
    mo.md(_msg)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    #### ✍️ ANSWER 1

    **1.1 Which information would a real robot have online, and which exists only to evaluate the algorithm?**

    > *Your answer…*

    **1.2 Which probabilistic motion model is more natural for these data: the velocity motion model or the odometry motion model? Explain your answer in terms of the inputs required by each model.**
    > *Your answer…*

    **1.3 Why might the robot's actual trajectory differ from the trajectory predicted using commanded velocities? Identify at least two possible sources of error.**
    > *Your answer…*

    **1.4** Before running dead reckoning: which of the three cases in the table do you expect to be hardest, and why?
    > *Your answer…*
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ---
    ## 2 · The velocity motion model (≈ 35 min)

    State $x=(x,y,\theta)^\top$, control $(v,\omega)$ held constant for $\Delta t$.

    **Euler:** $\;x'=x+v\Delta t\cos\theta,\quad y'=y+v\Delta t\sin\theta,\quad \theta'=\theta+\omega\Delta t$

    **Exact (circular arc of radius $v/\omega$):**

    $$
    x'=x-\tfrac{v}{\omega}\sin\theta+\tfrac{v}{\omega}\sin(\theta+\omega\Delta t),\qquad
    y'=y+\tfrac{v}{\omega}\cos\theta-\tfrac{v}{\omega}\cos(\theta+\omega\Delta t),\qquad
    \theta'=\theta+\omega\Delta t
    $$

    ### 🛠 TODO 2
    Implement the three functions. The pose is `pose = np.array([x, y, theta])` (m, m, rad); the controls `v` (m/s) and `w` (rad/s) are constant during `dt` (s).

    - `wrap_angle(a)`: map an angle, or an array of angles, to $[-\pi,\pi)$. Example: $3\pi/2 \to -\pi/2$.
    - `motion_euler(pose, v, w, dt)`: next pose with the Euler equations above.
    - `motion_exact(pose, v, w, dt)`: next pose with the circular-arc equations above. If $|\omega| < 10^{-6}$ the radius $v/\omega$ is undefined: use straight-line motion instead.

    Both motion functions must return a **new** array (do not modify `pose`) with $\theta$ wrapped by `wrap_angle`. You will reuse the three functions in L2, L3 and E1.
    """
    )
    return


@app.cell
def _(np):
    def wrap_angle(a):
        return None

    def motion_euler(pose, v, w, dt):
        return None

    def motion_exact(pose, v, w, dt):
        return None

    return motion_euler, motion_exact, wrap_angle


@app.cell(hide_code=True)
def _(mo, motion_euler, motion_exact, np, wrap_angle):
    def _check():
        _msgs = []
        try:
            _ok = np.allclose(wrap_angle(np.array([3 * np.pi / 2, -3 * np.pi / 2, 0.1])), [-np.pi / 2, np.pi / 2, 0.1])
            _msgs.append(("wrap_angle", _ok))
            _x0 = np.array([0.0, 0.0, 0.0])
            _msgs.append(("Euler, straight line", np.allclose(motion_euler(_x0, 1.0, 0.0, 1.0), [1, 0, 0])))
            _msgs.append(("Euler, quarter turn", np.allclose(motion_euler(_x0, 1.0, np.pi / 2, 1.0), [1, 0, np.pi / 2])))
            _msgs.append(("exact, straight line (ω = 0)", np.allclose(motion_exact(_x0, 1.0, 0.0, 1.0), [1, 0, 0])))
            _msgs.append(("exact, quarter circle", np.allclose(motion_exact(_x0, 1.0, np.pi / 2, 1.0), [2 / np.pi, 2 / np.pi, np.pi / 2])))
            _msgs.append(("Euler, θ wrapped", abs(motion_euler(np.array([0, 0, 3.1]), 0.0, 1.0, 0.1)[2] + 3.083) < 1e-3))
            _msgs.append(("exact, θ wrapped", abs(motion_exact(np.array([0, 0, 3.1]), 0.1, 1.0, 0.1)[2] + 3.083) < 1e-3))
            _p = np.array([1.0, 2.0, 0.5])
            motion_euler(_p, 1.0, 0.3, 0.5)
            motion_exact(_p, 1.0, 0.3, 0.5)
            _msgs.append(("input pose not modified", np.array_equal(_p, [1.0, 2.0, 0.5])))
        except Exception as _e:
            return f"⏳ **CHECK:** not implemented yet ({type(_e).__name__})."
        return "<br>".join(f"{'✅' if _ok else '❌'} {_n}" for _n, _ok in _msgs)

    mo.md("**CHECK — motion models**<br>" + _check())
    return


@app.cell(hide_code=True)
def _(np):
    # PROVIDED — dead reckoning. Zero-order hold: each command is applied from its
    # timestamp until the next one. `subsample=k` keeps only one command out of k.
    def dead_reckoning(run, model, subsample=1):
        _x = run["x0"].copy()
        _t = run["t0"]
        _u = (0.0, 0.0)
        _out = [[_t, *_x]]
        _k = 0
        # Use a common, ground-truth-supported endpoint at every subsampling rate.
        _t_end = min(run["t0"] + run["duration_s"], run["gt"][-1, 0])
        for _e in run["events"]:
            if _e[1] != 0:
                continue
            if _e[0] > _t_end:
                break
            _k += 1
            if (_k - 1) % subsample:
                continue
            _dt = _e[0] - _t
            if _dt > 0:
                _x = model(_x, _u[0], _u[1], _dt)
                if _x is None:
                    return None
                _t = _e[0]
                _out.append([_t, *_x])
            _u = (_e[2], _e[3])
        if _t_end > _t:
            _x = model(_x, _u[0], _u[1], _t_end - _t)
            if _x is None:
                return None
            _out.append([_t_end, *_x])
        return np.array(_out)

    return (dead_reckoning,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ---
    ## 3 · Dead reckoning and drift (≈ 35 min)

    $\text{ATE}=\sqrt{\tfrac1N\sum_i\lVert p_i-p^{gt}(t_i)\rVert^2}$ (average error) · $\text{FTE}=\lVert p_N-p^{gt}(t_N)\rVert$ (final error).

    Odometry (~66 Hz) and ground truth (~50 Hz) are sampled at **different instants**: comparing `est[i]` with `gt[i]` compares different times. You must evaluate the ground truth **at the estimate timestamps**.

    ### 🛠 TODO 3
    Implement `position_errors(states, gt)`: for each row `[t, x, y, θ]` of `states`, the Euclidean position error with respect to the ground truth linearly interpolated at `t` (use `np.interp`). ATE and FTE follow directly from it.
    """
    )
    return


@app.cell
def _(np):
    def position_errors(states, gt):
        return None

    return (position_errors,)


@app.cell(hide_code=True)
def _(dead_reckoning, motion_euler, motion_exact, run):
    dr_euler = dead_reckoning(run, motion_euler)
    dr_exact = dead_reckoning(run, motion_exact)
    return dr_euler, dr_exact


@app.cell(hide_code=True)
def _(dr_euler, dr_exact, evaluate, mo, np, pd, position_errors, run):
    if dr_euler is None or dr_exact is None:
        _out = mo.md("⏳ Implement TODO 2 first.")
    else:
        _e = position_errors(dr_euler, run["gt"])
        if _e is None:
            _chk = "⏳ **CHECK:** `position_errors` not implemented yet."
        else:
            _ref = evaluate(dr_euler, run["gt"])
            _ok = abs(np.sqrt(np.mean(_e**2)) - _ref["ATE (m)"]) < 1e-6 and abs(_e[-1] - _ref["FTE (m)"]) < 1e-6
            _chk = f"{'✅' if _ok else '❌'} **CHECK:** your ATE/FTE {'match' if _ok else 'do NOT match'} the reference evaluation."
        _tab = pd.DataFrame([{"model": "Euler", **evaluate(dr_euler, run["gt"])}, {"model": "exact", **evaluate(dr_exact, run["gt"])}]).round(3)
        _out = mo.vstack([mo.md(_chk), _tab, mo.md(f"Max. distance between Euler and exact trajectories: **{np.linalg.norm(dr_euler[:, 1:3] - dr_exact[:, 1:3], axis=1).max():.4f} m**")])
    _out
    return


@app.cell(hide_code=True)
def _(dr_euler, dr_exact, mo, plot_errors, plot_trajectories, run):
    mo.vstack(
        [
            mo.ui.plotly(plot_trajectories(run, {"DR Euler": dr_euler, "DR exact": dr_exact}, title="Dead reckoning vs ground truth")),
            mo.ui.plotly(plot_errors(run, {"DR Euler": dr_euler})),
        ]
    ) if dr_euler is not None else mo.md("")
    return


@app.cell(hide_code=True)
def _(mo):
    sub_sl = mo.ui.slider(1, 100, step=1, value=1, label="Keep one command out of k", show_value=True)
    mo.vstack([mo.md("**Experiment — temporal subsampling.** Simulate a slower command stream."), sub_sl])
    return (sub_sl,)


@app.cell(hide_code=True)
def _(dead_reckoning, evaluate, mo, motion_euler, motion_exact, np, pd, run, sub_sl):
    _a = dead_reckoning(run, motion_euler, sub_sl.value)
    _b = dead_reckoning(run, motion_exact, sub_sl.value)
    if _a is None or _b is None:
        _out = mo.md("⏳ Implement TODO 2 first.")
    else:
        _dt = np.diff(run["odom"][:: sub_sl.value, 0]).mean()
        _out = mo.vstack(
            [
                mo.md(f"k = {sub_sl.value} → mean Δt = {_dt:.3f} s · max. distance between Euler and exact = **{np.linalg.norm(_a[:, 1:3] - _b[:, 1:3], axis=1).max():.3f} m**"),
                pd.DataFrame([{"model": "Euler", **evaluate(_a, run["gt"])}, {"model": "exact", **evaluate(_b, run["gt"])}]).round(3),
            ]
        )
    _out
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    #### ✍️ ANSWER 3

    **3.1** Why is Euler only an approximation when the robot turns? What happens as $\Delta t\to0$? Relate it to the size of $\omega\Delta t$ in these data.

    > *Your answer…*

    **3.2** At the native rate, is the drift dominated by numerical integration or by imperfect motion information? Support it with numbers (ATE of both models, their difference, subsampling results).

    > *Your answer…*

    **3.3** Look at the heading-error plot. Approximately how much lateral error does a constant heading error of 5° produce after 10 m? What does this tell you about which error component dominates the position drift?

    > *Your answer…*
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ---
    ## 4 · Does it generalise? (≈ 20 min)

    The same implementation and the same settings are run on the five robots of the **selected dataset**. Then change the dataset in the controls at the top and compare. Press the button to run (it takes a few seconds).
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    gen_btn = mo.ui.run_button(label="Run on the 5 robots")
    gen_btn
    return (gen_btn,)


@app.cell(hide_code=True)
def _(dataset_dd, dead_reckoning, duration_sl, evaluate, gen_btn, load_run, mo, motion_euler, path_length, pd):
    mo.stop(not gen_btn.value, mo.md("Press the button to run."))
    _rows = []
    for _rb in [f"Robot{_j}" for _j in range(1, 6)]:
        _r = load_run(dataset_dd.value, _rb, duration_sl.value)
        _s = dead_reckoning(_r, motion_euler)
        if _s is None:
            break
        _rows.append({"robot": _rb, "path (m)": round(path_length(_r["gt"]), 1), **{k: round(v, 3) for k, v in evaluate(_s, _r["gt"]).items()}})
    gen_df = pd.DataFrame(_rows)
    gen_df
    return


@app.cell(hide_code=True)
def _(go, make_subplots, mo, np, run):
    # PROVIDED — commands vs. approximate signed velocities derived from ground truth (0.5 s windows)
    _gt = run["gt"]
    _T = np.arange(_gt[0, 0] + 1, _gt[-1, 0] - 1, 0.5)
    _x = np.interp(_T, _gt[:, 0], _gt[:, 1])
    _y = np.interp(_T, _gt[:, 0], _gt[:, 2])
    _th = np.interp(_T, _gt[:, 0], np.unwrap(_gt[:, 3]))
    _tm = _T[:-1] + 0.25
    # Project world-frame displacement onto the robot heading, retaining sign.
    _th_mid = (_th[:-1] + _th[1:]) / 2
    _vg = (np.diff(_x) * np.cos(_th_mid) + np.diff(_y) * np.sin(_th_mid)) / 0.5
    _wg = np.diff(_th) / 0.5
    # window averages of the commands (zero-order hold), not point samples
    _od = run["odom"]
    _cs = np.concatenate([[0.0], np.cumsum(_od[:-1, 1] * np.diff(_od[:, 0]))])
    _cw = np.concatenate([[0.0], np.cumsum(_od[:-1, 2] * np.diff(_od[:, 0]))])
    _vc = np.diff(np.interp(_T, _od[:, 0], _cs)) / 0.5
    _wc = np.diff(np.interp(_T, _od[:, 0], _cw)) / 0.5
    _fig = make_subplots(rows=2, cols=1, shared_xaxes=True, subplot_titles=("v (m/s)", "ω (rad/s)"))
    _t = _tm - run["t0"]
    _fig.add_trace(go.Scatter(x=_t, y=_vc, name="command v", line=dict(width=1)), row=1, col=1)
    _fig.add_trace(go.Scatter(x=_t, y=_vg, name="ground-truth v", line=dict(width=1, dash="dot")), row=1, col=1)
    _fig.add_trace(go.Scatter(x=_t, y=_wc, name="command ω", line=dict(width=1)), row=2, col=1)
    _fig.add_trace(go.Scatter(x=_t, y=_wg, name="ground-truth ω", line=dict(width=1, dash="dot")), row=2, col=1)
    _fig.update_layout(height=450, plot_bgcolor="white")
    mo.vstack(
        [
            mo.ui.plotly(_fig),
            mo.md(f"Mean (command − estimated ground truth): v = **{np.mean(_vc - _vg) * 1000:.1f} mm/s**, ω = **{np.degrees(np.mean(_wc - _wg)):.3f} °/s** over {_t[-1]:.0f} s."),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    #### ✍️ ANSWER 4

    **4.1** Does a longer path always produce a larger FTE? A larger ATE? Is FTE always larger than ATE? Use the five-robot table.

    > *Your answer…*

    **4.2** Why can the same motion model behave so differently for different robots or datasets? Use the velocity plots: do the command errors look like zero-mean white noise or do they show a bias? (Siegwart & Nourbakhsh: *systematic* vs. *non-systematic* errors.)

    > *Your answer…*

    **4.3 (bridge to L2)** Dead reckoning predicts but never corrects. Which information in MRCLAM could bound the error, and what would we need to know to use it?

    > *Your answer…*

    ---
    **Keep this notebook.** In E1 you will paste `wrap_angle`, `motion_euler`, `motion_exact` and `position_errors`.
    """
    )
    return


if __name__ == "__main__":
    app.run()