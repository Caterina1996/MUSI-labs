# MUSI Labs — L1: Motion Models and Dead Reckoning

Class exercise for **11765 · Navigation and Environment Modelling in Mobile Robotics (MUSI)**.

In this formative (ungraded) exercise, you will explore the [UTIAS MRCLAM dataset](http://asrl.utias.utoronto.ca/datasets/mrclam/), implement two velocity motion models, measure dead-reckoning drift, and compare robots and datasets. **Keep your completed notebook:** you will reuse your functions in E1.

## 1. Before you start: explore MRCLAM

Visit the [official MRCLAM dataset website](http://asrl.utias.utoronto.ca/datasets/mrclam/) before opening the notebook. Identify the robot odometry commands, range/bearing landmark measurements, ground truth, and landmark map. Consider which information is available to a robot during operation and which is provided only for evaluation.

## 2. Install prerequisites

Install [Git](https://git-scm.com/downloads/win) and [uv](https://docs.astral.sh/uv/getting-started/installation/). You can work from **Git Bash**, **PowerShell**, or a Linux/macOS terminal.

**Windows PowerShell:**

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**Git Bash / Linux / macOS:**

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Open a new terminal, then confirm that `uv --version` works. See the [uv installation guide](https://docs.astral.sh/uv/getting-started/installation/) if the command is not found.

## 3. Clone the repository and install dependencies

```bash
git clone https://github.com/Caterina1996/MUSI-labs.git
cd MUSI-labs
uv python install 3.10
uv sync
```

The `uv sync` step assumes this repository includes its `pyproject.toml` with the notebook dependencies (`marimo`, `numpy`, `pandas`, and `plotly`). If your copy does not include this configuration, ask your instructor for the project files.

## 4. Check the dataset files

The notebook looks for `data/MRCLAM_Dataset1/` (also one or two directories above the working directory). It requires the relevant `Barcodes.dat`, `Landmark_Groundtruth.dat`, and per-robot odometry, measurement, and ground-truth files.

Check that the datasets are present before launching. The notebook **always** uses datasets 1 and 3 for its initial comparison, even when another dataset is selected. The dataset selector offers datasets 1–9; all selected datasets must be downloaded and placed under `data/` if they are not distributed with the repository. Do not assume `git pull` downloads datasets hosted elsewhere.

## 5. Open and complete the notebook

Run the following **from the repository root** (the directory containing `data/`):

```bash
uv run marimo edit L1_dead_reckoning.py
```

If your instructor places the notebook in `notebooks/` instead of the repository root, use `uv run marimo edit notebooks/L1_dead_reckoning.py`.

Work through the notebook in order:

1. **Explore MRCLAM:** inspect the data and calculate odometry and ground-truth sampling frequencies.
2. **Implement the motion models:** complete `wrap_angle`, `motion_euler`, and `motion_exact`.
3. **Study dead-reckoning drift:** implement `position_errors`; compare Euler and exact integration and experiment with temporal subsampling.
4. **Test generalisation:** compare the five robots and examine differences between velocity commands and motion estimated from ground truth.

Complete each **🛠 TODO** code cell and **✍️ ANSWER** question, using numerical results from your own experiments. Use the **✅ CHECK** messages to check your implementations. The notebook deliberately starts with unfinished functions.

## Updating your copy

From inside `MUSI-labs`:

```bash
git pull
uv sync
```

If you have edited the notebook, **save or commit your work before pulling**. Avoid blindly stashing and reapplying changes if you are unfamiliar with resolving Git conflicts.

## Troubleshooting

- **`uv` not found:** restart the terminal and follow the [uv installation instructions](https://docs.astral.sh/uv/getting-started/installation/).
- **Marimo does not start:** run `uv sync` and `uv run marimo --version`. Check that Marimo is declared in `pyproject.toml`.
- **Dataset not found:** launch from the repository root and verify `data/MRCLAM_Dataset1/` and `data/MRCLAM_Dataset3/` contain the required files.
- **A CHECK says “not implemented yet”:** complete the earlier TODO functions first; later calculations depend on them.

## References

- [UTIAS MRCLAM dataset](http://asrl.utias.utoronto.ca/datasets/mrclam/)
- Thrun, Burgard & Fox, *Probabilistic Robotics*, §5.3.
- Siegwart & Nourbakhsh, *Introduction to Autonomous Mobile Robots*, Ch. 5.
- [Marimo documentation](https://docs.marimo.io/)
