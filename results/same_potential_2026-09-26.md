# Same-potential eSEN teacher pilot (26 September 2026)

## Setup

The eSEN-SM OMol checkpoint's **energy head** defines the teacher potential. Forces are its position derivatives; velocity Verlet at 0.5 fs defines the fine teacher trajectory. The PDD student starts from the **same checkpoint's direct-force head**, cloned into four heads, and learns on-policy mean-velocity targets from that teacher. AD-3 train/test frames provide initial phase states only. The teacher and student therefore target the same potential in this experiment.

Local Mac CPU pilot: force-head adaptation, 200 updates with batch size 4 and 2,048 train frames; PDD, 200 updates with batch size 4, L in {1,2,4}, at most two on-policy prefix blocks. Held-out test states are disjoint from train states. All timing is wall clock on this Mac CPU and includes per-step graph construction and integration. The fine and coarse Verlet references now cache their force between steps (N+1 force/gradient evaluations for N steps).

## Eight fine steps (4 fs), eight held-out initial states

| Method | Finite | Mean endpoint position RMSE (Å) | Mean absolute energy drift (kJ/mol) | Mean wall time (s) | Ratio vs fine teacher |
|---|---:|---:|---:|---:|---:|
| PDD L1 | 8/8 | 0.00198 | 6.51 | 0.344 | 2.36× |
| PDD L2 | 8/8 | 0.00368 | 23.78 | 0.175 | 4.62× |
| PDD L4 | 8/8 | 0.00794 | 60.12 | 0.0855 | 9.45× |
| Coarse Verlet L4 | 8/8 | 0.00250 | 3.31 | 0.266 | 3.00× |

The L4 student is faster, but coarse Verlet is more accurate and has much less energy drift. The current PDD model does not dominate this essential baseline.

### Colab T4 replication

The same experiment was rerun in [the saved Colab notebook](https://colab.research.google.com/drive/1cu_gd8PUYz4fgEVpNHXLB3mdIZxokhSX) on a free Tesla T4: 100 force-head updates and 200 PDD updates, batch size 4, 2,048 training frames. Eight-step evaluation used the same eight held-out indices. PDD L4 was finite 8/8, with mean endpoint position RMSE **0.00794 Å**, mean absolute energy drift **60.16 kJ/mol**, mean wall time **0.0631 s**, and **10.64×** speed ratio to the fine energy-gradient teacher. Coarse Verlet L4 had **0.00250 Å**, **3.31 kJ/mol**, **0.2128 s**, and **3.05×**, respectively. The result closely matches the CPU accuracy finding. Notebook cells 29–33 contain the data preparation, training, evaluation, and diagnostic commands and outputs; the Colab VM checkpoints are ephemeral.

## Longer L4 rollout, four held-out initial states

Median position RMSE is 0.00855 Å at 8 steps, 0.0286 Å at 20 steps, and 0.103 Å at 40 steps. All four trajectories have exceeded 0.1 Å by step 40 (20 fs). At 80 steps (40 fs), one is nonfinite; the other three are finite but all exceed 0.1 Å, with one as large as 4.09e17 Å. Fused versus ordinary head arithmetic differs by at most 9.54e-7 Å in one block, ruling out head fusion as the failure source.

On the T4 replica, the four-trajectory 40-step median is also **0.103 Å** and all four exceed 0.1 Å. At 80 steps one is nonfinite; the remaining three all exceed 0.1 Å (one reaches 1.69e17 Å). GPU head fusion differs from ordinary arithmetic by at most 4.77e-7 Å over one block.

## Interpretation

This establishes a functioning same-potential training and evaluation path, and a short-horizon compute advantage. It does **not** yet establish useful MD acceleration: long rollouts fail and even the eight-step energy drift exceeds the coarse integrator's. The next useful work is training and architecture aimed at stability and energy conservation, evaluated against coarse Verlet at matched accuracy. The eSEN OMol potential itself has not been validated as a physically faithful AD-3 potential.

Raw JSON checkpoints/outputs are in local ignored `runs/esen-energy/`; commands to reproduce are in the README. The short test used `eval_200_8_fair.json` and the long diagnostic used `diagnose_200_80.json`.
