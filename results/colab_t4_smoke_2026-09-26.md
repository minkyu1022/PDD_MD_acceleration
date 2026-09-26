# Colab T4 GPU smoke test (2026-09-26)

Notebook: https://colab.research.google.com/drive/1cu_gd8PUYz4fgEVpNHXLB3mdIZxokhSX

## Environment

- Free Tesla T4, 15,360 MiB VRAM; CUDA available.
- Colab runtime image `2025.10`, Python 3.12.12, PyTorch initially 2.8.0+cu126.
- `fairchem-core==2.23.0` installed through `pip install -e '.[esen]'`. The kernel was restarted after installation because pip replaced PyTorch dependencies.
- The current/latest Colab image had Python 3.13.15 and rejected the package's declared `>=3.11,<3.13` range.

## Verified operations

1. An initialized small FAIR-Chem `eSCNMDBackbone` with `ParallelStudent`, two cloned force heads, and synthetic 3-atom coordinates completed CUDA forward and backward. Outputs both had shape `[1, 2, 3, 3]`; the first cloned force head had a non-null gradient. This checks tensor wiring, not pretrained eSEN weights or AD-3 accuracy.
2. Public Timewarp AD-3 train and test NPZ/PDB files downloaded in Colab. A CLI argument mismatch found during this step was fixed in commit `9f3b963`.
3. `train-force --backend tiny --device cuda --max-frames 64 --steps 3 --batch-size 2` completed. Step force RMSEs were 1.945, 1.739, and 1.911 eV/Å. The teacher versus stored-force MAE was 0.237 eV/Å on eight sampled states.
4. `train-pdd --device cuda --max-frames 64 --steps 4 --batch-size 2 --max-block 4 --block-sizes 1 2 4` completed and saved `runs/colab_smoke/pdd.pt` in the ephemeral Colab VM. The four reported losses were 4.042, 2.317, 2.378, and 2.035.
5. Held-out AD-3 test states evaluated for eight fine steps, two initial states, and blocks 1, 2, 4. All trajectories remained finite. Mean endpoint position RMSE was 0.118, 0.119, and 0.119 Å; student speed ratios to the fine OpenMM teacher were 0.389, 0.848, and 1.292, respectively. Coarse velocity Verlet at block 4 had 0.002 Å endpoint RMSE and a 3.776 speed ratio under this very short setting.

## Pretrained OMol25 eSEN check

The user-approved `esen-sm-direct-all-omol` checkpoint (49 MB) was uploaded to the ephemeral Colab session and loaded as a local checkpoint. With `--backend esen --device cuda`, a one-step force warm start on 16 AD-3 train states (batch size 1) completed and reported force RMSE 0.540 eV/Å. A two-update PDD run with blocks 1 and 2 then completed; its losses were 0.0355 and 0.0381. The checkpoint remained in the Colab VM; it was not committed to Git.

Held-out evaluation used two AD-3 test initial states and eight fine teacher steps:

| Method | Endpoint position RMSE (Å) | Endpoint velocity RMSE (Å/ps) | Finite fraction | Speed ratio to OpenMM fine teacher |
|---|---:|---:|---:|---:|
| eSEN PDD L1 | 0.0088 | 3.50 | 1.0 | 0.049 |
| eSEN PDD L2 | 0.0097 | 4.70 | 1.0 | 0.136 |
| Coarse Verlet L1 | 0.0000 | 0.00 | 1.0 | 0.988 |
| Coarse Verlet L2 | 0.0003 | 0.21 | 1.0 | 2.057 |

All numbers here are software smoke tests with at most two updates. They do not establish learned dynamics quality or a useful acceleration. The eSEN student is much more expensive than the current classical OpenMM teacher; a future speed comparison needs a matched eSEN energy-gradient MD teacher and larger batch sizes. Colab runtime storage is ephemeral, so rerun training or copy checkpoints to persistent storage before session shutdown.
