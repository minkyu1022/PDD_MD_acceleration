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

These are software smoke-test numbers from just three force and four PDD updates with the tiny adapter. They do not demonstrate a useful accuracy or speed improvement. The pretrained gated OMol25 eSEN checkpoint has not yet been run in Colab.
