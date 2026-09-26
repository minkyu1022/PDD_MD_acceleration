# T4 eSEN energy-gradient compute benchmark (2026-09-26)

This run used the **1,600-update eSEN PDD L4 checkpoint** trained against the OpenMM teacher, the original **eSEN-sm-direct-all-omol** checkpoint's energy head, eight-step and 80-step intervals of 0.5 fs, and a Colab T4 GPU. Inputs came from the first 2,048 held-out AD-3 test states. Timers include eSEN graph construction, model evaluation, force differentiation for the energy baseline, velocity-Verlet integration, and GPU synchronization, with model loading and initial host-to-GPU copies excluded. Each cell used one warmup and then the median of 3 repeats for 8 steps or 2 repeats for 80 steps. The saved [Colab notebook](https://colab.research.google.com/drive/1cu_gd8PUYz4fgEVpNHXLB3mdIZxokhSX) contains the executed commands and outputs.

| Fine steps | Concurrent trajectories | PDD L4 time | eSEN energy-gradient Verlet time | Compute ratio (energy/PDD) | PDD endpoint finite | Energy endpoint finite |
|---:|---:|---:|---:|---:|---|---|
| 8 | 1 | 0.0543 s | 0.5520 s | 10.16× | yes | yes |
| 8 | 8 | 0.0612 s | 0.6164 s | 10.07× | yes | yes |
| 80 | 1 | 0.4802 s | 4.9259 s | 10.26× | no | yes |
| 80 | 8 | 0.6104 s | 5.7573 s | 9.43× | no | yes |
| 80 | 32 | 2.0477 s | 19.8443 s | 9.69× | no | yes |

PDD L4 uses 2 shared-backbone calls for eight fine steps or 20 for 80. Energy-gradient velocity Verlet uses 9 or 81 energy-forward/backward calls respectively, reusing the current force between steps.

**Interpretation:** These ratios measure a real compute opportunity on a T4, **not useful MD acceleration yet**. The current PDD student learned the OpenMM AMBER14/OBC1 step map, while eSEN's OMol energy head defines a different potential. It therefore has not been trained to reproduce the energy-gradient reference used in the timing comparison. Its 80-step endpoints were nonfinite in all measured batches; even the local eight-state evaluation of this 1,600-update checkpoint found every 80-step trajectory inaccurate or nonfinite. The direct-force eSEN checkpoint's energy head was not trained as a conservative force target, so a later scientific test should also inspect its energy-gradient force quality. The next decisive experiment is to generate eSEN energy-gradient teacher trajectories, train PDD on **that same** step map, then evaluate trajectory fidelity and timing together.

The CPU failure-horizon diagnosis and 1,200→1,600 update continuation are in [diagnostics_2026-09-26.md](diagnostics_2026-09-26.md).

```bash
pdd-md benchmark-mlip --data-root data --pdd-checkpoint /content/pdd_1600.pt --output runs/esen_colab_pilot/benchmark_mlip_t4.json --block 4 --fine-steps 8 --batch-sizes 1 8 --repeats 3 --max-frames 2048 --device cuda
pdd-md benchmark-mlip --data-root data --pdd-checkpoint /content/pdd_1600.pt --output runs/esen_colab_pilot/benchmark_mlip_t4_80.json --block 4 --fine-steps 80 --batch-sizes 1 8 32 --repeats 2 --max-frames 2048 --device cuda
```
