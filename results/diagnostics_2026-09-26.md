# Failure horizon and eSEN compute baseline (2026-09-26)

We diagnosed the existing **1,200-update local eSEN PDD checkpoint** on eight held-out AD-3 initial states (2,048-frame test prefix, seed indices 0, 292, 584, 877, 1169, 1462, 1754, 2047). The student uses L4 blocks and 0.5 fs fine steps. These numbers are separate from the later **200-update Colab T4** run.

| Fine steps | Simulated time | Finite states | Median position RMSE vs OpenMM | Median velocity RMSE |
|---:|---:|---:|---:|---:|
| 4 | 2 fs | 8/8 | 0.00187 Å | 2.64 Å/ps |
| 8 | 4 fs | 8/8 | 0.00689 Å | 3.48 Å/ps |
| 20 | 10 fs | 8/8 | 0.0259 Å | 10.42 Å/ps |
| 40 | 20 fs | 8/8 | 0.0711 Å | 32.43 Å/ps |
| 80 | 40 fs | 1/8 | 6.51×10²¹ Å among the one finite trajectory | 5.64×10²⁴ Å/ps |

All eight trajectories exceed 0.1 Å position RMSE or become nonfinite by 80 fine steps. “Finite” only checks NaN/Inf; the lone finite endpoint is physically meaningless. This points to compounding rollout error, not a failure in the fused implementation: `advance` versus `advance_fused` differed by at most **4.77×10⁻⁷ Å** in position and **3.43×10⁻⁵ Å/ps** in velocity across the eight starting states. The diagnostic checks those two calculations after **one L4 block**.

We resumed the same PDD run from 1,200 to **1,600 updates**, retaining batch size 4, block sampling 1/2/4, and up to two on-policy prefix blocks. At 40 fine steps, median position RMSE shifted from 0.0711 Å to **0.0667 Å**. At 80 fine steps, the finite count rose from 1/8 to **4/8**, but **all eight** were either nonfinite or exceeded 0.1 Å. The first 0.1 Å crossing occurred between **48 and 64 fine steps** (24–32 fs), depending on the initial state. Thus the extra 400 updates did not solve long-rollout accuracy.

We also timed **eSEN energy-head gradients integrated with velocity Verlet** against this PDD student on the **same Mac CPU**, with three measured repeats after one warmup. The energy-gradient baseline needs 9 energy/gradient evaluations over 8 fine steps; PDD L4 needs 2 backbone evaluations. Both use the same pretrained eSEN-sm-direct checkpoint architecture and 22-atom AD-3 states.

| Independent trajectories | PDD L4 | eSEN energy-gradient Verlet | Energy/PDD compute ratio |
|---:|---:|---:|---:|
| 1 | 0.0869 s | 0.900 s | 10.35× |
| 8 | 0.517 s | 5.24 s | 10.15× |

**Scope:** This is a compute comparison, not an MD-acceleration result. The PDD student was trained to an **OpenMM AMBER14/OBC1** teacher, while the eSEN energy head defines a different learned potential. The gradient of the energy head also was not the force-training target of the direct-force eSEN checkpoint. These paths cannot be compared for trajectory accuracy or claimed as a faithful acceleration of eSEN MD. A conclusive experiment must train PDD against trajectories from the eSEN energy-gradient force and test accuracy and speed on the same potential, preferably on the target GPU. This CPU measurement establishes that there is a large compute gap worth testing.

Reproduce the diagnostics after obtaining the gated original eSEN checkpoint and the trained PDD checkpoint:

```bash
pdd-md diagnose --data-root data --pdd-checkpoint runs/esen-pilot/pdd_1200.pt --output runs/esen-pilot/diagnose_1200_L4.json --horizons 4 8 20 40 80 --block 4 --samples 8 --max-frames 2048 --device cpu
pdd-md benchmark-mlip --data-root data --pdd-checkpoint runs/esen-pilot/pdd_1200.pt --output runs/esen-pilot/benchmark_mlip_cpu.json --block 4 --fine-steps 8 --batch-sizes 1 8 --repeats 3 --max-frames 2048 --device cpu
```
