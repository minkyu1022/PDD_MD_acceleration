# L4, batch-16 same-potential learning curve on Colab T4

The teacher is velocity Verlet on the eSEN-SM OMol energy gradient. The student uses the matching checkpoint's direct-force backbone/head warm start. All runs use 0.5 fs fine steps, 2,048 AD-3 train initial states, the separate AD-3 test split, and a Tesla T4. The batch-16 experiment uses four randomly selected intra-block targets per head in expectation over 16 samples, with L fixed to 4 and up to two student prefix blocks. The comparison 200-update run used batch 4 and variable L in {1,2,4}; these differ in more than one training choice.

The eSEN teacher's on-policy mean-velocity targets are now computed in a single batched forward/energy-gradient pair per batch, instead of a separate pair per state. On two CPU samples, batched and serial targets differed by at most 3.6e-6 Å/ps in position mean velocity and 0.0071 Å/ps² in acceleration, within the numerical precision of the eSEN implementation. Batch 16 completed on the free T4. The 500-update L4/batch-16 run took 257 seconds, versus 154 seconds for the earlier 200-update variable-L/batch-4 run (different target implementation); per sampled state this is about 6× greater training throughput.

## Fixed held-out metrics

The fixed validation set has 64 evenly spaced AD-3 test initial states. `validate-pdd` computes teacher targets at **all four student-produced internal states**, separately for each head, with no optimizer update. Lower scaled PD loss is better. The eight-step rollout metrics use eight fixed test states. `q` is mean endpoint position RMSE against the fine eSEN teacher; drift is mean absolute total-energy drift under that potential.
These states come from one held-out trajectory, so they are correlated and do not establish transfer across molecules or thermodynamic conditions.

| Training | Updates | Mean scaled PD loss | Head 4 scaled loss | 8-step q RMSE (Å) | 8-step drift (kJ/mol) |
|---|---:|---:|---:|---:|---:|
| Variable L, batch 4 | 200 | 0.06354 | 0.14854 | 0.00794 | 60.16 |
| L4 only, batch 16 | 500 | 0.04150 | 0.08682 | 0.00874 | 12.84 |
| L4 only, batch 16 | 1,000 | 0.04359 | 0.09688 | 0.00887 | 11.21 |

The 500-update run improves held-out PD loss and energy drift, but does not improve short endpoint position accuracy. Another 500 updates do not improve these metrics substantially. The loss is dominated by acceleration error under the current `velocity_scale=10`, `accel_scale=10000` normalization; position mean-velocity error receives much less weight. This is a hypothesis for the position-error plateau, tested next by increasing its weight.

## Position mean-velocity loss ablation

Starting from the same 1,000-update checkpoint, I resumed L4/batch-16 training to 1,500 updates with either `velocity_scale=10` (control) or `velocity_scale=2.5` (16× the mean-velocity loss weight), leaving the acceleration scale at 10,000. Both runs reused the saved optimizer and NumPy sampling state. On the same 64 held-out states, the weighted model reduced head-4 mean-velocity RMSE from 0.942 to 0.669 Å/ps, while head-4 acceleration RMSE increased from 2,979 to 3,107 Å/ps². Scaled PD losses cannot be compared directly because their normalization differs.

| 1,500-update continuation | Head-4 mean-velocity RMSE (Å/ps) | 8-step q RMSE (Å) | 8-step drift (kJ/mol) | 40-step median q RMSE (Å) | 80-step median q RMSE (Å) |
|---|---:|---:|---:|---:|---:|
| Original weight (`velocity_scale=10`) | 0.942 | 0.00873 | 11.09 | 0.0755 | 0.7341 |
| Higher position weight (`velocity_scale=2.5`) | 0.669 | 0.00859 | 15.01 | 0.0751 | 0.1846 |

At 40 fine steps, both models kept all four test states below 0.1 Å. At 80 steps, all four positions were technically finite, but every state exceeded 0.1 Å for both models. The original-weight model had one catastrophic error of 2.6e26 Å; the higher-weight model's worst case was 106.6 Å. Finite output alone therefore does not mean a usable trajectory. The weighted loss improves its own velocity target and somewhat improves these 80-step errors, but sacrifices energy drift and still does not make a stable MD sampler. The 1,000-update checkpoint had much better 80-step errors on the three finite states, showing that more updates alone are not a reliable selection rule; validation should include long rollouts and drift.

## Long rollout, L4, four fixed test states

At 1,000 updates the median position RMSE is 0.0333 Å after 20 steps and 0.0771 Å after 40 steps; all four are below 0.1 Å at 40 steps. At 80 steps, one trajectory is nonfinite and the other three are between roughly 0.1 and 0.21 Å. The first 0.1 Å crossing occurs at fine steps 44–64. This is markedly more stable than the variable-L/batch-4 200-update pilot, whose four states all crossed 0.1 Å by step 40 and included enormous 80-step errors. The change cannot be attributed solely to batch size because training duration and block sampling also changed.

All commands and raw output are in the [saved Colab notebook](https://colab.research.google.com/drive/1cu_gd8PUYz4fgEVpNHXLB3mdIZxokhSX), cells 34–42. Checkpoint files in the Colab VM are ephemeral.
