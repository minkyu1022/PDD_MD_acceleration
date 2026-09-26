# PDD for molecular dynamics: AD-3 proof of concept

An executable experiment adapting [Parallel Decoding Distillation](https://arxiv.org/abs/2607.26004) to 22-atom alanine dipeptide. One shared MLIP backbone and several cloned direct-force heads predict the mean phase-space velocities of consecutive MD intervals in one backbone evaluation. The primary student is **OMol25 eSEN-sm-direct**; a small built-in equivariant model is provided for end-to-end smoke tests and an architecture ablation.

## What this experiment tests

We test whether a student can replace (L) evaluations of a deterministic fine-step teacher with one evaluation while following its short trajectory. Let (z=(q,v)), with (q) in Å and (v) in Å/ps. The teacher map (Phi_h(z)) is an OpenMM velocity-Verlet step of (h=0.0005) ps by default. The student returns (L) predictions ((\bar{\dot q}_k,\bar{\dot v}_k)) from the *same initial state* and a single shared eSEN backbone evaluation. Its internal states are cumulative sums of those predictions. At training time, we select a random internal step (k), evaluate the teacher map at the **student-produced state** with stop-gradient, and regress the corresponding head toward

\[
 \left((q' - \hat q_k)/h,\;(v' - \hat v_k)/h\right),\quad
 (q',v')=\Phi_h(\hat q_k,\hat v_k).
\]

This is the [PDD on-policy mean-velocity objective](https://arxiv.org/html/2607.26004v1#S3), applied to physical phase space. The force component starts with (L) copies of eSEN's direct-force head. Zero-initialized velocity-dependent vector corrections allow later heads to account for motion during the block. The position-velocity heads begin at a constant-acceleration approximation and learn corrections. We train with block sizes 1, 2, 4, and 8. At evaluation, we also train a one-forward **direct coarse-transition** model with the same adapter, and compare with coarse velocity Verlet.

## Data and the deterministic teacher

The public [Timewarp AD-3 files](https://huggingface.co/datasets/microsoft/timewarp/tree/main/AD-3) contain one training and one test trajectory. The downloaded NPZ files have **800,000 train frames** and **400,000 test frames**, including positions, velocities, forces, energies, simulation step, and time. We found a repeating saved-step pattern with gaps **1, 9, 90, 900**; `time / step` indicates **0.001 ps per simulation step**. This differs from the dataset README's simplified “save every 1,000 steps” and 0.5 fs descriptions. The loader reads the time array instead of assuming either description.

The original trajectories are Langevin trajectories; their random increments were not saved. For the first clean PDD test, the AD-3 states supply starting configurations and velocities. The teacher branches deterministically from them using AMBER14 + OBC1 implicit solvent in OpenMM, with no bond constraints, a 2 nm nonperiodic cutoff, and velocity Verlet. These settings follow [Timewarp's `amber14-implicit` implementation](https://github.com/microsoft/timewarp/blob/main/simulation/md.py). We use a 0.5 fs fine step because it gives smaller teacher integration drift than 1 fs on sampled states. **This experiment does not claim to reproduce the original stochastic AD-3 path.**

The program reports force agreement between stored AD-3 forces and the recreated OpenMM system. With OpenMM 8.6 on our development host, the mean absolute difference on eight sampled train frames was about **0.265 eV/Å**, large enough that mixing these labels with the new teacher would be inconsistent. Thus the default force warm-start labels come from the *same OpenMM teacher* used in PDD. The `--label-source ad3` option is kept for an explicit ablation. The cause of the stored-force discrepancy has not yet been established.

## Setup

Use Python 3.11 on Linux with an NVIDIA GPU for the main eSEN run. A CPU works for smoke tests. The eSEN adapter was verified with FAIR-Chem 2.23. Install PyTorch for your CUDA version first if needed, then:

```bash
python -m venv .venv
. .venv/bin/activate
pip install -e '.[esen,test]'
pdd-md download-data --data-root data
pdd-md inspect-data --data-root data --split train
```

The [eSEN checkpoint](https://huggingface.co/facebook/OMol25) is gated by its model license. Accept access on that page and log in with `hf auth login`; no token is stored in this repository. The default eSEN model ID is `esen-sm-direct-all-omol`. A local checkpoint path can also be supplied with `--checkpoint`.

### Google Colab T4 smoke test

In Colab's **Change runtime type** dialog, select **T4 GPU** and runtime version **2025.10**. The latest image on 2026-09-26 used Python 3.13, outside this project's supported range; image 2025.10 provided Python 3.12.12. In a code cell:

```python
!git clone https://github.com/minkyu1022/PDD_MD_acceleration.git
%cd PDD_MD_acceleration
%pip install -e ".[esen]"
```

Restart the Colab session after installation so the notebook kernel loads the installed PyTorch and editable package, then `%cd /content/PDD_MD_acceleration` again. For a short public-data CUDA run:

```python
!pdd-md download-data --data-root data
!pdd-md train-force --backend tiny --data-root data --output runs/smoke/force.pt --max-frames 64 --steps 3 --batch-size 2 --device cuda --platform CPU
!pdd-md train-pdd --data-root data --force-checkpoint runs/smoke/force.pt --output runs/smoke/pdd.pt --max-frames 64 --steps 4 --batch-size 2 --max-block 4 --block-sizes 1 2 4 --device cuda --platform CPU
!pdd-md evaluate --data-root data --pdd-checkpoint runs/smoke/pdd.pt --output runs/smoke/eval.json --max-frames 64 --samples 2 --fine-steps 8 --blocks 1 2 4 --device cuda --platform CPU
```

The [recorded Colab T4 run](results/colab_t4_smoke_2026-09-26.md) confirms GPU execution; its three/four training updates are only a wiring check. The pretrained eSEN checkpoint requires Hugging Face authentication in that Colab session or a local checkpoint file there.

Training writes an atomic checkpoint at `--output` every 500 updates by default and on completion. Pass `--resume runs/esen/pdd.pt` together with a larger `--steps` value to continue an interrupted run with optimizer and sampling RNG restored. Checkpoints and data stay outside Git.

## Small local smoke test

These commands exercise the complete pipeline without a gated checkpoint. The few training steps are only a software check; their metrics are not scientifically meaningful.

```bash
pdd-md train-force --backend tiny --data-root data --output runs/smoke/force.pt --max-frames 64 --steps 3 --batch-size 2
pdd-md evaluate-force --data-root data --force-checkpoint runs/smoke/force.pt --output runs/smoke/force_eval.json --max-frames 64 --samples 8
pdd-md train-pdd --data-root data --force-checkpoint runs/smoke/force.pt --output runs/smoke/pdd.pt --max-frames 64 --steps 4 --batch-size 2 --max-block 4 --block-sizes 1 2 4
pdd-md train-direct --data-root data --force-checkpoint runs/smoke/force.pt --output runs/smoke/direct.pt --max-frames 64 --steps 2 --batch-size 1 --coarse-factor 4
pdd-md evaluate --data-root data --pdd-checkpoint runs/smoke/pdd.pt --direct-checkpoint runs/smoke/direct.pt --output runs/smoke/eval.json --max-frames 64 --samples 2 --fine-steps 8 --blocks 1 2 4
```

## Main experiment

Run on a CUDA machine after eSEN checkpoint access is approved. This is a starting budget; increase the number of iterations after measuring learning curves and held-out results.

`bash scripts/run_experiment.sh` runs download, force adaptation, held-out force check, PDD training, direct baseline training, trajectory evaluation, and batched inference benchmarking in order. Set `RUN_ROOT`, `DEVICE`, `FORCE_STEPS`, `STUDENT_STEPS`, and batch-size environment variables to change the default budget.

```bash
pdd-md train-force --backend esen --data-root data --output runs/esen/force.pt --steps 5000 --batch-size 16 --device cuda --platform CPU
pdd-md evaluate-force --data-root data --force-checkpoint runs/esen/force.pt --output runs/esen/force_eval.json --samples 128 --device cuda --platform CPU
pdd-md train-pdd --data-root data --force-checkpoint runs/esen/force.pt --output runs/esen/pdd.pt --steps 10000 --batch-size 8 --max-block 8 --block-sizes 1 2 4 8 --prefix-blocks 2 --device cuda --platform CPU
pdd-md train-direct --data-root data --force-checkpoint runs/esen/force.pt --output runs/esen/direct_L8.pt --coarse-factor 8 --steps 10000 --batch-size 8 --device cuda --platform CPU
pdd-md evaluate --data-root data --pdd-checkpoint runs/esen/pdd.pt --direct-checkpoint runs/esen/direct_L8.pt --output runs/esen/eval.json --blocks 1 2 4 8 --fine-steps 80 --samples 32 --device cuda --platform CPU
pdd-md benchmark --data-root data --pdd-checkpoint runs/esen/pdd.pt --output runs/esen/throughput_L8.json --block 8 --fine-steps 80 --batch-sizes 1 8 32 --device cuda
```

For the pretrained-weight ablation, use `--backend tiny` with a matched parameter/compute budget. The two-backbone comparison is *not* an equal-architecture random-init test; a fair equal-architecture ablation would initialize the same eSEN architecture randomly and train it through the same procedure.

## Metrics and interpretation

The first measured CPU run is recorded in [results/cpu_pilot_2026-09-26.md](results/cpu_pilot_2026-09-26.md). It demonstrates force adaptation and exposes severe 40 fs rollout instability at L4; it does not establish acceleration.

The [failure-horizon and eSEN compute diagnostic](results/diagnostics_2026-09-26.md) uses the later 1,200-update local PDD checkpoint. `pdd-md diagnose` measures held-out rollout error at specified fine-step horizons and checks that fused heads match the ordinary head calculation. `pdd-md benchmark-mlip` times PDD block decoding against velocity Verlet with forces obtained by differentiating the eSEN energy head. That benchmark isolates compute; the current PDD student and energy-gradient trajectory use **different potentials**, so its timing ratio is not a validated MD speedup. Example:

```bash
pdd-md diagnose --data-root data --pdd-checkpoint runs/esen/pdd.pt --output runs/esen/diagnose.json --block 4 --horizons 4 8 20 40 80 --samples 8 --device cuda
pdd-md benchmark-mlip --data-root data --pdd-checkpoint runs/esen/pdd.pt --output runs/esen/benchmark_mlip.json --block 4 --fine-steps 80 --batch-sizes 1 8 32 --device cuda
```

Evaluation uses held-out initial states from the AD-3 **test trajectory**. `evaluate-force` first measures force RMSE against the OpenMM teacher. Trajectory evaluation compares each method with the deterministic fine-step teacher at its block endpoints and reports position and velocity path/endpoint RMSE, final absolute energy drift measured with the teacher potential, backbone evaluations, inference wall time, and speed ratio to the fine teacher. The `finite_fraction` field counts trajectories with finite coordinates, velocities, and final energy; it does **not** imply physically accurate dynamics. Nonfinite trajectories have null errors; summary RMSE uses only finite trajectories. The eSEN student algebraically fuses its linear heads when only a block endpoint is needed. Evaluation also reports an equal-forward direct transition baseline and coarse Verlet. `benchmark` separately measures batched student throughput to expose scaling with concurrent trajectories. Its timer includes graph construction, backbone, heads, and integration on already-loaded device tensors, but excludes initial host-to-device transfer. Training and test states are never mixed. Timing is hardware and batch-size specific; it should not be read as a portable speedup.

The CPU smoke test establishes that code runs, not that PDD improves accuracy or speed. This PoC compares a relatively expensive eSEN student against a fast classical OpenMM teacher; its wall-time ratio is a software diagnostic, not an estimate of speedup over an eSEN energy-gradient MD teacher. That next experiment needs timing on a target GPU and should include graph building, heads, integration, and batch size. Long-time equilibrium sampling, free energy surfaces, and stochastic Langevin transitions are outside this first deterministic PoC.

## Units

| Quantity | Internal | AD-3 NPZ | eSEN |
|---|---:|---:|---:|
| Position | Å | nm | Å |
| Velocity | Å/ps | nm/ps | added input |
| Force | eV/Å | kJ/(mol nm) | eV/Å |
| Time | ps | `time` array | — |
| Mass | dalton | PDB/OpenMM | — |

Conversions and the force-to-acceleration factor are in `src/pdd_md/units.py`. Dataset files, checkpoint weights, and run outputs are ignored by Git.
