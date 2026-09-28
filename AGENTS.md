# Agent handoff

Read [`docs/SERVER_EXPERIMENT_GUIDE.md`](docs/SERVER_EXPERIMENT_GUIDE.md) before extending experiments. The current primary experiment uses an eSEN energy-gradient velocity-Verlet teacher; `scripts/run_experiment.sh` defaults to a different OpenMM teacher. Verify the teacher metadata in every checkpoint.

The existing CLI is single-GPU. Use four GPUs for independent runs until DDP is implemented and checked. The current validation/evaluation commands read the AD-3 test trajectory, so create a train-derived validation split before selecting new hyperparameters and reserve the test trajectory for final evaluation. Preserve data/model provenance and keep large datasets, tokens, and checkpoints out of Git.
