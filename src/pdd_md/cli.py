"""Command-line entry point for a reproducible AD-3 proof of concept."""

from __future__ import annotations

import argparse
import json

from .data import download_ad3, load_ad3
from .evaluate import benchmark_inference, evaluate, evaluate_force
from .teacher import OpenMMTeacher
from .train import train_direct, train_force, train_pdd, validate_teacher


def build_parser():
    parser = argparse.ArgumentParser(prog="pdd-md")
    commands = parser.add_subparsers(dest="command", required=True)
    download = commands.add_parser(
        "download-data", help="Fetch public Timewarp AD-3 train and test files"
    )
    download.add_argument("--data-root", default="data")
    inspect = commands.add_parser(
        "inspect-data", help="Inspect actual time grid and teacher force agreement"
    )
    inspect.add_argument("--data-root", default="data")
    inspect.add_argument("--split", choices=["train", "test"], default="train")
    inspect.add_argument("--platform", default="CPU")
    for name in ("train-force", "train-pdd", "train-direct"):
        command = commands.add_parser(name)
        command.add_argument("--data-root", default="data")
        command.add_argument("--output", required=True)
        command.add_argument(
            "--steps", type=int, default=1000 if name == "train-force" else 2000
        )
        command.add_argument(
            "--batch-size", type=int, default=8 if name == "train-force" else 4
        )
        command.add_argument(
            "--learning-rate",
            type=float,
            default=1e-4 if name == "train-force" else 2e-5,
        )
        command.add_argument("--max-frames", type=int)
        command.add_argument("--device", default="cpu")
        command.add_argument("--platform", default="CPU")
        command.add_argument("--seed", type=int, default=7)
        command.add_argument("--log-every", type=int, default=50)
        command.add_argument("--save-every", type=int, default=500)
        command.add_argument(
            "--resume", help="Continue training from a saved checkpoint"
        )
    force = commands.choices["train-force"]
    force.add_argument("--backend", choices=["esen", "tiny"], default="esen")
    force.add_argument("--checkpoint", default="esen-sm-direct-all-omol")
    force.add_argument("--hidden", type=int, default=32)
    force.add_argument("--label-source", choices=["teacher", "ad3"], default="teacher")
    pdd = commands.choices["train-pdd"]
    pdd.add_argument("--force-checkpoint", required=True)
    pdd.add_argument("--max-block", type=int, default=8)
    pdd.add_argument("--block-sizes", type=int, nargs="+", default=[1, 2, 4, 8])
    pdd.add_argument("--dt-ps", type=float)
    pdd.add_argument("--prefix-blocks", type=int, default=0)
    pdd.add_argument("--velocity-scale", type=float, default=10.0)
    pdd.add_argument("--accel-scale", type=float, default=10000.0)
    direct = commands.choices["train-direct"]
    direct.add_argument("--force-checkpoint", required=True)
    direct.add_argument("--coarse-factor", type=int, default=8)
    direct.add_argument("--dt-ps", type=float, default=0.0005)
    direct.add_argument("--velocity-scale", type=float, default=10.0)
    direct.add_argument("--accel-scale", type=float, default=10000.0)
    force_test = commands.add_parser("evaluate-force")
    force_test.add_argument("--data-root", default="data")
    force_test.add_argument("--force-checkpoint", required=True)
    force_test.add_argument("--output", required=True)
    force_test.add_argument("--samples", type=int, default=64)
    force_test.add_argument("--max-frames", type=int)
    force_test.add_argument("--batch-size", type=int, default=16)
    force_test.add_argument("--device", default="cpu")
    force_test.add_argument("--platform", default="CPU")
    test = commands.add_parser("evaluate")
    test.add_argument("--data-root", default="data")
    test.add_argument("--pdd-checkpoint", required=True)
    test.add_argument("--direct-checkpoint")
    test.add_argument("--output", required=True)
    test.add_argument("--blocks", type=int, nargs="+", default=[1, 2, 4, 8])
    test.add_argument("--fine-steps", type=int, default=80)
    test.add_argument("--samples", type=int, default=8)
    test.add_argument("--max-frames", type=int)
    test.add_argument("--device", default="cpu")
    test.add_argument("--platform", default="CPU")
    bench = commands.add_parser("benchmark")
    bench.add_argument("--data-root", default="data")
    bench.add_argument("--pdd-checkpoint", required=True)
    bench.add_argument("--output", required=True)
    bench.add_argument("--block", type=int, default=8)
    bench.add_argument("--fine-steps", type=int, default=80)
    bench.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 8, 32])
    bench.add_argument("--repeats", type=int, default=3)
    bench.add_argument("--max-frames", type=int)
    bench.add_argument("--device", default="cpu")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    kw = vars(args)
    name = kw.pop("command")
    if name == "download-data":
        for path in download_ad3(kw["data_root"]):
            print(path)
    elif name == "inspect-data":
        trajectory = load_ad3(kw["data_root"], kw["split"])
        teacher = OpenMMTeacher(trajectory.pdb_path, kw["platform"])
        result = {
            "split": kw["split"],
            "frames": len(trajectory),
            "atoms": trajectory.positions.shape[1],
            "first_steps": trajectory.steps[:8].tolist(),
            "dt_ps_per_md_step": trajectory.dt_ps,
            "unique_saved_step_gaps": sorted(
                {int(x) for x in trajectory.steps[1:100] - trajectory.steps[:99]}
            ),
            "teacher_agreement": validate_teacher(trajectory, teacher),
        }
        print(json.dumps(result, indent=2))
    elif name == "train-force":
        train_force(**kw)
    elif name == "train-pdd":
        kw["block_sizes"] = tuple(kw["block_sizes"])
        train_pdd(**kw)
    elif name == "train-direct":
        train_direct(**kw)
    elif name == "evaluate-force":
        evaluate_force(**kw)
    elif name == "evaluate":
        kw["block_sizes"] = tuple(kw.pop("blocks"))
        evaluate(**kw)
    elif name == "benchmark":
        kw["batch_sizes"] = tuple(kw["batch_sizes"])
        benchmark_inference(**kw)


if __name__ == "__main__":
    main()
