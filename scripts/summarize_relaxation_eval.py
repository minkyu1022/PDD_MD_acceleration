"""Summarize paired OC20 relaxation runs by system, including failures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--bootstrap", type=int, default=10_000)
    args = parser.parse_args()

    payload = json.loads(args.input.read_text())
    grouped: dict[str, dict[str, dict]] = {}
    for row in payload["rows"]:
        methods = grouped.setdefault(row["sid"], {})
        if row["method"] in methods:
            raise ValueError(f"Duplicate result: {row['sid']} {row['method']}")
        methods[row["method"]] = row

    optimizer = payload["config"]["optimizer"].upper()
    baseline_name = optimizer
    proposal_names = [f"PDD4+{optimizer}", f"direct4+{optimizer}"]
    names = [baseline_name, *proposal_names]
    complete = [sid for sid, methods in grouped.items() if all(name in methods for name in names)]
    print(f"Complete systems: {len(complete)}; partial systems: {len(grouped) - len(complete)}")
    if not complete:
        return

    baseline = [grouped[sid][baseline_name] for sid in complete]
    baseline_calls = np.asarray([row["mlip_calls"] for row in baseline])
    baseline_wall = np.asarray([row["wall_seconds"] for row in baseline])
    rng = np.random.default_rng(42)
    samples = rng.integers(0, len(complete), size=(args.bootstrap, len(complete)))

    for name in names:
        rows = [grouped[sid][name] for sid in complete]
        calls = np.asarray([row["mlip_calls"] for row in rows])
        wall = np.asarray([row["wall_seconds"] for row in rows])
        result = {
            "method": name,
            "n": len(rows),
            "converged": sum(row["converged"] for row in rows),
            "mlip_calls": int(calls.sum()),
            "wall_seconds": round(float(wall.sum()), 2),
            "median_calls": float(np.median(calls)),
            "median_wall_seconds": round(float(np.median(wall)), 2),
        }
        if name != baseline_name:
            quality_pairs = [(row, ref) for row, ref in zip(rows, baseline, strict=True) if row["converged"] and ref["converged"]]
            energy_delta = np.asarray([row["final_energy_ev"] - ref["final_energy_ev"] for row, ref in quality_pairs])
            ads_error_delta = np.asarray([row["final_dft_adsorbate_mae_A"] - ref["final_dft_adsorbate_mae_A"] for row, ref in quality_pairs])
            boot_saving = 1 - calls[samples].sum(axis=1) / baseline_calls[samples].sum(axis=1)
            boot_wall_saving = 1 - wall[samples].sum(axis=1) / baseline_wall[samples].sum(axis=1)
            result.update(
                fewer_calls=int(np.sum(calls < baseline_calls)),
                call_saving_fraction=round(float(1 - calls.sum() / baseline_calls.sum()), 4),
                call_saving_95ci=[round(float(v), 4) for v in np.quantile(boot_saving, [0.025, 0.975])],
                wall_saving_fraction=round(float(1 - wall.sum() / baseline_wall.sum()), 4),
                wall_saving_95ci=[round(float(v), 4) for v in np.quantile(boot_wall_saving, [0.025, 0.975])],
                quality_pairs=len(quality_pairs),
                energy_plus_0_05_ev=int(np.sum(energy_delta > 0.05)),
                energy_plus_0_10_ev=int(np.sum(energy_delta > 0.10)),
                ads_error_plus_0_20_a=int(np.sum(ads_error_delta > 0.20)),
                worst_energy_delta_ev=round(float(energy_delta.max()), 4) if len(energy_delta) else None,
                worst_ads_error_delta_a=round(float(ads_error_delta.max()), 4) if len(ads_error_delta) else None,
                accepted_blocks=sum(row["accepted_blocks"] for row in rows),
                rejected_proposals=sum(row["proposal_rejected"] for row in rows),
            )
        print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
