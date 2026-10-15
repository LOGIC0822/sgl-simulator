#!/usr/bin/env python3
"""Write a simulator config for a collected-SGLang ML replay experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--accelerator-name", default="a100_pcie_local")
    parser.add_argument("--hbm-capacity-gb", type=float, default=80)
    parser.add_argument("--hbm-bandwidth-gb", type=float, default=1935)
    parser.add_argument("--intra-node-bandwidth-gb", type=float, default=32)
    parser.add_argument("--inter-node-bandwidth-gb", type=float, default=25)
    parser.add_argument("--memory-bandwidth-gb", type=float, default=64)
    parser.add_argument("--disk-bandwidth-gb", type=float, default=8)
    parser.add_argument("--backend-version", default="0.5.10")
    args = parser.parse_args(argv)
    args.output = Path(args.output).expanduser().resolve()
    args.bundle = Path(args.bundle).expanduser().resolve()
    if not args.bundle.is_file():
        parser.error(f"ML bundle does not exist: {args.bundle}")
    if args.output.exists():
        parser.error(f"refusing to overwrite config: {args.output}")
    for name in (
        "hbm_capacity_gb",
        "hbm_bandwidth_gb",
        "intra_node_bandwidth_gb",
        "inter_node_bandwidth_gb",
        "memory_bandwidth_gb",
        "disk_bandwidth_gb",
    ):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    config = {
        "platform": {
            "accelerator": {
                "name": args.accelerator_name,
                "hbm_capacity_gb": args.hbm_capacity_gb,
                "hbm_bandwidth_gb": args.hbm_bandwidth_gb,
                "intra_node_bandwidth_gb": args.intra_node_bandwidth_gb,
                "inter_node_bandwidth_gb": args.inter_node_bandwidth_gb,
            },
            "disk_read_bandwidth_gb": args.disk_bandwidth_gb,
            "disk_write_bandwidth_gb": args.disk_bandwidth_gb,
            "memory_read_bandwidth_gb": args.memory_bandwidth_gb,
            "memory_write_bandwidth_gb": args.memory_bandwidth_gb,
            "num_device_per_node": 1,
        },
        "predictor": {
            "name": "ml",
            "database_path": str(args.bundle),
            "latency_scale": 1.0,
        },
        "scheduler": {
            "tp_size": 1,
            "ep_size": 1,
            "dp_size": 1,
            "backend_name": "sglang",
            "backend_version": args.backend_version,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as output:
        json.dump(config, output, indent=2)
        output.write("\n")
    print(args.output)


if __name__ == "__main__":
    main()
