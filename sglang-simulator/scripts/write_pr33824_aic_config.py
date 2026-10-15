#!/usr/bin/env python3
"""Create the AIC Silicon simulator configuration for the local A100 PCIe."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--database-path", type=Path, required=True)
    parser.add_argument("--backend-version", default="0.5.10")
    args = parser.parse_args()

    config = {
        "platform": {
            "accelerator": {
                "name": "a100_pcie_local",
                "hbm_capacity_gb": 80,
                "hbm_bandwidth_gb": 1935,
                "intra_node_bandwidth_gb": 32,
                "inter_node_bandwidth_gb": 25,
            },
            "disk_read_bandwidth_gb": 8,
            "disk_write_bandwidth_gb": 8,
            "memory_read_bandwidth_gb": 64,
            "memory_write_bandwidth_gb": 64,
            "num_device_per_node": 1,
        },
        "predictor": {
            "name": "aiconfigurator",
            "database_path": str(args.database_path.resolve()),
            "database_system": "a100_pcie_local",
            "database_mode": "SILICON",
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
    args.output.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
