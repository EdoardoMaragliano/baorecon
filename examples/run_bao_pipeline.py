"""Run the end-to-end BAO reconstruction pipeline from a YAML config file."""

from __future__ import annotations

import argparse

from baorecon.pipeline import ReconstructionPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run BAO reconstruction pipeline from YAML config.")
    parser.add_argument(
        "config",
        help="Path to YAML config file (see bao_pipeline_example.yaml)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    pipeline = ReconstructionPipeline(args.config)
    data_path, random_path = pipeline.run()
    print("Saved data catalog: {0}".format(data_path))
    print("Saved random catalog: {0}".format(random_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
