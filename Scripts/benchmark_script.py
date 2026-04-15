from __future__ import annotations

import sys
from pathlib import Path

from run_benchmark import main


if __name__ == "__main__":
    default_config = Path(__file__).resolve().parents[1] / "configs" / "benchmark.yaml"
    if len(sys.argv) == 1:
        sys.stderr.write(
            "benchmark_script.py is deprecated. Use `python Scripts/run_benchmark.py --config configs/benchmark.yaml`.\n"
        )
        raise SystemExit(main(["--config", str(default_config)]))
    raise SystemExit(main(sys.argv[1:]))
