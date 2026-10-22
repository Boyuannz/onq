# Source lineage: project-owned QAT orchestration, contracts, or tests.
# Upstream integrations are recorded in OnQ/UPSTREAMS.md.
from __future__ import annotations

import sys

from OnQ.config.loader import load_train_config
from OnQ.config.schema import ConfigError


def main(argv: list[str] | None = None) -> None:
    try:
        cfg = load_train_config(argv)
        if cfg.stage == "calibration":
            from OnQ.calibration.run import run_calibration

            run_calibration(cfg)
        else:
            raise ConfigError(f"unsupported stage {cfg.stage!r}")
    except ConfigError as exc:
        print(f"CONFIG ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()
