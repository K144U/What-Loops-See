"""The free GPU memory a job's runs need, in MiB, read from their own configs.

    python -m loopvision.train.mem_floor "id|rev/c2_app_x2.yaml|400" ...

Config paths in a run spec are relative to configs/, because multirun.pbs
launches each run with --config "configs/${cfg}". This resolves them the
same way. On 2026-10-11 a job list written with a configs/ prefix of its own
made every run look for configs/configs/..., fail at launch, and chain into
a new job that failed the same way.

multirun.pbs exports this as LOOPVISION_MIN_FREE_MIB before choosing cards,
so the floor is a property of the runs and not of whoever submitted them. A
successor started by chain_submit at the wall therefore asks for what its
parent did, with nothing forwarded through qsub -v, which is the forwarding
D-058 records outliving every later change.

Measured twice. scripts/probe_step_cost.py on 2026-10-11, job 10884, read
torch's own peak: 7.9 GB at pixel scale 1 and 28.3 GB at pixel scale 2
(D-067). That is not what a card loses. nvidia-smi on the node on 2026-10-12,
eleven hours into the real runs and stored in runs/_launch_oom_20261011/, read 8728 to 8918 MiB per scale 1 run and
30060 to 30164 MiB per scale 2 run, because the CUDA context and the
allocator's reserve sit on top of the tensors. The picker divides a card's
free memory by this floor, so the floor has to cover the nvidia-smi figure.
The first scale 2 floor, 30720, covered it by 556 MiB, under two percent,
and is 32768 since D-071.
"""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

FLOOR_MIB = {1: 10240, 2: 32768}
#: Where multirun.pbs reads run configs from, relative to the repository.
CONFIGS_ROOT = Path("configs")
DEFAULT_MIB = FLOOR_MIB[1]


def floor_for(cfg: dict) -> int:
    scale = int(cfg.get("pixel_scale") or 1)
    if scale not in FLOOR_MIB:
        raise ValueError(f"no measured memory floor for pixel_scale {scale}")
    return FLOOR_MIB[scale]


def main(argv: list[str] | None = None) -> int:
    tokens = sys.argv[1:] if argv is None else argv
    floors = []
    for token in tokens:
        parts = token.split("|")
        if len(parts) < 2:
            raise ValueError(f"run spec {token!r} is not id|config|seed")
        cfg = yaml.safe_load((CONFIGS_ROOT / parts[1]).read_text(encoding="utf-8")) or {}
        floors.append(floor_for(cfg))
    print(max(floors) if floors else DEFAULT_MIB)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
