"""Parallel data generation must be byte identical to serial.

Training is generation bound, so parallelising it is the single biggest
lever on wall clock. It is also the easiest place to silently break the
exact-resume guarantee: if the parallel path produced even slightly
different batches, a run killed at the PBS wall would resume onto a
different data stream and no test elsewhere would notice.

Every sample is a pure function of its global index, so splitting a batch
across processes and reassembling in index order must reproduce the serial
result exactly. That is what these tests hold to.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from loopvision.train import cli


def base_cfg(**overrides) -> dict:
    cfg = dict(cli.DEFAULTS)
    cfg.update({"family": "A", "batch_size": 32, "depths": [1], "num_workers": 0})
    cfg.update(overrides)
    return cfg


@pytest.mark.parametrize("family,depths", [("A", [1]), ("A", [3]), ("B", None), ("C", None)])
def test_parallel_batches_match_serial_exactly(family: str, depths) -> None:
    device = torch.device("cpu")
    serial = base_cfg(family=family, depths=depths, num_workers=0)
    parallel = base_cfg(family=family, depths=depths, num_workers=4)

    for cursor in (0, 32, 4096):
        a = cli.make_batch(serial, "train", cursor, device)
        b = cli.make_batch(parallel, "train", cursor, device)
        assert torch.equal(a[0], b[0]), f"{family}: images differ at cursor {cursor}"
        assert torch.equal(a[1], b[1]), f"{family}: queries differ at cursor {cursor}"
        assert torch.equal(a[2], b[2]), f"{family}: labels differ at cursor {cursor}"
        assert np.array_equal(a[3], b[3])
        assert np.array_equal(a[4], b[4])


def test_worker_count_does_not_change_the_data() -> None:
    """Any worker count must give the same batch, including uneven splits.

    A batch of 32 across 5 workers chunks unevenly, which is exactly where
    an off-by-one in reassembly would show up.
    """
    device = torch.device("cpu")
    reference = cli.make_batch(base_cfg(num_workers=0), "train", 128, device)
    for workers in (1, 2, 3, 5, 8):
        got = cli.make_batch(base_cfg(num_workers=workers), "train", 128, device)
        assert torch.equal(reference[0], got[0]), f"{workers} workers changed the images"
        assert torch.equal(reference[2], got[2]), f"{workers} workers changed the labels"


def test_batches_advance_with_the_cursor() -> None:
    """Guards against a pool returning a cached or stale batch."""
    device = torch.device("cpu")
    cfg = base_cfg(num_workers=4)
    first = cli.make_batch(cfg, "train", 0, device)
    second = cli.make_batch(cfg, "train", 32, device)
    assert not torch.equal(first[0], second[0])


def test_splits_stay_separate_under_parallel_generation() -> None:
    device = torch.device("cpu")
    cfg = base_cfg(num_workers=4)
    train = cli.make_batch(cfg, "train", 0, device)
    val = cli.make_batch(cfg, "iid_val", 0, device)
    assert not torch.equal(train[0], val[0])
