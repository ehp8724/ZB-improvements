#!/usr/bin/env python
"""Run one enhancement stage:  python scripts/run_enhancements.py <stage> [options]."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import torch  # noqa: E402

from enhancements import experiments as ex  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("stage")
p.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
p.add_argument("--steps", type=int, default=None)
p.add_argument("--threads", type=int, default=2)
a = p.parse_args()
torch.set_num_threads(a.threads)
ctx = ex.make_ctx()
fn = getattr(ex, f"stage_{a.stage}")
kw = {"seeds": tuple(a.seeds)} if "seeds" in fn.__code__.co_varnames else {}
if a.steps is not None and "steps" in fn.__code__.co_varnames:
    kw["steps"] = a.steps
fn(ctx, **kw)
