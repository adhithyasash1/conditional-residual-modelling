"""Package init with thread guards for macOS numerical library collisions."""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
for _var in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(_var, "1")

_cache_root = Path(tempfile.gettempdir()) / "conditional_residual_modelling_cache"
_mpl_root = _cache_root / "matplotlib"
_cache_root.mkdir(parents=True, exist_ok=True)
_mpl_root.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("XDG_CACHE_HOME", str(_cache_root))
os.environ.setdefault("MPLCONFIGDIR", str(_mpl_root))

try:
    import torch as _torch

    _torch.set_num_threads(1)
    if hasattr(_torch, "set_num_interop_threads"):
        _torch.set_num_interop_threads(1)
except ImportError:  # pragma: no cover
    pass
