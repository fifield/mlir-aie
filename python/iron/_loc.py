# _loc.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2026 Advanced Micro Devices, Inc.
"""Helpers for capturing user Python source locations and projecting them
into MLIR ``Location`` attributes.

IRON entry points (``ObjectFifo``, ``Worker``, ``Runtime.fill`` / ``drain``,
``Kernel`` / ``ExternalFunction``) call ``capture_user_loc`` from their
constructor or call site. The captured ``ir.Location`` is later applied as a
context (``with self._user_loc:``) inside ``resolve()`` so MLIR ops created
from it inherit a ``FileLineColLoc`` pointing at the user's program rather
than ``UnknownLoc``.

A ``NameLoc`` wrapper is used when a logical name is available (e.g. an
ObjectFifo name) so the printed location reads
``loc("of0"("user.py":42:4))`` -- the IRON-level name annotates the
underlying source position. This mirrors how upstream MLIR uses
``NameLoc`` to mark transformation provenance.
"""
from __future__ import annotations

import inspect
import os
import sys
from pathlib import Path
from typing import Optional

from .. import ir  # type: ignore


def _is_internal_frame(filename: str) -> bool:
    """Frames inside the IRON / aie python packages or stdlib are skipped."""
    if not filename:
        return True
    # The IRON sources live next to this file; anything in the same package
    # tree is "internal" and should be walked past.
    pkg_root = Path(__file__).resolve().parent.parent
    p = Path(filename).resolve()
    try:
        p.relative_to(pkg_root)
        return True
    except ValueError:
        pass
    # Anything under sys.prefix (stdlib, site-packages of the venv where
    # mlir, numpy, etc. live) is also internal.
    try:
        p.relative_to(Path(sys.prefix).resolve())
        return True
    except ValueError:
        pass
    return False


def capture_user_loc(
    name: Optional[str] = None,
    skip: int = 0,
) -> Optional["ir.Location"]:
    """Walk the Python call stack and return an ``ir.Location`` pointing at
    the first frame outside the IRON / mlir-aie package.

    Args:
        name: Optional logical name. If provided, the returned location is
            ``NameLoc.get(name, FileLineColLoc(...))`` so callers see the
            IRON-level identifier in MLIR diagnostics.
        skip: Number of additional frames (beyond this function's own frame)
            to skip before starting the search. Useful when the caller is
            itself a thin wrapper.

    Returns:
        An ``ir.Location`` if an MLIR ``Context`` is active and a non-internal
        frame is found, otherwise ``None``. Returning ``None`` is safe -- the
        IRON ``resolve()`` paths fall back to ``Location.unknown()``.
    """
    if ir.Context.current is None:
        return None

    frame = inspect.currentframe()
    if frame is None:
        return None
    # Skip our own frame plus any caller-requested frames.
    for _ in range(1 + skip):
        if frame.f_back is None:
            return None
        frame = frame.f_back
    while frame is not None and _is_internal_frame(frame.f_code.co_filename):
        frame = frame.f_back
    if frame is None:
        return None

    info = inspect.getframeinfo(frame)
    col = 0
    if sys.version_info >= (3, 11) and getattr(info, "positions", None):
        col = info.positions.col_offset or 0
    file_loc = ir.Location.file(info.filename, info.lineno, col=col)
    if name is None:
        return file_loc
    return ir.Location.name(name, file_loc)


def loc_or_unknown(loc: Optional["ir.Location"]) -> "ir.Location":
    """Return ``loc`` if non-None, else ``Location.unknown()``. Cheap helper
    so resolve() bodies can write ``with loc_or_unknown(self._user_loc):``."""
    return loc if loc is not None else ir.Location.unknown()
