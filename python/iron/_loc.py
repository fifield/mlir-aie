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
constructor or call site. This is *eager* frame capture: the helper records
``(filename, line, col)`` plus an optional logical name into a tiny
dataclass that does not require an MLIR ``Context``.

Later, when ``resolve()`` runs inside ``mlir_mod_ctx`` and a context is
active, it calls ``loc.materialize()`` (or uses the captured object as a
context manager: ``with self._user_loc:``) which lazily builds a
``FileLineColLoc``, optionally wrapped in a ``NameLoc`` so the printed form
reads ``loc("of0"("user.py":42:4))``. This mirrors how upstream MLIR uses
``NameLoc`` to mark transformation provenance.
"""
from __future__ import annotations

import inspect
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .. import ir  # type: ignore


@dataclass(frozen=True)
class CapturedLoc:
    """A frame snapshot taken before any MLIR Context exists.

    ``materialize()`` is a no-op until called inside an active MLIR
    ``Context`` — at which point it builds a real ``ir.Location``.
    """

    filename: str
    line: int
    col: int
    name: Optional[str] = None

    def materialize(self) -> "ir.Location":
        """Build the MLIR Location. Must be called with an active Context."""
        file_loc = ir.Location.file(self.filename, self.line, col=self.col)
        if self.name is None:
            return file_loc
        return ir.Location.name(self.name, file_loc)

    def __enter__(self):
        # Materialize on entry so callers can write
        # `with self._user_loc:` even though this is not a real Location.
        self._active = self.materialize()
        self._active.__enter__()
        return self._active

    def __exit__(self, *args):
        self._active.__exit__(*args)
        # frozen dataclass: we set _active via object.__setattr__ above? no,
        # __enter__ above set it via normal attribute assignment which
        # works because frozen=True only forbids assignment via __init__-
        # produced attrs. Actually frozen dataclasses *do* forbid all
        # assignment, so use object.__setattr__:
        object.__setattr__(self, "_active", None)


def _is_internal_frame(filename: str) -> bool:
    """Frames inside the IRON / aie python packages or stdlib are skipped."""
    if not filename:
        return True
    pkg_root = Path(__file__).resolve().parent.parent
    p = Path(filename).resolve()
    try:
        p.relative_to(pkg_root)
        return True
    except ValueError:
        pass
    try:
        p.relative_to(Path(sys.prefix).resolve())
        return True
    except ValueError:
        pass
    return False


def capture_user_loc(
    name: Optional[str] = None,
    skip: int = 0,
) -> Optional[CapturedLoc]:
    """Walk the Python call stack and return a ``CapturedLoc`` pointing at
    the first frame outside the IRON / mlir-aie package.

    Returns ``None`` if no non-internal frame can be found (extremely rare;
    only if the entire stack is within framework code). The returned object
    works as a context manager, so callers can write
    ``with capture_user_loc(...) or ir.Location.unknown(): ...`` or use
    the helper :func:`loc_or_unknown`.

    No MLIR ``Context`` is required at capture time — only at materialize
    time (i.e. when used as a context manager inside ``resolve()``).
    """
    frame = inspect.currentframe()
    if frame is None:
        return None
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
    return CapturedLoc(filename=info.filename, line=info.lineno, col=col, name=name)


@contextmanager
def loc_or_unknown(loc):
    """``with`` block that activates ``loc`` (a ``CapturedLoc`` or
    ``ir.Location``) if non-None, else falls back to ``Location.unknown()``.
    Materialization happens here (inside an active MLIR Context)."""
    if loc is None:
        active = ir.Location.unknown()
    elif isinstance(loc, CapturedLoc):
        active = loc.materialize()
    else:
        active = loc
    with active:
        yield active
