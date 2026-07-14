"""Test helpers for the Hermes host adapter package.

Provider lifecycle tests stub Hermes agent imports; runtime tests import
MnemeRuntime directly. Engine tests stay at repo root against mneme.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

HOST = Path(__file__).resolve().parent
ROOT = HOST.parents[1]

# Ensure repo root is importable for engine tests colocated via path hacks.
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(HOST) not in sys.path:
    sys.path.insert(0, str(HOST))
