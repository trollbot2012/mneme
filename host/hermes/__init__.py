"""Hermes host adapter for Mneme (source of truth; synced into the active profile).

The engine remains host-agnostic in repo-root mneme.py. This package owns:
  - MnemeRuntime  -- session bank isolation, write scope, trust evidence
  - MnemeProvider -- Hermes MemoryProvider plugin surface (provider.py)

Deploy with:  python scripts/sync_hermes_bundle.py --profile <HERMES_HOME>
Verify with:  python scripts/verify_hermes_bundle.py --profile <HERMES_HOME>
"""

from .runtime import MnemeRuntime

__all__ = ["MnemeRuntime"]
