"""Lifecycle tests for progressive episode finalization (ADR-0006)."""

from __future__ import annotations

import importlib.util
import sqlite3
import sys
from pathlib import Path

import pytest

HOST = Path(__file__).resolve().parent
ROOT = HOST.parents[1]
ENGINE = ROOT / "mneme.py"


def load_provider_class():
    # Load engine under mneme_core-like path the provider expects as sibling.
    # Tests construct a minimal stub host by importing provider with a fake
    # agent.memory_provider base.
    import types

    if "agent.memory_provider" not in sys.modules:
        agent = types.ModuleType("agent")
        agent.__path__ = []  # mark as package for submodule imports
        mp = types.ModuleType("agent.memory_provider")

        class MemoryProvider:  # minimal ABC stand-in
            pass

        mp.MemoryProvider = MemoryProvider
        rcwd = types.ModuleType("agent.runtime_cwd")
        rcwd.resolve_agent_cwd = lambda: str(Path.cwd())
        tools = types.ModuleType("tools")
        reg = types.ModuleType("tools.registry")
        reg.tool_error = lambda msg: f"ERROR: {msg}"
        sys.modules["agent"] = agent
        sys.modules["agent.memory_provider"] = mp
        sys.modules["agent.runtime_cwd"] = rcwd
        sys.modules["tools"] = tools
        sys.modules["tools.registry"] = reg

    # Stage sibling mneme.py next to a temp provider package layout via path hacks.
    spec = importlib.util.spec_from_file_location(
        "mneme_host_provider", HOST / "provider.py",
        submodule_search_locations=[str(HOST)],
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    # Provide runtime as package-relative import target
    rt_spec = importlib.util.spec_from_file_location(
        "mneme_host_provider.runtime", HOST / "runtime.py")
    rt = importlib.util.module_from_spec(rt_spec)
    sys.modules["mneme_host_provider.runtime"] = rt
    rt_spec.loader.exec_module(rt)
    # Point provider's sibling mneme.py loader at canonical engine by
    # temporarily writing nothing — patch _get_mneme_class after load.
    # Fix relative import: rewrite package context
    mod.__package__ = "mneme_host_provider"
    spec.loader.exec_module(mod)

    eng_spec = importlib.util.spec_from_file_location("mneme_engine_for_host", ENGINE)
    eng = importlib.util.module_from_spec(eng_spec)
    sys.modules["mneme_engine_for_host"] = eng
    eng_spec.loader.exec_module(eng)
    mod._get_mneme_class = lambda: eng.Mneme
    return mod.MnemeProvider


@pytest.fixture
def provider(tmp_path, monkeypatch):
    cls = load_provider_class()
    home = tmp_path / "hermes"
    (home / "mneme" / "notes").mkdir(parents=True)
    (home / "mneme" / "models").mkdir(parents=True)
    p = cls()
    p.initialize("sess-1", hermes_home=str(home), platform="test")
    assert p._runtime is not None
    # Force a stable cwd outside any git repo for bank=global
    p._runtime._cwd_resolver = lambda: tmp_path
    yield p
    p.shutdown()


def test_progressive_sync_turn_upgrades_checkpoint(provider):
    provider.prefetch("fix cobalt deploy auth")
    provider.sync_turn(
        "fix cobalt deploy auth",
        "Patched the token refresh and verified green.",
    )
    key = f"run:{provider._run_id}"
    with sqlite3.connect(provider._mneme.db_path) as conn:
        body, tags = conn.execute(
            "SELECT body, tags FROM mem WHERE dedupe_key=?", (key,)
        ).fetchone()
    assert tags == "session_progress"
    assert "User development:" in body
    assert "Final response/result:" in body
    assert "Checkpointed at first serve" not in body


def test_segment_rotation_finalizes_previous_run(provider):
    provider.prefetch("investigate routing")
    provider.sync_turn("investigate routing", "Found stale route table.")
    old = provider._run_id
    provider._rotate_run_segment("test")
    assert provider._run_id != old
    with sqlite3.connect(provider._mneme.db_path) as conn:
        body, tags = conn.execute(
            "SELECT body, tags FROM mem WHERE dedupe_key=?",
            (f"run:{old}",),
        ).fetchone()
    assert tags == "session_segment"
    assert "User development:" in body
