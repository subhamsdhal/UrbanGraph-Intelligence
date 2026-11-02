"""Unit and integration tests for Centralized Multi-Replica Token Store & Safety Enforcement."""

from __future__ import annotations

import os
import time
from pathlib import Path
import pytest

from urbangraph.data import DataError
from urbangraph.security.token_rotation import TokenRotationEngine


def test_sqlite_centralized_token_store(tmp_path: Path) -> None:
    db_file = tmp_path / "shared_tokens.db"

    # Replica 1 starts
    engine1 = TokenRotationEngine(rotation_ttl_seconds=10.0, grace_period_seconds=2.0, store_path=db_file)
    w_token1 = engine1.get_primary_token("worker")
    a_token1 = engine1.get_primary_token("admin")

    # Replica 2 starts pointing to the SAME store
    engine2 = TokenRotationEngine(rotation_ttl_seconds=10.0, grace_period_seconds=2.0, store_path=db_file)
    
    # Replica 2 immediately sees and validates the primary tokens created by Replica 1
    assert engine2.get_primary_token("worker") == w_token1
    assert engine2.validate_token("worker", w_token1) is True
    assert engine2.validate_token("admin", a_token1) is True

    # Replica 1 rotates the worker token
    w_token2 = engine1.rotate_token("worker")
    assert w_token2 != w_token1

    # Replica 2 immediately validates the NEW worker token!
    assert engine2.validate_token("worker", w_token2) is True
    assert engine2.get_primary_token("worker") == w_token2

    # Replica 2 also validates the OLD token during the grace period
    assert engine2.validate_token("worker", w_token1) is True

    # Replica 1 revokes the new worker token
    revoked = engine1.revoke_token("worker", w_token2)
    assert revoked is True

    # Replica 2 immediately sees the token as revoked and rejects it!
    assert engine2.validate_token("worker", w_token2) is False

    # Audit log is shared and synchronized
    audit = engine2.get_audit_log()
    actions = [entry["action"] for entry in audit]
    assert "TOKEN_ROTATED" in actions
    assert "TOKEN_REVOKED" in actions


def test_sqlite_restart_persistence(tmp_path: Path) -> None:
    db_file = tmp_path / "persist_tokens.db"

    engine = TokenRotationEngine(rotation_ttl_seconds=3600.0, grace_period_seconds=900.0, store_path=db_file)
    tok = engine.get_primary_token("admin")

    # Simulate process crash and restart
    restarted_engine = TokenRotationEngine(rotation_ttl_seconds=3600.0, grace_period_seconds=900.0, store_path=db_file)
    assert restarted_engine.get_primary_token("admin") == tok
    assert restarted_engine.validate_token("admin", tok) is True


def test_distributed_mode_safety_restriction(monkeypatch: pytest.MonkeyPatch) -> None:
    # Set multi-replica mode without configuring a centralized store
    monkeypatch.setenv("URBANGRAPH_MULTI_REPLICA", "1")
    monkeypatch.delenv("URBANGRAPH_TOKEN_STORE_PATH", raising=False)

    engine = TokenRotationEngine(store_path=None)
    admin_token = engine.get_primary_token("admin")
    assert engine.validate_token("admin", admin_token) is True

    # Unsafe dynamic rotation must be blocked with an informative error
    with pytest.raises(DataError, match="Dynamic token rotation is securely restricted in distributed multi-replica mode"):
        engine.rotate_token("worker")

