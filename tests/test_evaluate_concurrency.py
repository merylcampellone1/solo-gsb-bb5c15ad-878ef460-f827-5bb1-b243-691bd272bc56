"""Concurrency tests for rule evaluation vs. input writes.

Regression for a stale write-back: while a manual evaluation had read its
inputs but not yet saved its result, a telemetry import committed a newer
sample and re-evaluated; the stale evaluation committed last and resurrected
a fault interval that should already have been shortened.

The contract enforced here:

* ``services.evaluate_all_rules`` reads its inputs (events, rules, channel
  TTLs and maintenance windows) *and* replaces the fault/suppressed rows
  inside one serialized write transaction, so an evaluation and a concurrent
  import (or rule / TTL / maintenance write) are linearized in the same
  order — the last committed result always reflects the last committed
  inputs;
* if the evaluation fails, the whole replacement rolls back: no rule keeps
  its old faults deleted while the new faults are only half written;
* readers (pages/API built on the committed-view read layer) only ever see
  the previously committed faults while an evaluation is still open.
"""

import json
import os
import tempfile
import threading

import pytest

from app import db, services


def sample(ts: float, value: float) -> str:
    return json.dumps({"device": "d", "metric": "m", "ts": ts, "value": value})


@pytest.fixture()
def scratch_db():
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    db.DB_PATH = tmp.name
    db._conn = None
    db.init_db()
    yield tmp.name
    for suffix in ("", "-wal", "-shm"):
        try:
            os.unlink(tmp.name + suffix)
        except OSError:
            pass


def _create_rule() -> int:
    rule_id = db.execute(
        "INSERT INTO rules (name, logic, duration, created_at) VALUES (?,?,?,?)",
        ("high", "AND", 0.0, 0.0),
    ).lastrowid
    db.execute(
        "INSERT INTO rule_conditions (rule_id, device, metric, op, threshold, ord)"
        " VALUES (?,?,?,?,?,?)",
        (rule_id, "d", "m", ">=", 80, 0),
    )
    return rule_id


def _fault_endpoints() -> list[tuple[float, float]]:
    return [
        (r["start_ts"], r["end_ts"])
        for r in db.query("SELECT start_ts, end_ts FROM fault_intervals ORDER BY id")
    ]


def test_stale_evaluation_cannot_overwrite_newer_import(scratch_db, monkeypatch):
    # Initial committed inputs: high sample at 0s, low sample at 20s =>
    # the fault holds over [0, 20).
    services.import_events(
        "\n".join((sample(0, 90), sample(20, 10))), reevaluate=False
    )
    rule_id = _create_rule()
    services.evaluate_all_rules()
    assert _fault_endpoints() == [(0, 20)]

    # Pause an evaluation right after *all* its inputs were loaded.  The last
    # input read is the channel-TTL load on both the fixed code (inside the
    # serialized write transaction) and the old code (a separate read
    # snapshot taken before any write lock was held), making it an
    # implementation-neutral interception point.
    original_load_ttls = services._load_channel_ttls
    inputs_read = threading.Event()
    release = threading.Event()
    fired = {"v": False}

    def paused_load_ttls(*args):
        result = original_load_ttls(*args)
        if not fired["v"]:
            fired["v"] = True
            inputs_read.set()
            release.wait(10)
        return result

    monkeypatch.setattr(services, "_load_channel_ttls", paused_load_ttls)

    # The "manual evaluation": reads {0: high, 20: low}, then stalls.
    manual = threading.Thread(target=services.evaluate_all_rules)
    manual.start()
    assert inputs_read.wait(5), "manual evaluation never reached its read point"

    # Import the 10s low sample (which shortens the fault to [0, 10)) and
    # re-evaluate.  With serialization this must not finish until the manual
    # evaluation has committed, because the manual evaluation still holds the
    # writer lock while paused.
    imported = threading.Event()

    def do_import():
        services.import_events(sample(10, 10))
        imported.set()

    importer = threading.Thread(target=do_import)
    importer.start()
    assert not imported.wait(0.5), (
        "import committed while the in-flight evaluation still held its inputs"
    )

    # While the evaluation is still open, readers see only the last committed
    # result, never a half-replaced table.
    assert _fault_endpoints() == [(0, 20)]

    release.set()
    manual.join(5)
    importer.join(5)
    assert not manual.is_alive() and not importer.is_alive()

    # The last committed result must describe the newest committed inputs:
    # the low sample at 10s cuts the fault to [0, 10) — a stale [0, 20)
    # write-back must never win.
    assert _fault_endpoints() == [(0, 10)]
    assert imported.is_set()


def test_evaluation_rerun_after_blocked_import_is_consistent(scratch_db):
    # End-state variant without injected timing: after the serialized
    # sequence settles, a fresh evaluation must be a no-op rewrite, i.e. the
    # stored faults already match the newest inputs.
    services.import_events(
        "\n".join((sample(0, 90), sample(20, 10))), reevaluate=False
    )
    _create_rule()
    services.evaluate_all_rules()
    services.import_events(sample(10, 10))
    assert _fault_endpoints() == [(0, 10)]
    again = services.evaluate_all_rules()
    assert again["fault_intervals"] == 1
    assert _fault_endpoints() == [(0, 10)]


def test_failed_evaluation_leaves_no_partial_replacement(scratch_db, monkeypatch):
    # Two rules, both with a stored fault.
    services.import_events(
        "\n".join(sample(m, 90) for m in range(0, 30, 10)), reevaluate=False
    )
    rule_a = _create_rule()
    rule_b = _create_rule()
    services.evaluate_all_rules()
    before = [
        dict(r)
        for r in db.query(
            "SELECT rule_id, start_ts, end_ts FROM fault_intervals ORDER BY id"
        )
    ]
    assert {row["rule_id"] for row in before} == {rule_a, rule_b}

    # Make the second rule's evaluation blow up *after* the first rule has
    # already been replaced inside the write transaction.
    original_eval = services.evaluate_rule
    seen: list[int] = []

    def failing_eval(rule_id, *args, **kwargs):
        seen.append(rule_id)
        if rule_id == rule_b:
            raise RuntimeError("boom during evaluation")
        return original_eval(rule_id, *args, **kwargs)

    monkeypatch.setattr(services, "evaluate_rule", failing_eval)
    with pytest.raises(RuntimeError):
        services.evaluate_all_rules()

    # The transaction rolled back completely: every rule still has exactly
    # its previously committed fault rows (same ids/intervals).
    after = [
        dict(r)
        for r in db.query(
            "SELECT rule_id, start_ts, end_ts FROM fault_intervals ORDER BY id"
        )
    ]
    assert after == before


def test_ttl_write_then_evaluation_reflects_newest_input(scratch_db):
    # A TTL change and an evaluation share the same serialization order:
    # faults committed after the TTL write must reflect the capped hold.
    services.import_events(
        "\n".join((sample(0, 90), sample(20, 10))), reevaluate=False
    )
    _create_rule()
    services.evaluate_all_rules()
    assert _fault_endpoints() == [(0, 20)]
    # A 10s validity period expires the high sample before the 20s low one;
    # the truth gap splits/shortens the fault.
    services.set_channel_ttl("d", "m", 10)
    endpoints = _fault_endpoints()
    assert endpoints == [(0, 10)]
