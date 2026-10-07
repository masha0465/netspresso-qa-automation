"""Input gate + accounting helpers for a single authorized real NetsPresso operation (pure Python).

Used by ``scripts/run_e5_1_experiment.py``. Nothing here touches the network or the SDK; the
functions decide *whether* the one authorized call may happen and *how* its outcome is accounted.

Rules encoded:
* the input artifact must match the exact expected SHA-256 (no silent substitution);
* the framework precheck must be READY;
* the ledger must show the expected number of prior real operations (exactly-one guard);
* the observed credit delta is classified, never assumed to equal the client-side constant.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

MATCH, MISMATCH, NOT_OBSERVED = "MATCH", "MISMATCH", "NOT_OBSERVED"
CHAIN_LINKS = ("source_model", "r1_fx_artifact", "r1_sha256", "netspresso_execution", "candidate_artifact", "candidate_sha256",
               "local_validation", "accuracy_result", "quality_gate", "credit_record")


def sha256_of(path: Path | str) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_input_artifact(path: Path | str, *, expected_sha256: str, expected_size_bytes: int | None = None,
                          forbidden_paths: tuple[str, ...] = ()) -> dict[str, Any]:
    """Exact-identity check of the compression input. ``status`` PASS only when every check holds."""
    p = Path(path)
    checks: dict[str, bool | None] = {"exists": p.is_file()}
    details: dict[str, Any] = {"path": p.as_posix(), "expected_sha256": expected_sha256}
    if not checks["exists"]:
        return {"status": "BLOCKED", "checks": checks, "details": details, "message": "input artifact not found"}
    checks["not_forbidden_path"] = not any(p.as_posix().endswith(f) or p.name == f for f in forbidden_paths)
    sha = sha256_of(p)
    details["sha256"], details["size_bytes"] = sha, p.stat().st_size
    checks["sha256_matches"] = sha.lower() == expected_sha256.lower()
    checks["size_matches"] = None if expected_size_bytes is None else details["size_bytes"] == expected_size_bytes
    ok = all(v is not False for v in checks.values())
    msg = "input artifact identity verified" if ok else "input artifact identity check FAILED: " + ", ".join(k for k, v in checks.items() if v is False)
    return {"status": "PASS" if ok else "BLOCKED", "checks": checks, "details": details, "message": msg}


def authorize_single_real_operation(*, input_identity: Mapping[str, Any], precheck_status: str, ledger_operations_before: int,
                                    expected_operations_before: int, confirm_credit_use: bool) -> dict[str, Any]:
    """Return ``{"authorized": bool, "reasons": [...]}`` - every reason is a hard stop (no API call)."""
    reasons = []
    if input_identity.get("status") != "PASS":
        reasons.append(f"input identity {input_identity.get('status')}: {input_identity.get('message')}")
    if precheck_status != "READY":
        reasons.append(f"precheck is {precheck_status}, not READY")
    if ledger_operations_before != expected_operations_before:
        reasons.append(f"ledger shows {ledger_operations_before} prior real operations, expected {expected_operations_before} (exactly-one guard)")
    if not confirm_credit_use:
        reasons.append("--confirm-credit-use not given")
    return {"authorized": not reasons, "reasons": reasons, "max_operations": 1}


def classify_credit_delta(*, before: int | None, after: int | None, expected: int) -> dict[str, Any]:
    """Classify the observed account delta against the client-side pre-check constant (never assumes equality)."""
    if before is None or after is None:
        return {"status": NOT_OBSERVED, "expected": expected, "observed": None, "note": "account balance not observed before and/or after; actual deduction unknown"}
    observed = before - after
    if observed == expected:
        return {"status": MATCH, "expected": expected, "observed": observed, "note": "observed account delta equals the client-side pre-check constant"}
    return {"status": MISMATCH, "expected": expected, "observed": observed,
            "note": "observed account delta differs from the client-side constant; recorded as observed, no corrective call made",
            "defect_hint": "CONFIGURATION_ERROR (credit estimate basis) unless the account also changed for another reason"}


def ledger_transition_ok(*, before: Mapping[str, Any], after: Mapping[str, Any], observed_delta: int | None) -> dict[str, Any]:
    """Check that the ledger moved by exactly one operation and by the observed (not assumed) credit delta."""
    ops_delta = len(after.get("operations", [])) - len(before.get("operations", []))
    used_delta = int(after.get("used_credit", 0)) - int(before.get("used_credit", 0))
    checks = {"exactly_one_operation_added": ops_delta == 1,
              "used_delta_equals_observed": None if observed_delta is None else used_delta == observed_delta,
              "remaining_consistent": int(after.get("remaining_estimate", -1)) == int(before.get("remaining_estimate", 0)) - used_delta}
    return {"ok": all(v is not False for v in checks.values()), "checks": checks, "operations_delta": ops_delta, "used_delta": used_delta}


def build_traceability_chain(**links: Any) -> dict[str, Any]:
    missing = [k for k in CHAIN_LINKS if k not in links or links[k] in (None, "", {})]
    chain = {k: links.get(k) for k in CHAIN_LINKS}
    chain["complete"] = not missing
    chain["missing_links"] = missing
    chain["question_answered"] = "Exactly which input artifact produced this candidate artifact? -> see r1_fx_artifact / r1_sha256 -> netspresso_execution -> candidate_sha256" if not missing else None
    return chain
