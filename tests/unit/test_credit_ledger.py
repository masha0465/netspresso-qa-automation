import json
import re

import pytest

from framework.credit_ledger import CreditLedger, CreditLedgerError
from framework.pipeline.result import CreditUsageType

pytestmark = pytest.mark.unit


def test_new_ledger_starts_at_zero(tmp_path):
    ledger = CreditLedger(tmp_path / "ledger.json", starting_credit=500)
    assert ledger.summary() == {
        "starting_credit": 500, "used_credit": 0, "remaining_estimate": 500,
        "simulated_total": 0, "operations": 0, "has_actual_usage": False,
    }


def test_simulated_entries_never_touch_real_balance(tmp_path):
    ledger = CreditLedger(tmp_path / "ledger.json")
    ledger.record(operation="int8_quantization", model="m", purpose="mock", usage_type="simulated",
                  estimated_credit=50, adapter="mock")
    assert ledger.used_credit == 0 and ledger.remaining_estimate == 500 and ledger.simulated_total == 50
    assert ledger.has_actual_usage() is False
    with pytest.raises(CreditLedgerError, match="actual_credit"):
        ledger.record(operation="x", model="m", purpose="p", usage_type="simulated", estimated_credit=1, actual_credit=1)


def test_real_entries_require_confirmation_and_evidence(tmp_path):
    ledger = CreditLedger(tmp_path / "ledger.json")
    with pytest.raises(CreditLedgerError, match="confirmation"):
        ledger.record(operation="x", model="m", purpose="p", usage_type=CreditUsageType.ACTUAL, estimated_credit=25, actual_credit=25)
    with pytest.raises(CreditLedgerError, match="account API"):
        ledger.record(operation="x", model="m", purpose="p", usage_type="actual", estimated_credit=25, confirmation=True)
    with pytest.raises(CreditLedgerError, match="'none'"):
        ledger.record(operation="x", model="m", purpose="p", usage_type="none", estimated_credit=0)

    entry = ledger.record(operation="profile", model="m", purpose="p", usage_type="estimated", estimated_credit=25,
                          confirmation=True, adapter="netspresso")
    assert entry.usage_type == "estimated" and ledger.remaining_estimate == 475
    ledger.record(operation="compress", model="m", purpose="p", usage_type="actual", estimated_credit=25, actual_credit=30,
                  confirmation=True, adapter="netspresso")
    assert ledger.used_credit == 55 and ledger.remaining_estimate == 445 and ledger.has_actual_usage()


def test_save_and_reload_round_trip(tmp_path):
    path = tmp_path / "ledger.json"
    ledger = CreditLedger(path, starting_credit=500)
    ledger.record(operation="x", model="m", purpose="p", usage_type="simulated", estimated_credit=25, timestamp="t0")
    ledger.save()
    reloaded = CreditLedger(path)
    assert reloaded.summary() == ledger.summary()
    assert reloaded.operations[0]["timestamp"] == "t0"
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["last_updated"] is not None


def test_repository_ledger_integrity(root):
    """Guard: the real project ledger must be internally consistent and contain only confirmed real entries.

    Mock runs never write here (they use separate simulated ledgers), so every entry must be a
    confirmed real operation whose charge is explained by an observed (actual) or estimated figure.
    """
    ledger = CreditLedger(root / "reports" / "credit_usage.json")
    assert ledger.starting_credit == 500 and ledger.simulated_total == 0
    charged = 0
    for op in ledger.operations:
        assert op["confirmation"] is True and op["adapter"] == "netspresso"
        assert op["usage_type"] in ("actual", "estimated")
        if op["usage_type"] == "actual":
            assert op["actual_credit"] is not None
            charged += op["actual_credit"]
        else:
            assert op["actual_credit"] is None
            charged += op["estimated_credit"] or 0
        assert not re.search(r"[A-Za-z]:[\/]Users|/Users/|/home/", json.dumps(op))  # no local paths in the public ledger
    assert ledger.used_credit == charged
    assert ledger.remaining_estimate == ledger.starting_credit - ledger.used_credit
    assert ledger.remaining_estimate >= 100  # project reserve


def test_malformed_ledger_is_rejected(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text('{"starting_credit": 500}', encoding="utf-8")
    with pytest.raises(CreditLedgerError, match="missing"):
        CreditLedger(path)
