"""Hardware preparation must stay offline and retain failed-job receipts."""
import json
from types import SimpleNamespace

import pytest

from experiments.quantum import run_hardware_qaoa as hardware


def test_community_listing_is_offline_and_uses_revision_protocol(monkeypatch, tmp_path):
    def forbidden(*args, **kwargs):
        raise AssertionError("Listing communities must not initialize a cloud backend or QAOA")
    monkeypatch.setattr(hardware, "setup_pyqpanda", forbidden)
    monkeypatch.setattr(hardware, "run_sim", forbidden)
    hardware.main(["--list-communities", "--out-dir", str(tmp_path)])
    record = json.loads(next(tmp_path.glob("HW_*.json")).read_text())
    assert record["partition_max_community_size"] == 16
    assert record["risk_factor"] == record["q_eff"] == .5
    assert not record["local_risk_rebalancing"]
    assert sorted(map(len, record["communities"]), reverse=True) == [14, 10, 8, 8]
    assert sum(record["static_state"]) == sum(record["subproblem_cardinalities"]) == 20
    for community, budget in zip(record["communities"], record["subproblem_cardinalities"]):
        assert sum(record["static_state"][i] for i in community) == budget


@pytest.mark.parametrize("options", [[], ["--community", "0", "--shots", "0"],
    ["--community", "0", "--risk-factor", "nan"],
    ["--community", "0", "--submit", "--local"]])
def test_invalid_cli_never_initializes_backend(monkeypatch, options):
    monkeypatch.setattr(hardware, "setup_pyqpanda", lambda *args, **kwargs: pytest.fail("backend called"))
    with pytest.raises(SystemExit) as exc:
        hardware.main(options)
    assert exc.value.code == 2


def test_submitted_job_receipt_survives_pending_timeout(monkeypatch, tmp_path):
    receipt_path = tmp_path / "receipt.json"
    def save(status):
        hardware._save_record(receipt_path, status)
    def forbidden(**kwargs):
        pytest.fail("An unfinished task must not fetch result data")
    job = SimpleNamespace(job_id=lambda: "job-123", status=lambda: SimpleNamespace(name="RUNNING"),
                          result=forbidden)
    monkeypatch.setattr(hardware, "IS_LOCAL", False)
    monkeypatch.setattr(hardware, "CHIP_BACKEND", None)
    monkeypatch.setattr(hardware, "JOB_STATUS_CALLBACK", save)
    monkeypatch.setattr(hardware, "BACKEND", SimpleNamespace(run_instruction=lambda *args: job))
    with pytest.raises(RuntimeError, match="job-123"):
        hardware._run_counts(SimpleNamespace(to_instruction=lambda: None), 4, 2,
                             poll_interval_s=0, max_polls=1)
    record = json.loads(receipt_path.read_text())
    assert record["job_id"] == "job-123"
    assert record["final_status"] == "RUNNING"
    assert record["status_trace"] == ["RUNNING"]
