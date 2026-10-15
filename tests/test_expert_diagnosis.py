import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from robotwin_phys import expert_diagnosis as diagnosis


def test_same_seed_comparison_preserves_original_physics_and_records_both_cases(tmp_path, monkeypatch):
    requests = []

    def run(argv, **kwargs):
        request = json.loads(Path(argv[-2]).read_text())
        requests.append(request)
        protocol = request["episode"]["condition"]["protocol"]
        result = ({"status": "verified"} if protocol == "official_random" else
                  {"status": "rejected", "reason": "expert_failed",
                   "planning_failure": {"kind": "no_reachable_grasp"}})
        Path(argv[-1]).write_text(json.dumps(result))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(diagnosis.subprocess, "run", run)
    summary = diagnosis.compare("adjust_bottle", 100054, tmp_path)
    original, physical = [r["episode"]["condition"] for r in requests]
    assert original["episode_seed"] == physical["episode_seed"] == 100054
    assert original["geometry_scale"] == original["mass_scale"] == 1
    assert original["table_tilt_deg"] == original["force_mg"] == 0
    assert original["friction"] is None and original["table_height_offset_m"] is None
    assert physical["active_factors"]
    assert [c["status"] for c in summary["cases"]] == ["verified", "rejected"]
    assert json.loads((tmp_path / "summary.json").read_text()) == summary
    with pytest.raises(FileExistsError):
        diagnosis.compare("adjust_bottle", 100054, tmp_path)


def test_runtime_failure_in_first_case_does_not_hide_second_case(tmp_path, monkeypatch):
    def run(argv, **kwargs):
        protocol = json.loads(Path(argv[-2]).read_text())["episode"]["condition"]["protocol"]
        Path(argv[-1]).write_text(json.dumps({"status": "error", "error": "CUDA failure"}
                                           if protocol == "official_random" else {"status": "verified"}))
        return SimpleNamespace(returncode=1 if protocol == "official_random" else 0)

    monkeypatch.setattr(diagnosis.subprocess, "run", run)
    summary = diagnosis.compare("adjust_bottle", 100054, tmp_path)
    assert [c["status"] for c in summary["cases"]] == ["error", "verified"]
    assert summary["cases"][0]["error"] == "CUDA failure"
