"""Fail closed if the documented external-source decision register drifts."""

import json
from copy import deepcopy
from pathlib import Path

import pytest

REGISTER = Path(__file__).resolve().parents[2] / "docs/governance/source-register.json"
EXTERNAL_IDS = {
    "metr-la",
    "pems-bay",
    "opsd-electricity-2020-10-06",
    "opsd-weather-2020-09-16",
    "entsoe-transparency",
    "era5-reanalysis",
    "gisco-2020-boundaries",
}


def _assert_register(register: dict[str, object]) -> None:
    assert set(register) == {"schema_version", "policy", "reviewed_on", "sources"}
    assert register["schema_version"] == 1
    assert register["policy"] == "default_deny"

    sources = register["sources"]
    assert isinstance(sources, list)
    assert {source["id"] for source in sources} == EXTERNAL_IDS | {"eg006-invented-fixture"}
    assert len(sources) == len({source["id"] for source in sources})
    for source in sources:
        assert set(source) == {
            "id",
            "kind",
            "source_identity",
            "source_document",
            "rights_holder_or_contact",
            "permission_evidence",
            "approved_uses",
            "held_uses",
            "missing_evidence",
            "derived_artifacts",
        }
        assert source["source_identity"] and source["source_document"]
        assert source["rights_holder_or_contact"] and source["derived_artifacts"]
        assert isinstance(source["approved_uses"], list)
        assert isinstance(source["held_uses"], list) and source["held_uses"]
        assert not (set(source["approved_uses"]) & set(source["held_uses"]))
        assert (REGISTER.parents[2] / source["source_document"]).is_file()
        if source["id"] in EXTERNAL_IDS:
            assert source["kind"].startswith("external")
            assert source["approved_uses"] == []
            assert source["missing_evidence"]
        else:
            assert source["kind"] == "synthetic"
            assert source["permission_evidence"]
            assert "real_world_accuracy_claim" in source["held_uses"]


def test_register_is_complete_and_default_deny() -> None:
    _assert_register(json.loads(REGISTER.read_text(encoding="utf-8")))


@pytest.mark.parametrize(
    "defect",
    ["external_approval", "duplicate_source", "missing_rights_gap", "conflicting_use"],
)
def test_register_rejects_unsafe_edits(defect: str) -> None:
    register = deepcopy(json.loads(REGISTER.read_text(encoding="utf-8")))
    sources = register["sources"]
    external = next(source for source in sources if source["id"] == "metr-la")
    if defect == "external_approval":
        external["approved_uses"] = ["new_research_execution"]
        external["held_uses"].remove("new_research_execution")
    elif defect == "duplicate_source":
        sources.append(deepcopy(external))
    elif defect == "missing_rights_gap":
        external["missing_evidence"] = []
    else:
        external["held_uses"].append("private_code_review")
        external["approved_uses"] = ["private_code_review"]
    with pytest.raises(AssertionError):
        _assert_register(register)
