import csv
import json
from unittest.mock import Mock

import yaml

from eval import core_eval


def test_official_metadata_is_cached_without_overwriting_bundle(tmp_path, monkeypatch):
    bundle = tmp_path / "eval_bundle"
    bundle.mkdir()
    legacy = bundle / "eval_meta_data.csv"
    legacy.write_text("original bundle metadata")
    response = Mock(content=b"Eval Task,Random baseline\ntoy,25\n")
    download = Mock(return_value=response)
    monkeypatch.setattr(core_eval.requests, "get", download)

    path = core_eval.ensure_core_metadata(str(bundle))
    assert core_eval.ensure_core_metadata(str(bundle)) == path
    download.assert_called_once_with(core_eval.DCLM_METADATA_URL, timeout=60)
    assert legacy.read_text() == "original bundle metadata"
    assert core_eval.DCLM_REVISION in path


def test_core_uses_external_metadata_and_records_provenance(tmp_path, monkeypatch):
    bundle = tmp_path / "eval_bundle"
    (bundle / "eval_data").mkdir(parents=True)
    (bundle / "core.yaml").write_text(yaml.safe_dump({"icl_tasks": [{
        "label": "toy", "icl_task_type": "multiple_choice",
        "num_fewshot": [0], "dataset_uri": "toy.jsonl",
    }]}))
    (bundle / "eval_data/toy.jsonl").write_text(json.dumps({"query": "toy"}) + "\n")
    metadata = tmp_path / "official.csv"
    with metadata.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Eval Task", "Random baseline"])
        writer.writerow(["toy", 25])
    monkeypatch.setattr(core_eval, "ensure_core_metadata", lambda _: str(metadata))
    monkeypatch.setattr(core_eval, "evaluate_task_batched", lambda *args: 0.625)

    result = core_eval.evaluate_core(None, None, None, str(bundle))
    assert result["Core_v2"] == result["core_metric"] == 0.5
    assert result["eval_version"] == "v2"
    assert result["eval_bundle"]["metadata_sha256"] == core_eval.file_sha256(metadata)
