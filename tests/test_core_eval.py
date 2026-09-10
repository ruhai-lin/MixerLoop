import csv
import json

import pytest
import yaml

from eval import core_eval


def test_core_v2_overrides_only_the_three_baselines(tmp_path, monkeypatch):
    tasks = ["commonsense_qa", "agi_eval_lsat_ar", "bigbench_language_identification", "toy"]
    original = [20, 20, 9.1, 50]
    corrected = [40.3, 25, 25, 50]
    (tmp_path / "eval_data").mkdir()
    (tmp_path / "core.yaml").write_text(yaml.safe_dump({"icl_tasks": [{
        "label": name, "icl_task_type": "multiple_choice",
        "num_fewshot": [0], "dataset_uri": f"{name}.jsonl",
    } for name in tasks]}))
    for name in tasks:
        (tmp_path / f"eval_data/{name}.jsonl").write_text(json.dumps({"query": name}) + "\n")
    metadata = tmp_path / "eval_meta_data.csv"
    with metadata.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Eval Task", "Random baseline"])
        writer.writerows(zip(tasks, original))
    before = metadata.read_bytes()
    monkeypatch.setattr(core_eval, "evaluate_task_batched", lambda *args: 0.625)

    result = core_eval.evaluate_core(None, None, None, str(tmp_path))
    expected = [(0.625 - b / 100) / (1 - b / 100) for b in corrected]
    assert list(result["centered_results"].values()) == pytest.approx(expected)
    assert result["Core_v2"] == pytest.approx(sum(expected) / len(expected))
    assert result["core_metric"] == result["Core_v2"]
    assert result["eval_version"] == "v2"
    assert metadata.read_bytes() == before
