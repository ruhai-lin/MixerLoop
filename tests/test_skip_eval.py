from __future__ import annotations

import csv
import json
import pytest

import eval.skip_eval as skip_eval
from eval.skip_eval import (
    FORMAL_CHECKPOINTS,
    _init_offline_wandb,
    _write_self_audited_json,
    bootstrap_spearman,
    select_extreme_positions,
    spearman_rho,
    load_mitr_index,
    validate_formal_scope,
)


def test_spearman_rho_and_bootstrap_are_deterministic():
    rows = [
        {"mITR": 0.1, "delta_loss": 0.2},
        {"mITR": 0.2, "delta_loss": 0.4},
        {"mITR": 0.3, "delta_loss": 0.1},
    ]
    assert spearman_rho([0.1, 0.2, 0.3], [0.2, 0.4, 0.1]) == pytest.approx(-0.5)
    first = bootstrap_spearman(rows, bootstrap_samples=100, seed=20260722)
    second = bootstrap_spearman(rows, bootstrap_samples=100, seed=20260722)
    assert first == second
    assert first["bootstrap_samples"] == 100
    assert first["bootstrap_unit"] == "layer_pass"


def test_bootstrap_excludes_undefined_pass_one_mitr():
    result = bootstrap_spearman(
        [
            {"mITR": "", "delta_loss": 2.0},
            {"mITR": 0.1, "delta_loss": 0.2},
            {"mITR": 0.2, "delta_loss": 0.4},
        ],
        bootstrap_samples=20,
    )
    assert result["usable_rows"] == 2


def test_select_extremes_is_deterministic_and_labeled():
    rows = [
        {"checkpoint": "mixerloop-15m", "scale": "15m", "condition": "mixerloop", "layer": i, "pass": 1, "delta_loss": float(i), "mITR": ""}
        for i in range(8)
    ]
    selected = select_extreme_positions(rows, count=2)
    assert [row["layer"] for row in selected["bottom"]] == [0, 1]
    assert [row["layer"] for row in selected["top"]] == [7, 6]
    assert all(set(row) == {"checkpoint", "scale", "condition", "layer", "pass", "delta_loss", "mITR"} for group in selected.values() for row in group)


def test_offline_wandb_initializes_a_local_run_directory(tmp_path):
    output_root = tmp_path / "output"
    run, metadata = _init_offline_wandb(output_root)
    try:
        assert metadata["mode"] == "offline"
        assert metadata["cloud_sync"] is False
        assert metadata["run_id"]
        assert metadata["run_dir"]
        assert not skip_eval.Path(metadata["run_dir"]).is_relative_to(output_root)
    finally:
        run.finish()


def test_run_skip_eval_requires_an_absent_output_root_before_runtime(tmp_path, monkeypatch):
    output_root = tmp_path / "output"
    output_root.mkdir()
    monkeypatch.setattr(skip_eval, "_init_offline_wandb", lambda _: pytest.fail("W&B must not initialize"))
    with pytest.raises(FileExistsError, match="absent before launch"):
        skip_eval.run_skip_eval(
            spec01_root=tmp_path / "spec01",
            data_dir=tmp_path / "data",
            output_root=output_root,
            checkpoint_names=["mixerloop-15m"],
        )


def test_write_audit_persists_its_own_byte_record(tmp_path):
    output_root = tmp_path / "output"
    output_root.mkdir()
    audit_path = output_root / "write_audit.json"
    record = _write_self_audited_json(
        audit_path,
        [{"path": "prior.json", "bytes": 3, "pre_write_root_bytes": 0, "post_write_root_bytes": 3, "cap_bytes": 1000}],
        output_root=output_root,
        cap_bytes=1000,
    )
    payload = json.loads(audit_path.read_text())
    assert payload["self_write_record"] == record
    assert record["post_write_root_bytes"] == skip_eval.root_size_bytes(output_root)


def test_self_audited_checks_can_verify_prior_artifacts(tmp_path):
    output_root = tmp_path / "output"
    output_root.mkdir()
    prior = output_root / "prior.json"
    prior.write_text("{}\n")
    checks_path = output_root / "run_checks.json"
    record = _write_self_audited_json(
        checks_path,
        [{"path": str(prior), "bytes": prior.stat().st_size}],
        output_root=output_root,
        cap_bytes=1000,
        fields={"required_artifacts_written": True},
    )
    payload = json.loads(checks_path.read_text())
    assert payload["required_artifacts_written"] is True
    assert payload["self_write_record"] == record


def test_validate_formal_scope_enforces_six_checkpoint_208_position_contract(tmp_path, monkeypatch):
    class Dataset:
        sample_ids = list(range(1024))

    inventory = []
    layer_counts = {"15m": 6, "42m": 8, "110m": 12}
    for checkpoint in FORMAL_CHECKPOINTS:
        condition, scale = checkpoint.rsplit("-", 1)
        inventory.append(
            {
                "checkpoint": checkpoint,
                "condition": condition,
                "scale": scale,
                "num_layers": layer_counts[scale],
                "loop_count": 4,
            }
        )
    monkeypatch.setattr(skip_eval, "load_fixed_probe_dataset", lambda _: Dataset())
    monkeypatch.setattr(skip_eval, "parse_released_checkpoint_inventory", lambda _: inventory)

    mitr_paths = []
    source_groups = [
        [checkpoint for checkpoint in FORMAL_CHECKPOINTS if checkpoint != "fullloop-110m"],
        ["fullloop-110m"],
    ]
    for index in range(2):
        path = tmp_path / f"mitr_{index}.csv"
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["condition", "scale", "layer", "pass", "mITR"],
            )
            writer.writeheader()
            for checkpoint in source_groups[index]:
                condition, scale = checkpoint.rsplit("-", 1)
                for layer in range(layer_counts[scale]):
                    for pass_index in range(2, 5):
                        writer.writerow(
                            {
                                "condition": condition,
                                "scale": scale,
                                "layer": layer,
                                "pass": pass_index,
                                "mITR": "0.5",
                            }
                        )
        mitr_paths.append(path)

    plan = validate_formal_scope(
        spec01_root=tmp_path / "spec01",
        outputs_root=tmp_path / "outputs",
        output_root=tmp_path / "new-formal-root",
        mitr_paths=mitr_paths,
    )

    assert plan["checkpoint_count"] == 6
    assert plan["skip_position_count"] == 208
    assert plan["expected_skip_sample_rows"] == 212_992
    assert plan["frozen_mitr_key_count"] == 156
    assert plan["frozen_mitr_source_separation_pass"] is True
    assert plan["runtime_started"] is False


def test_load_mitr_index_rejects_duplicate_source_keys(tmp_path):
    path = tmp_path / "mitr.csv"
    path.write_text("condition,scale,layer,pass,mITR\nmixerloop,15m,0,2,0.5\n")

    with pytest.raises(ValueError, match="duplicate frozen mITR key"):
        load_mitr_index([path, path])


def test_formal_preflight_cli_does_not_start_runtime(monkeypatch, capsys):
    plan = {"checkpoint_count": 6, "skip_position_count": 208, "runtime_started": False}
    monkeypatch.setattr(skip_eval, "validate_formal_scope", lambda **_: plan)
    monkeypatch.setattr(skip_eval, "run_skip_eval", lambda **_: pytest.fail("formal preflight must not run models"))
    monkeypatch.setattr(
        "sys.argv",
        [
            "skip_eval",
            "--spec01-root",
            "spec01",
            "--data-dir",
            "data",
            "--output-root",
            "formal",
            "--formal-preflight",
        ],
    )

    skip_eval.main()

    assert json.loads(capsys.readouterr().out) == plan


def test_merge_core_scores_fills_only_selected_positions():
    detail = [
        {"checkpoint": "mixerloop-15m", "layer": 1, "pass": 2, "core_score_if_run": ""},
        {"checkpoint": "mixerloop-15m", "layer": 1, "pass": 3, "core_score_if_run": ""},
    ]
    results = [
        {
            "core_metric": 0.42,
            "selection": {"checkpoint": "mixerloop-15m", "layer": 1, "pass": 2},
        }
    ]

    merged, applied = skip_eval.merge_core_scores(detail, results)

    assert applied == 1
    assert merged[0]["core_score_if_run"] == pytest.approx(0.42)
    assert merged[1]["core_score_if_run"] == ""
