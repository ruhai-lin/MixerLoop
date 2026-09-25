from __future__ import annotations

import dataclasses
import importlib
import json
from pathlib import Path

import pytest
import torch

from eval.theory_capture import model_state_digest


@pytest.fixture()
def theory_probe():
    return importlib.import_module("eval.theory_probe")


def _write_spec01_artifacts(root: Path) -> tuple[list[int], list[int], list[int]]:
    records = [
        {
            "sample_id": sample_id,
            "input_sha256": f"sha-{sample_id}",
            "num_tokens": 512,
            "num_non_padding_tokens": 512,
        }
        for sample_id in range(10)
    ]
    theory = {
        "seed": 20260722,
        "sample_count": 10,
        "seq_len": 512,
        "metric_positions": {"start": 32, "end": 510},
        "records": records,
    }
    split = {
        "seed": 20260722,
        "sample_count": 10,
        "fit_count": 8,
        "heldout_count": 2,
        "fit_sample_ids": list(range(8)),
        "heldout_sample_ids": [8, 9],
    }
    root.mkdir(parents=True, exist_ok=True)
    (root / "theory_eval_ids.json").write_text(json.dumps(theory, indent=2) + "\n")
    (root / "probe_split_metadata.json").write_text(json.dumps(split, indent=2) + "\n")
    return list(range(10)), list(range(8)), [8, 9]


class _Identity(torch.nn.Module):
    def forward(self, x):
        return x


class _ToyBase(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Module()
        self.model.norm = _Identity()
        self.lm_head = torch.nn.Linear(3, 5, bias=False)


def test_loads_spec01_split_and_rejects_alternate_authority(tmp_path: Path, theory_probe):
    sample_ids, fit_ids, heldout_ids = _write_spec01_artifacts(tmp_path)

    dataset = theory_probe.load_fixed_probe_dataset(tmp_path)

    assert dataset.sample_ids == sample_ids
    assert dataset.fit_sample_ids == fit_ids
    assert dataset.heldout_sample_ids == heldout_ids
    assert dataset.metric_start == 32
    assert dataset.metric_end == 510
    assert dataset.seed == 20260722
    assert dataclasses.is_dataclass(dataset)


def test_run_config_has_no_full_raw_archive_input(theory_probe):
    fields = {field.name for field in dataclasses.fields(theory_probe.Spec02RunConfig)}

    assert "spec01_root" in fields
    assert "output_root" in fields
    assert not any("raw_hidden" in field or "raw_logit" in field or "archive" in field for field in fields)


def test_target_enumeration_preserves_no_loop_and_record_type_separation(theory_probe):
    gdn = theory_probe.enumerate_probe_targets(condition="gdn", scale="15m", num_layers=2, loop_count=4)
    mixer = theory_probe.enumerate_probe_targets(condition="mixerloop", scale="15m", num_layers=1, loop_count=4)
    full = theory_probe.enumerate_probe_targets(condition="fullloop", scale="15m", num_layers=1, loop_count=4)

    assert [(t.layer, t.pass_index, t.record_type) for t in gdn] == [(0, 1, "hA"), (1, 1, "hA")]
    assert [(t.layer, t.pass_index, t.record_type) for t in mixer] == [
        (0, 1, "hA"),
        (0, 2, "hA"),
        (0, 3, "hA"),
        (0, 4, "hA"),
    ]
    assert [(t.layer, t.pass_index, t.record_type) for t in full] == [
        (0, 1, "hA"),
        (0, 1, "hF"),
        (0, 2, "hA"),
        (0, 2, "hF"),
        (0, 3, "hA"),
        (0, 3, "hF"),
        (0, 4, "hA"),
        (0, 4, "hF"),
    ]


def test_probe_hyperparameters_and_tail_batches_match_design(theory_probe):
    config = theory_probe.ProbeTrainingConfig()
    batches = theory_probe.build_metric_token_batches(
        fit_sample_ids=[10, 11, 12],
        positions=range(32, 37),
        token_batch_size=7,
    )

    assert config.optimizer == "AdamW"
    assert config.learning_rate == pytest.approx(1e-3)
    assert config.weight_decay == pytest.approx(1e-4)
    assert config.token_batch_size == 2048
    assert config.epochs == 5
    assert config.keep_tail_batch is True
    assert config.selection_rule == "best_heldout_loss"
    assert [batch["token_count"] for batch in batches] == [7, 7, 1]
    assert batches[-1]["is_tail"] is True


def test_replay_and_probe_step_do_not_mutate_base_model(theory_probe):
    model = _ToyBase()
    before = model_state_digest(model)
    hidden = torch.randn(2, 4, 3)
    labels = torch.tensor([[1, 2, 3, 4], [0, 1, 2, 3]])

    proof = theory_probe.train_linear_probe_step(
        hidden,
        labels,
        num_classes=5,
        config=theory_probe.ProbeTrainingConfig(),
        base_model=model,
    )

    assert proof["base_state_before"] == before
    assert proof["base_state_after"] == before
    assert proof["base_state_unchanged"] is True
    assert proof["probe_parameter_delta_l1"] > 0


def test_frozen_lm_head_readout_fits_no_parameters(theory_probe):
    model = _ToyBase()
    before = model_state_digest(model)
    hidden = torch.randn(1, 512, 3)
    input_ids = torch.randint(0, 5, (1, 512))
    target = theory_probe.ProbeTarget(
        checkpoint="mixerloop-15m",
        scale="15m",
        condition="mixerloop",
        layer=0,
        pass_index=1,
        record_type="hA",
    )

    row = theory_probe.frozen_lm_head_stats_row(
        model,
        hidden,
        input_ids,
        target=target,
        sample_id=7,
        precision="float32",
        input_sha256="abc",
    )

    assert row["readout"] == "frozen_lm_head"
    assert row["record_type"] == "hA"
    assert row["sample_id"] == 7
    assert row["num_positions"] == 479
    assert model_state_digest(model) == before


def test_bootstrap_ci_uses_sample_rows_and_required_schema(theory_probe):
    rows = [
        {"sample_id": 1, "ce_sum": 10.0, "num_positions": 5, "top1_correct": 2, "top5_correct": 4},
        {"sample_id": 2, "ce_sum": 20.0, "num_positions": 5, "top1_correct": 3, "top5_correct": 5},
        {"sample_id": 3, "ce_sum": 30.0, "num_positions": 5, "top1_correct": 4, "top5_correct": 5},
    ]

    result = theory_probe.bootstrap_readout_ci(rows, n_bootstrap=1000, seed=20260722)

    assert result["bootstrap_unit"] == "sample_id"
    assert result["bootstrap_samples"] == 1000
    assert result["num_samples"] == 3
    assert set(theory_probe.PROBE_DETAIL_COLUMNS) == {
        "checkpoint",
        "scale",
        "condition",
        "layer",
        "pass",
        "record_type",
        "readout",
        "heldout_loss",
        "top1",
        "top5",
        "ci_low",
        "ci_high",
    }
    assert result["ci_low"] <= result["heldout_loss"] <= result["ci_high"]


def test_guarded_spec02_write_stops_before_over_budget(tmp_path: Path, theory_probe):
    output_root = tmp_path / "out"
    output_root.mkdir()
    (output_root / "existing.bin").write_bytes(b"1234567890")
    target = output_root / "records" / "manifest.json"

    with pytest.raises(theory_probe.OutputBudgetError):
        theory_probe.write_spec02_json_guarded(target, {"payload": "x" * 100}, output_root=output_root, cap_bytes=32)

    assert not target.exists()


def test_wandb_mode_gate_for_non_formal_runs(theory_probe):
    no_wandb = theory_probe.build_wandb_mode_plan(formal=False, wandb_enabled=False)
    offline = theory_probe.build_wandb_mode_plan(formal=False, wandb_enabled=True)

    assert no_wandb["wandb_enabled"] is False
    assert no_wandb["cloud_sync"] is False
    assert offline["wandb_enabled"] is True
    assert offline["mode"] == "offline"
    assert offline["env"]["WANDB_MODE"] == "offline"
    assert offline["cloud_sync"] is False


def test_acceleration_proof_records_have_required_fields(theory_probe):
    matrix = theory_probe.build_mixerloop_acceleration_matrix(device_type="cpu", wandb_enabled=False)
    required = {
        "row",
        "semantic_owner",
        "reference_source_path",
        "chosen_path",
        "required_runtime_evidence",
        "observed_runtime_evidence",
        "result",
        "proof_limit",
    }
    required_semantic_owners = {
        "BF16/autocast",
        "sequence packing",
        "segment-causal masks",
        "fixed packed row counts",
        "fastest attention/FlexAttention applicability",
        "torch.compile",
        "fused AdamW",
        "pin-memory CPU batches",
        "non-blocking CPU-to-GPU transfer",
        "microbatch/gradient-accumulation accounting",
        "phase timing",
        "GPU utilization/memory trace",
        "same-sample/same-weight equivalence",
        "durable manifest/check artifacts",
        "W&B online-first/offline fallback",
    }

    assert len(matrix) >= len(required_semantic_owners)
    assert all(required.issubset(row) for row in matrix)
    assert {row["semantic_owner"] for row in matrix} >= required_semantic_owners
    assert all(row["observed_runtime_evidence"] == "placeholder: no Spec 02 runtime yet" for row in matrix)
    assert all(row["result"] == "INSUFFICIENT_INFORMATION" for row in matrix)
    assert all("runtime" in row["proof_limit"] for row in matrix)


def _write_checkpoint_inventory(root: Path) -> None:
    model_types = {"gdn": "ffnloop", "mixerloop": "mixerloop", "fullloop": "fullloop"}
    layers_by_scale = {"15m": 2, "42m": 3, "110m": 4}
    for condition, model_type in model_types.items():
        for scale, layers in layers_by_scale.items():
            checkpoint_dir = root / f"{condition}-{scale}"
            checkpoint_dir.mkdir(parents=True)
            (checkpoint_dir / "config.json").write_text(
                json.dumps(
                    {
                        "model_type": model_type,
                        "hidden_size": 8,
                        "num_hidden_layers": layers,
                        "loop_count": 4,
                    },
                    indent=2,
                )
                + "\n"
            )
            (checkpoint_dir / "model.safetensors").write_bytes(b"not-loaded")


def test_released_checkpoint_inventory_parses_configs_without_loading_weights(tmp_path: Path, theory_probe):
    outputs_root = tmp_path / "outputs"
    _write_checkpoint_inventory(outputs_root)

    inventory = theory_probe.parse_released_checkpoint_inventory(outputs_root)

    assert len(inventory) == 9
    assert {row["checkpoint"] for row in inventory} == {
        f"{condition}-{scale}"
        for condition in ("gdn", "mixerloop", "fullloop")
        for scale in ("15m", "42m", "110m")
    }
    assert {row["weights_path"] for row in inventory} == {
        str(outputs_root / row["checkpoint"] / "model.safetensors") for row in inventory
    }
    assert all(row["weights_loaded"] is False for row in inventory)
    assert all(row["config_sha256"] for row in inventory)
    assert {row["model_type"] for row in inventory} == {"ffnloop", "mixerloop", "fullloop"}


def test_runner_manifest_reuses_fixed_split_and_deterministic_audit(tmp_path: Path, theory_probe):
    _, fit_ids, heldout_ids = _write_spec01_artifacts(tmp_path / "spec01")
    dataset = theory_probe.load_fixed_probe_dataset(tmp_path / "spec01")
    _write_checkpoint_inventory(tmp_path / "outputs")
    inventory = theory_probe.parse_released_checkpoint_inventory(tmp_path / "outputs")

    audit = theory_probe.select_deterministic_audit_sample_ids(dataset, full_raw_count=4, hidden_state_count=6)
    manifest = theory_probe.build_spec02_run_manifest(
        dataset=dataset,
        checkpoint_inventory=inventory,
        output_root=tmp_path / "spec02",
        command=["python", "-m", "eval.theory_probe", "smoke"],
        wandb_plan=theory_probe.build_wandb_mode_plan(formal=False, wandb_enabled=True),
        cap_bytes=99_000_000_000,
        audit_selection=audit,
    )

    assert manifest["spec01"]["fit_sample_ids"] == fit_ids
    assert manifest["spec01"]["heldout_sample_ids"] == heldout_ids
    assert manifest["spec01"]["theory_eval_ids_sha256"] == dataset.theory_eval_ids_sha256
    assert manifest["spec01"]["probe_split_metadata_sha256"] == dataset.probe_split_metadata_sha256
    assert manifest["audit_selection"]["full_raw_sample_ids"] == [0, 1, 2, 3]
    assert manifest["audit_selection"]["hidden_state_sample_ids"] == [0, 1, 2, 3, 4, 5]
    assert manifest["no_full_raw_archive_dependency"] is True
    assert manifest["output_root_cap_bytes"] == 99_000_000_000
    assert manifest["wandb"]["mode"] == "offline"
    assert manifest["checkpoint_count"] == 9
    assert manifest["git"]["commit"] == "NOT_RECORDED"
    assert manifest["git"]["status_short"] == "NOT_RECORDED"
    assert manifest["runtime_proof_placeholders"]["checkpoint_loaded"] == "INSUFFICIENT_INFORMATION"
    assert manifest["runtime_proof_placeholders"]["probe_training"] == "INSUFFICIENT_INFORMATION"
    assert manifest["runtime_proof_placeholders"]["frozen_head_eval"] == "INSUFFICIENT_INFORMATION"
    assert manifest["runtime_proof_placeholders"]["runtime_acceleration"] == "INSUFFICIENT_INFORMATION"


def test_runtime_target_manifest_preserves_canonical_readout_boundary(tmp_path: Path, theory_probe):
    _write_checkpoint_inventory(tmp_path / "outputs")
    inventory = theory_probe.parse_released_checkpoint_inventory(tmp_path / "outputs")

    manifest = theory_probe.build_runtime_target_manifest(inventory)

    gdn = [row for row in manifest if row["condition"] == "gdn" and row["scale"] == "15m"]
    mixer = [row for row in manifest if row["condition"] == "mixerloop" and row["scale"] == "15m"]
    full = [row for row in manifest if row["condition"] == "fullloop" and row["scale"] == "15m"]

    assert {(row["pass"], row["record_type"]) for row in gdn} == {(1, "hA")}
    assert {(row["pass"], row["record_type"]) for row in mixer} == {
        (1, "hA"),
        (2, "hA"),
        (3, "hA"),
        (4, "hA"),
    }
    assert {(row["pass"], row["record_type"]) for row in full} == {
        (1, "hA"),
        (1, "hF"),
        (2, "hA"),
        (2, "hF"),
        (3, "hA"),
        (3, "hF"),
        (4, "hA"),
        (4, "hF"),
    }
    assert all(row["readouts"] == ["linear_probe", "frozen_lm_head"] for row in manifest)


def test_frozen_state_transition_records_replay_and_readout_mutation(theory_probe):
    model = _ToyBase()

    replay = theory_probe.record_frozen_state_transition(
        model,
        stage="replay",
        operation=lambda: torch.zeros(1),
    )
    assert replay["stage"] == "replay"
    assert replay["frozen"] is True
    assert replay["before_digest"] == replay["after_digest"]

    def mutate_readout():
        with torch.no_grad():
            model.lm_head.weight.add_(1.0)

    readout = theory_probe.record_frozen_state_transition(model, stage="readout", operation=mutate_readout)
    assert readout["stage"] == "readout"
    assert readout["frozen"] is False
    assert readout["before_digest"] != readout["after_digest"]


def test_streamed_stats_bootstrap_and_detail_table_schema(theory_probe):
    target = theory_probe.ProbeTarget(
        checkpoint="gdn-15m",
        scale="15m",
        condition="gdn",
        layer=0,
        pass_index=1,
        record_type="hA",
    )
    rows = [
        theory_probe.build_streamed_sample_stats_row(
            target=target,
            sample_id=sample_id,
            readout="linear_probe",
            ce_sum=10.0 + sample_id,
            num_positions=5,
            top1_correct=2,
            top5_correct=4,
            precision="float32",
            input_sha256=f"input-{sample_id}",
        )
        for sample_id in range(3)
    ]

    bootstrap = theory_probe.bootstrap_readout_ci(rows, n_bootstrap=1000, seed=20260722)
    detail = theory_probe.build_probe_detail_row(target=target, readout="linear_probe", bootstrap=bootstrap)

    assert rows[0]["precision"] == "float32"
    assert rows[0]["input_sha256"] == "input-0"
    assert bootstrap["bootstrap_samples"] == 1000
    assert set(detail) == set(theory_probe.PROBE_DETAIL_COLUMNS)
    assert detail["record_type"] == "hA"
    assert detail["heldout_loss"] == pytest.approx(bootstrap["heldout_loss"])
    assert detail["top1"] == pytest.approx(bootstrap["top1"])
    assert detail["top5"] == pytest.approx(bootstrap["top5"])


def test_command_run_checks_manifest_and_output_guard(tmp_path: Path, theory_probe):
    output_root = tmp_path / "spec02"
    output_root.mkdir()
    manifest = {
        "command": ["python", "-m", "eval.theory_probe", "smoke"],
        "checkpoint_count": 9,
        "output_root_cap_bytes": 99_000_000_000,
        "wandb": theory_probe.build_wandb_mode_plan(formal=False, wandb_enabled=False),
    }
    checks = theory_probe.build_spec02_run_checks(
        manifest,
        acceleration_matrix=theory_probe.build_mixerloop_acceleration_matrix(device_type="cpu", wandb_enabled=False),
        runtime_executed=False,
    )

    assert checks["runtime_executed"] is False
    assert checks["runtime_rows_result"] == "INSUFFICIENT_INFORMATION"
    assert checks["runtime_proof_status"] == "INSUFFICIENT_INFORMATION"
    assert checks["runtime_proof_placeholders"]["checkpoint_loaded"] == "INSUFFICIENT_INFORMATION"
    assert checks["wandb_mode"] == "not_initialized"
    theory_probe.write_spec02_json_guarded(output_root / "run_manifest.json", manifest, output_root=output_root, cap_bytes=10_000)
    assert (output_root / "run_manifest.json").is_file()


def test_run_checks_do_not_pass_with_incomplete_runtime_placeholders(theory_probe):
    manifest = {
        "command": ["python", "-m", "eval.theory_probe", "smoke"],
        "checkpoint_count": 9,
        "output_root_cap_bytes": 99_000_000_000,
        "wandb": theory_probe.build_wandb_mode_plan(formal=False, wandb_enabled=False),
        "no_full_raw_archive_dependency": True,
        "runtime_proof_placeholders": {
            "checkpoint_loaded": "INSUFFICIENT_INFORMATION",
            "probe_training": "INSUFFICIENT_INFORMATION",
            "frozen_head_eval": "INSUFFICIENT_INFORMATION",
            "runtime_acceleration": "INSUFFICIENT_INFORMATION",
        },
    }
    matrix = theory_probe.build_mixerloop_acceleration_matrix(device_type="cpu", wandb_enabled=False)

    checks = theory_probe.build_spec02_run_checks(manifest, acceleration_matrix=matrix, runtime_executed=True)

    assert checks["runtime_executed"] is True
    assert checks["runtime_proof_status"] == "INSUFFICIENT_INFORMATION"
    assert checks["runtime_rows_result"] == "INSUFFICIENT_INFORMATION"


def test_figure_table_input_contracts_require_real_rows(theory_probe):
    target = theory_probe.ProbeTarget(
        checkpoint="fullloop-42m",
        scale="42m",
        condition="fullloop",
        layer=1,
        pass_index=4,
        record_type="hF",
    )
    detail = theory_probe.build_probe_detail_row(
        target=target,
        readout="frozen_lm_head",
        bootstrap={
            "heldout_loss": 1.25,
            "top1": 0.2,
            "top5": 0.7,
            "ci_low": 1.0,
            "ci_high": 1.5,
        },
    )

    contract = theory_probe.build_figure_table_inputs([detail])

    assert contract["detail_columns"] == theory_probe.PROBE_DETAIL_COLUMNS
    assert detail["record_type"] == "hF"
    assert contract["record_types"] == ["hF"]
    assert contract["heatmap_index"] == ["checkpoint", "condition", "scale", "readout", "record_type", "layer", "pass"]
    assert contract["scale_curve_index"] == ["condition", "scale", "readout", "record_type"]
    assert contract["row_count"] == 1
    assert contract["runtime_artifact_status"] == "NOT_CREATED"


def test_figure_table_input_contract_rejects_empty_detail_rows(theory_probe):
    with pytest.raises(ValueError, match="detail_rows"):
        theory_probe.build_figure_table_inputs([])


def test_train_probe_full_split_records_formal_acceleration_evidence_on_cuda(theory_probe):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for formal acceleration evidence")

    train_features = torch.randn(16, 4)
    train_labels = torch.arange(16) % 7
    heldout_features = torch.randn(8, 4)
    heldout_labels = torch.arange(8) % 7

    _, training = theory_probe._train_probe_full_split(
        train_features,
        train_labels,
        heldout_features,
        heldout_labels,
        config=theory_probe.ProbeTrainingConfig(epochs=1, token_batch_size=5),
        vocab_size=7,
        device=torch.device("cuda"),
    )

    acceleration = training["acceleration"]
    assert acceleration["torch_compile"]["result"] == "PASS"
    assert acceleration["torch_compile"]["formal_run_runtime"] is True
    assert acceleration["pin_memory_cpu_batches"]["result"] == "PASS"
    assert acceleration["pin_memory_cpu_batches"]["cpu_to_gpu_pinned_memory"] is True
    assert acceleration["non_blocking_cpu_to_gpu_transfer"]["result"] == "PASS"
    assert acceleration["non_blocking_cpu_to_gpu_transfer"]["cpu_to_gpu_copy_non_blocking"] is True
    assert acceleration["non_blocking_cpu_to_gpu_transfer"]["cpu_to_gpu_copy_event_count"] > 0
    assert acceleration["gpu_utilization_memory_trace"]["result"] == "PASS"
    assert acceleration["gpu_utilization_memory_trace"]["formal_run_runtime"] is True
    assert acceleration["gpu_utilization_memory_trace"]["peak_gpu_memory_allocated_mb"] >= 0
