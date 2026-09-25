from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from eval.theory_capture import (
    AUTHORIZED_OUTPUT_LIMIT_BYTES,
    BYTE_BUDGET,
    DEFAULT_METRIC_END,
    DEFAULT_METRIC_START,
    OUTPUT_ROOT_CAP_BYTES,
    OutputBudgetError,
    audit_tensor_filename,
    build_audit_plan,
    build_fixed_sample_artifacts,
    build_record_specs,
    build_stats_row,
    collect_record_hidden_states,
    logits_from_record_hidden,
    model_state_digest,
    next_token_sufficient_stats,
    write_json_guarded,
)


def _write_validation_bin(data_dir: Path, *, num_windows: int, seq_len: int = 512) -> Path:
    data_dir.mkdir(parents=True)
    token_count = (num_windows + 1) * seq_len + 1
    tokens = (np.arange(token_count, dtype=np.uint32) % 31999 + 1).astype(np.uint16)
    path = data_dir / "shard_06542.bin"
    tokens.tofile(path)
    return path


def test_deterministic_ids_and_split_metadata_are_persisted(tmp_path: Path):
    data_dir = tmp_path / "data"
    output_a = tmp_path / "out_a"
    output_b = tmp_path / "out_b"
    _write_validation_bin(data_dir, num_windows=1100)

    first = build_fixed_sample_artifacts(data_dir, output_a, sample_count=1024, seed=20260722)
    second = build_fixed_sample_artifacts(data_dir, output_b, sample_count=1024, seed=20260722)

    ids_a = [record["sample_id"] for record in first["selection"]["records"]]
    ids_b = [record["sample_id"] for record in second["selection"]["records"]]
    assert ids_a == ids_b
    assert len(ids_a) == 1024
    assert first["selection"]["seed"] == 20260722
    assert first["selection"]["metric_positions"] == {"start": DEFAULT_METRIC_START, "end": DEFAULT_METRIC_END}
    assert all(record["num_non_padding_tokens"] == 512 for record in first["selection"]["records"])

    split = first["split"]
    assert len(split["fit_sample_ids"]) == 819
    assert len(split["heldout_sample_ids"]) == 205
    assert set(split["fit_sample_ids"]).isdisjoint(split["heldout_sample_ids"])
    assert set(split["fit_sample_ids"]) | set(split["heldout_sample_ids"]) == set(ids_a)

    persisted = json.loads((output_a / "theory_eval_ids.json").read_text())
    persisted_split = json.loads((output_a / "probe_split_metadata.json").read_text())
    assert persisted == first["selection"]
    assert persisted_split == split


def test_streaming_storage_budget_and_hard_stop_prevent_over_budget_write(tmp_path: Path):
    assert sum(BYTE_BUDGET.values()) == OUTPUT_ROOT_CAP_BYTES
    assert OUTPUT_ROOT_CAP_BYTES == 99_000_000_000
    assert OUTPUT_ROOT_CAP_BYTES < AUTHORIZED_OUTPUT_LIMIT_BYTES

    output_root = tmp_path / "out"
    output_root.mkdir()
    (output_root / "existing.bin").write_bytes(b"1234567890")
    target = output_root / "new.json"

    with pytest.raises(OutputBudgetError):
        write_json_guarded(target, {"payload": "x" * 100}, output_root=output_root, cap_bytes=32)
    assert not target.exists()

    write_json_guarded(target, {"ok": True}, output_root=output_root, cap_bytes=256)
    assert json.loads(target.read_text()) == {"ok": True}


def test_audit_selection_and_filename_schema_are_deterministic():
    sample_ids = list(range(100, 170))
    plan = build_audit_plan(sample_ids)
    assert plan.full_hidden_plus_logit_sample_ids == sample_ids[:4]
    assert plan.hidden_state_sample_ids == sample_ids[:64]
    assert plan.hidden_only_sample_ids == sample_ids[4:64]
    assert set(plan.full_hidden_plus_logit_sample_ids).isdisjoint(plan.hidden_only_sample_ids)

    assert plan.should_persist(sample_ids[0], "hidden")
    assert plan.should_persist(sample_ids[0], "logits")
    assert plan.should_persist(sample_ids[63], "hidden")
    assert not plan.should_persist(sample_ids[63], "logits")
    assert not plan.should_persist(sample_ids[64], "hidden")

    name = audit_tensor_filename(
        checkpoint="mixerloop-15m",
        scale="15m",
        condition="mixerloop",
        layer=3,
        pass_index=4,
        sample_id=101,
        record_type="hF",
        kind="logits",
    )
    assert name == "checkpoint=mixerloop-15m__scale=15m__condition=mixerloop__layer=3__pass=4__sample_id=101__record_type=hF__kind=logits.pt"


def test_loop_record_contract_keeps_allowed_locations_separate():
    gdn = build_record_specs(condition="gdn", scale="15m", num_layers=2, loop_count=4)
    assert [(record.layer, record.pass_index, record.record_type) for record in gdn] == [
        (0, 1, "hA"),
        (1, 1, "hA"),
    ]

    mixer = build_record_specs(condition="mixerloop", scale="15m", num_layers=1, loop_count=4)
    assert [(record.layer, record.pass_index, record.record_type) for record in mixer] == [
        (0, 1, "hA"),
        (0, 2, "hA"),
        (0, 3, "hA"),
        (0, 4, "hA"),
        (0, 4, "hF"),
    ]

    full = build_record_specs(condition="fullloop", scale="15m", num_layers=1, loop_count=4)
    assert [(record.layer, record.pass_index, record.record_type) for record in full] == [
        (0, 1, "hA"),
        (0, 1, "hF"),
        (0, 2, "hA"),
        (0, 2, "hF"),
        (0, 3, "hA"),
        (0, 3, "hF"),
        (0, 4, "hA"),
        (0, 4, "hF"),
    ]


class _Identity(torch.nn.Module):
    def forward(self, x):
        return x


class _AddMixer(torch.nn.Module):
    def forward(self, x, **kwargs):
        return torch.ones_like(x), None, None


class _AddFfn(torch.nn.Module):
    def forward(self, x, **kwargs):
        return torch.full_like(x, 100)


class _FakeBlock(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.attn_norm = _Identity()
        self.ffn_norm = _Identity()
        self.mixer = _AddMixer()
        self.ffn = _AddFfn()


class _FakeBackbone(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embeddings = torch.nn.Embedding(8, 1)
        torch.nn.init.zeros_(self.embeddings.weight)
        self.layers = torch.nn.ModuleList([_FakeBlock()])
        self.norm = _Identity()
        self.loop_count = 2
        self.residual_weight = torch.nn.Parameter(torch.zeros(2, 1), requires_grad=False)


class _FakeLm(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = _FakeBackbone()
        self.lm_head = torch.nn.Linear(1, 4, bias=False)
        self.lm_head.weight.data[:, 0] = torch.tensor([0.0, 1.0, 2.0, 3.0])


def test_collect_record_hidden_states_computes_loop_locations_in_memory():
    input_ids = torch.tensor([[1, 2, 3]])

    mixer_records = collect_record_hidden_states(_FakeLm(), input_ids, condition="mixerloop", scale="15m")
    assert [(record.pass_index, record.record_type) for record, _ in mixer_records] == [(1, "hA"), (2, "hA"), (2, "hF")]
    assert [float(hidden[0, 0, 0]) for _, hidden in mixer_records] == [1.0, 2.0, 102.0]

    full_records = collect_record_hidden_states(_FakeLm(), input_ids, condition="fullloop", scale="15m")
    assert [(record.pass_index, record.record_type) for record, _ in full_records] == [
        (1, "hA"),
        (1, "hF"),
        (2, "hA"),
        (2, "hF"),
    ]
    assert [float(hidden[0, 0, 0]) for _, hidden in full_records] == [1.0, 101.0, 102.0, 202.0]


def test_logits_from_record_hidden_uses_final_norm_and_lm_head():
    model = _FakeLm()
    hidden = torch.tensor([[[2.0]]])
    logits = logits_from_record_hidden(model, hidden)
    assert logits.tolist() == [[[0.0, 2.0, 4.0, 6.0]]]


def test_record_and_logit_computation_uses_eval_no_grad_and_preserves_state():
    model = _FakeLm()
    model.train()
    before = model_state_digest(model)
    input_ids = torch.tensor([[1, 2, 3]])

    records = collect_record_hidden_states(model, input_ids, condition="mixerloop", scale="15m")
    logits = logits_from_record_hidden(model, records[0][1])

    assert not model.training
    assert all(not hidden.requires_grad for _, hidden in records)
    assert not logits.requires_grad
    assert model_state_digest(model) == before


def test_baseline_loss_uses_hidden_positions_predicting_next_token():
    input_ids = torch.tensor([[10, 11, 12, 13, 14, 15, 16, 17]])
    vocab_size = 32
    logits = torch.zeros((1, input_ids.shape[1], vocab_size), dtype=torch.float32)
    for position in range(2, 6):
        next_token = int(input_ids[0, position + 1])
        logits[0, position, next_token] = 50.0
    logits[0, 1, int(input_ids[0, 2])] = -50.0

    stats = next_token_sufficient_stats(logits, input_ids, metric_start=2, metric_end=5)
    assert stats["num_positions"] == 4
    assert stats["top1_correct"] == 4
    assert stats["top5_correct"] == 4
    assert stats["ce_sum"] < 1e-5


def test_streamed_stats_row_has_required_schema():
    row = build_stats_row(
        checkpoint="gdn-15m",
        scale="15m",
        condition="gdn",
        sample_id=95,
        readout="baseline",
        stats={"ce_sum": 12.5, "num_positions": 479, "top1_correct": 3, "top5_correct": 19},
        precision="bf16",
        input_sha256="a" * 64,
        layer=None,
        pass_index=None,
        record_type="baseline",
    )
    assert row == {
        "checkpoint": "gdn-15m",
        "scale": "15m",
        "condition": "gdn",
        "sample_id": 95,
        "layer": "",
        "pass": "",
        "record_type": "baseline",
        "readout": "baseline",
        "precision": "bf16",
        "input_sha256": "a" * 64,
        "ce_sum": 12.5,
        "num_positions": 479,
        "top1_correct": 3,
        "top5_correct": 19,
    }
