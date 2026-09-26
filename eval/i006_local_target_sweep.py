"""Score the approved local target-block one-to-four Attention sweep.

All non-target MixerLoop blocks execute four native Attention calls.  The
selected target block executes one, two, three, or four calls and then its
original FFN exactly once.  No state, parameter, or output is replaced.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import torch


def set_all_loops(model, loops: int) -> None:
    model.model.loop_count = loops
    for block in model.model.layers:
        block.loop_count = loops


def target_block_once(block, hidden, residual_weight, loops: int):
    h = hidden
    for loop_index in range(loops):
        h_input = h
        attention_output, _, _ = block.mixer(
            block.attn_norm(h), attention_mask=None,
        )
        h = h + attention_output
        h = h + residual_weight[loop_index].view(1, 1, -1) * h_input
    return h + block.ffn(block.ffn_norm(h))


def score_windows(model, hidden, labels, positions):
    logits = model.lm_head(hidden[:, positions]).float()
    target = labels.to(logits.device)
    log_prob = logits.log_softmax(dim=-1)
    true_log_prob = log_prob.gather(-1, target.unsqueeze(-1)).squeeze(-1)
    correct = logits.argmax(dim=-1).eq(target)
    return {
        "ce": float((-true_log_prob).mean().detach().cpu()),
        "top1": float(correct.float().mean().detach().cpu()),
        "p_correct": float(true_log_prob.exp().mean().detach().cpu()),
    }


def run(args):
    repo = args.repo.resolve()
    sys.path.insert(0, str(repo))
    from eval.i004_capture import load_model
    from eval.i004_data import load_windows

    data = load_windows(args.selection, args.data_root)
    indices = [i for i, split in enumerate(data["split"]) if split == "validation"]
    if args.limit_windows is not None:
        indices = indices[: args.limit_windows]
    if args.pairs:
        pairs = json.loads(args.pairs.read_text())
    else:
        pairs = []
        seen = set()
        with args.quality.open(newline="") as quality_file:
            for row in csv.DictReader(quality_file):
                if row["pair_key"] not in seen:
                    pairs.append(row["pair_key"])
                    seen.add(row["pair_key"])
    if args.limit_models is not None:
        pairs = pairs[: args.limit_models]

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    rows_path = output / "local_target_quality.jsonl"
    manifest = {
        "experiment": "I006_local_target_attention_sweep",
        "pairs": pairs,
        "windows": len(indices),
        "target_layers": "mid,end",
        "other_blocks_loops": 4,
        "target_loops": [1, 2, 3, 4],
        "ffn_semantics": "one original FFN after the selected target-block Attention sequence",
        "precision": "float32; parameters frozen; native residual coefficients",
        "conditions": [],
    }
    positions = data["positions"].cuda()
    with rows_path.open("w") as rows_file:
        for pair in pairs:
            scale, training = pair.split("/", 1)
            checkpoint = args.weights_root / f"mixerloop-{scale}" / training
            model = load_model(checkpoint, device="cuda", dtype=torch.float32)
            model.requires_grad_(False)
            depth = len(model.model.layers)
            targets = [(depth + 1) // 2, depth]
            for target_layer in targets:
                for target_loops in (1, 2, 3, 4):
                    set_all_loops(model, 4)
                    model.model.layers[target_layer - 1].loop_count = target_loops
                    for offset in range(0, len(indices), args.batch_windows):
                        chosen = indices[offset:offset + args.batch_windows]
                        input_ids = data["input_ids"][chosen].cuda(non_blocking=True)
                        labels = data["labels"][chosen]
                        with torch.inference_mode():
                            hidden = model.model.embeddings(input_ids)
                            for index in range(target_layer - 1):
                                hidden = model.model.layers[index](
                                    hidden, model.model.residual_weight,
                                    attention_mask=None,
                                )
                            hidden = target_block_once(
                                model.model.layers[target_layer - 1], hidden,
                                model.model.residual_weight, target_loops,
                            )
                            for index in range(target_layer, depth):
                                hidden = model.model.layers[index](
                                    hidden, model.model.residual_weight,
                                    attention_mask=None,
                                )
                            hidden = model.model.norm(hidden)
                            score = score_windows(model, hidden, labels, positions)
                        row = {
                            "pair": pair,
                            "target_layer": target_layer,
                            "target_loops": target_loops,
                            "offset": offset,
                            "window_ids": [int(data["sample_ids"][i]) for i in chosen],
                            **score,
                        }
                        rows_file.write(json.dumps(row, ensure_ascii=False) + "\n")
                        rows_file.flush()
                        manifest["conditions"].append({
                            "pair": pair, "target_layer": target_layer,
                            "target_loops": target_loops, "offset": offset,
                            "windows": len(chosen),
                        })
                        print(json.dumps({"pair": pair, "layer": target_layer,
                                          "target_loops": target_loops,
                                          "offset": offset, **score}), flush=True)
                        del hidden, input_ids
                        torch.cuda.empty_cache()
            del model
            torch.cuda.empty_cache()
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps({"output": str(output), "pairs": len(pairs),
                      "windows": len(indices)}, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--weights-root", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--quality", type=Path, required=True)
    parser.add_argument("--pairs", type=Path)
    parser.add_argument("--batch-windows", type=int, default=16)
    parser.add_argument("--limit-models", type=int)
    parser.add_argument("--limit-windows", type=int)
    args = parser.parse_args()
    torch.manual_seed(20260919)
    torch.set_float32_matmul_precision("highest")
    run(args)


if __name__ == "__main__":
    main()
