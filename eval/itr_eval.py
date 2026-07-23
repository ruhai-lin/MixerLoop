#!/usr/bin/env python3
"""Estimate ITR and per-step marginal ITR for MixerLoop checkpoints."""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SCRIPT_DIR = Path(__file__).resolve().parent
RUN_ROOT = SCRIPT_DIR.parent
if str(RUN_ROOT) not in sys.path:
    sys.path.insert(0, str(RUN_ROOT))

import custom_models  # noqa: E402,F401


def _active_indices(gram: torch.Tensor, relative_threshold: float) -> list[int]:
    diagonal = gram.diag().double().clamp_min(0)
    if diagonal.numel() == 0 or float(diagonal.max()) == 0:
        return []
    cutoff = relative_threshold**2 * float(diagonal.max())
    return torch.nonzero(diagonal > cutoff, as_tuple=False).flatten().tolist()


def effective_rank(gram: torch.Tensor, relative_threshold: float = 1e-6) -> float:
    """Participation-ratio rank after unit-normalizing active operators."""
    active = _active_indices(gram, relative_threshold)
    if not active:
        return 0.0
    selected = gram.double()[active][:, active]
    norms = selected.diag().clamp_min(1e-30).sqrt()
    normalized = selected / (norms[:, None] * norms[None, :])
    return float(normalized.trace().square() / normalized.square().sum().clamp_min(1e-30))


def marginal_itr(
    increment_gram: torch.Tensor,
    relative_threshold: float = 1e-6,
    pinv_rcond: float = 1e-8,
) -> list[float]:
    """Novel energy of each increment after projection onto preceding increments."""
    gram = 0.5 * (increment_gram.double() + increment_gram.double().T)
    active = set(_active_indices(gram, relative_threshold))
    values: list[float] = []
    for depth in range(gram.shape[0]):
        energy = float(gram[depth, depth].clamp_min(0))
        if depth not in active or energy == 0:
            values.append(0.0)
            continue
        prior = [index for index in range(depth) if index in active]
        if not prior:
            values.append(1.0)
            continue
        previous = gram[prior][:, prior]
        cross = gram[prior, depth]
        eigenvalues, eigenvectors = torch.linalg.eigh(previous)
        cutoff = pinv_rcond * float(eigenvalues.max().clamp_min(0))
        inverse = torch.where(eigenvalues > cutoff, eigenvalues.reciprocal(), 0)
        projected = float(((eigenvectors.T @ cross).square() * inverse).sum())
        residual = min(max(energy - projected, 0.0), energy)
        values.append(residual / energy)
    return values


def gram_summary(
    cumulative_gram: torch.Tensor,
    increment_gram: torch.Tensor,
    relative_threshold: float,
    pinv_rcond: float,
) -> dict[str, object]:
    loops = cumulative_gram.shape[0]
    return {
        'itr': effective_rank(cumulative_gram, relative_threshold),
        'itr_by_depth': [
            effective_rank(cumulative_gram[:depth, :depth], relative_threshold)
            for depth in range(1, loops + 1)
        ],
        'marginal_itr': marginal_itr(increment_gram, relative_threshold, pinv_rcond),
        'cumulative_active_depths': [
            index + 1 for index in _active_indices(cumulative_gram, relative_threshold)
        ],
        'increment_active_depths': [
            index + 1 for index in _active_indices(increment_gram, relative_threshold)
        ],
        'cumulative_gram': cumulative_gram.double().cpu().tolist(),
        'increment_gram': increment_gram.double().cpu().tolist(),
    }


def load_windows(args, tokenizer) -> torch.Tensor:
    required = args.num_windows * args.seq_len
    if args.text_file:
        text = Path(args.text_file).read_text(encoding='utf-8')
        tokens = tokenizer.encode(text, add_special_tokens=False)
        if len(tokens) < required:
            raise ValueError(f'{args.text_file} has {len(tokens)} tokens; {required} required')
        return torch.tensor(tokens[:required], dtype=torch.long).view(args.num_windows, args.seq_len)

    data_dir = Path(args.data_dir).expanduser().resolve()
    if args.dataset_split == 'validation':
        shards = [data_dir / 'shard_06542.bin']
    else:
        shards = [data_dir / f'shard_{index:05d}.bin' for index in range(170)]
    missing = [path for path in shards if not path.is_file()]
    if missing:
        raise FileNotFoundError(f'ClimbMix data is incomplete; first missing shard: {missing[0]}')

    rng = random.Random(args.data_seed)
    windows = []
    for _ in range(args.num_windows):
        shard = shards[rng.randrange(len(shards))]
        tokens = np.memmap(shard, dtype=np.uint16, mode='r')
        if len(tokens) < args.seq_len:
            raise ValueError(f'{shard} has fewer than {args.seq_len} tokens')
        start = rng.randrange(len(tokens) - args.seq_len + 1)
        windows.append(torch.from_numpy(tokens[start:start + args.seq_len].astype(np.int64)))
    return torch.stack(windows)


@torch.no_grad()
def layer_inputs(model, windows: torch.Tensor, layer_index: int, device: torch.device) -> list[torch.Tensor]:
    base = model.model
    states = []
    for window in windows:
        hidden = base.embeddings(window.unsqueeze(0).to(device))
        for layer in base.layers[:layer_index]:
            hidden = layer(hidden, residual_weight=base.residual_weight)
        states.append(hidden.detach())
    return states


def covariance_geometry(
    states: list[torch.Tensor],
    device: torch.device,
    ridge: float,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float | int]]:
    rows = torch.cat([state.float().cpu().reshape(-1, state.shape[-1]) for state in states]).double()
    rows -= rows.mean(dim=0, keepdim=True)
    covariance = rows.T @ rows / max(rows.shape[0] - 1, 1)
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    raw_minimum = float(eigenvalues.min())
    raw_maximum = float(eigenvalues.max())
    floor = max(ridge * raw_maximum, torch.finfo(torch.float64).eps)
    regularized = eigenvalues.clamp_min(floor)
    covariance = eigenvectors @ regularized.diag() @ eigenvectors.T
    inverse_sqrt = eigenvectors @ regularized.rsqrt().diag() @ eigenvectors.T
    diagnostics = {
        'samples': rows.shape[0],
        'features': rows.shape[1],
        'raw_minimum_eigenvalue': raw_minimum,
        'raw_maximum_eigenvalue': raw_maximum,
        'eigenvalue_floor': floor,
        'regularized_condition_number': float(regularized.max() / regularized.min()),
    }
    return covariance.float().to(device), inverse_sqrt.float().to(device), diagnostics


def mixer_trajectory(layer, h0: torch.Tensor, residual_weight: torch.Tensor | None, loops: int):
    states = []
    hidden = h0
    for loop_index in range(loops):
        loop_input = hidden
        update, _, _ = layer.mixer(layer.attn_norm(hidden))
        hidden = hidden + update
        if residual_weight is not None:
            hidden = hidden + residual_weight[loop_index].view(1, 1, -1) * loop_input
        states.append(hidden)
    return states


def metric_inner(left: torch.Tensor, right: torch.Tensor, covariance: torch.Tensor) -> torch.Tensor:
    left = left.float().reshape(-1, left.shape[-1])
    right = right.float().reshape(-1, right.shape[-1])
    return torch.einsum('td,df,tf->', left, covariance, right)


def relational_vjps(
    states: list[torch.Tensor],
    h0: torch.Tensor,
    output_token: int,
    output_vector: torch.Tensor,
) -> list[torch.Tensor]:
    vectors = []
    for state in states:
        probe = torch.zeros_like(state)
        probe[:, output_token, :] = output_vector
        sensitivity = torch.autograd.grad((state * probe).sum(), h0, retain_graph=True)[0]
        sensitivity = sensitivity.clone()
        sensitivity[:, output_token, :] = 0
        vectors.append(sensitivity.detach())
    return vectors


def estimate_window(
    layer,
    h0_value: torch.Tensor,
    residual_weight: torch.Tensor | None,
    loops: int,
    covariance: torch.Tensor,
    inverse_sqrt: torch.Tensor,
    probes: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    h0 = h0_value.detach().requires_grad_(True)
    states = mixer_trajectory(layer, h0, residual_weight, loops)
    cumulative_gram = torch.zeros((loops, loops), dtype=torch.float64, device=h0.device)
    increment_gram = torch.zeros_like(cumulative_gram)
    generator = torch.Generator(device=h0.device).manual_seed(seed)

    for _ in range(probes):
        output_token = int(torch.randint(h0.shape[1], (1,), generator=generator, device=h0.device))
        signs = torch.randint(0, 2, (h0.shape[-1],), generator=generator, device=h0.device)
        with torch.autocast(device_type=h0.device.type, enabled=False):
            output_vector = inverse_sqrt @ (2 * signs - 1).float()
        cumulative = relational_vjps(states, h0, output_token, output_vector.to(h0.dtype))
        increments = [cumulative[0]] + [
            cumulative[index] - cumulative[index - 1] for index in range(1, loops)
        ]
        for row in range(loops):
            for column in range(row, loops):
                cumulative_value = metric_inner(cumulative[row], cumulative[column], covariance)
                increment_value = metric_inner(increments[row], increments[column], covariance)
                cumulative_gram[row, column] += cumulative_value
                increment_gram[row, column] += increment_value
                if row != column:
                    cumulative_gram[column, row] += cumulative_value
                    increment_gram[column, row] += increment_value

    return cumulative_gram / probes, increment_gram / probes


def write_results(out_dir: Path, result: dict[str, object]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / 'itr_eval.json').open('w', encoding='utf-8') as handle:
        json.dump(result, handle, indent=2)

    rows = []
    for record in result['replicates']:
        for depth, (itr, mitr) in enumerate(
            zip(record['itr_by_depth'], record['marginal_itr']), start=1
        ):
            rows.append(
                {
                    'window': record['window'],
                    'probe_seed': record['probe_seed'],
                    'depth': depth,
                    'itr': itr,
                    'marginal_itr': mitr,
                }
            )
    with (out_dir / 'itr_eval.csv').open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    aggregate = result['aggregate']
    with (out_dir / 'itr_eval.md').open('w', encoding='utf-8') as handle:
        handle.write('# Iterative Transport Evaluation\n\n')
        handle.write(f"Model: `{result['model_path']}`  \n")
        handle.write(f"Layer: {result['layer']}  \n")
        handle.write(f"ITR: {aggregate['itr']:.4f}  \n")
        handle.write(
            'Marginal ITR: '
            + ', '.join(f'{value:.4f}' for value in aggregate['marginal_itr'])
            + '\n'
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--tokenizer_path', default=None)
    parser.add_argument('--out_dir', default=None)
    parser.add_argument('--layer', type=int, default=-1)
    parser.add_argument('--seq_len', type=int, default=128)
    parser.add_argument('--num_windows', type=int, default=8)
    parser.add_argument('--probes', type=int, default=16)
    parser.add_argument('--probe_seeds', type=int, default=2)
    parser.add_argument('--relative_threshold', type=float, default=1e-6)
    parser.add_argument('--pinv_rcond', type=float, default=1e-8)
    parser.add_argument('--covariance_ridge', type=float, default=1e-5)
    parser.add_argument('--data_dir', default='data/climbmix-10b')
    parser.add_argument('--dataset_split', choices=['train', 'validation'], default='validation')
    parser.add_argument('--data_seed', type=int, default=2027)
    parser.add_argument('--text_file', default=None)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    device = torch.device(args.device)
    tokenizer_path = args.tokenizer_path or args.model_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        dtype=torch.bfloat16 if device.type == 'cuda' else torch.float32,
    ).to(device)
    model.eval()

    if model.config.model_type != 'mixerloop':
        raise ValueError(
            f'ITR trajectory extraction requires MixerLoop, got {model.config.model_type}'
        )
    base = model.model
    layer_index = args.layer if args.layer >= 0 else len(base.layers) + args.layer
    if not 0 <= layer_index < len(base.layers):
        raise ValueError(f'layer {args.layer} resolves to invalid index {layer_index}')
    loops = int(model.config.loop_count)
    # FLA switches eval-mode sequences of length <=64 to a fused recurrent
    # kernel whose backward pass is intentionally unavailable. ITR needs VJPs,
    # so force the target mixer onto its training-mode chunk kernel. There is no
    # dropout or parameter update in this evaluation.
    base.layers[layer_index].mixer.train()

    windows = load_windows(args, tokenizer)
    inputs = layer_inputs(model, windows, layer_index, device)
    covariance, inverse_sqrt, covariance_info = covariance_geometry(
        inputs, device, args.covariance_ridge
    )

    cumulative_total = torch.zeros((loops, loops), dtype=torch.float64, device=device)
    increment_total = torch.zeros_like(cumulative_total)
    replicates = []
    for window_index, hidden in enumerate(inputs):
        for probe_seed in range(args.probe_seeds):
            seed = 1009 * args.data_seed + 9176 * window_index + probe_seed
            cumulative, increment = estimate_window(
                base.layers[layer_index],
                hidden,
                base.residual_weight,
                loops,
                covariance,
                inverse_sqrt,
                args.probes,
                seed,
            )
            cumulative_total += cumulative
            increment_total += increment
            summary = gram_summary(
                cumulative,
                increment,
                args.relative_threshold,
                args.pinv_rcond,
            )
            summary.update({'window': window_index, 'probe_seed': probe_seed})
            replicates.append(summary)
            print(
                f"[itr] window={window_index} seed={probe_seed} "
                f"ITR={summary['itr']:.4f} mITR={summary['marginal_itr']}"
            )

    count = args.num_windows * args.probe_seeds
    aggregate = gram_summary(
        cumulative_total / count,
        increment_total / count,
        args.relative_threshold,
        args.pinv_rcond,
    )
    replicate_itr = np.asarray([record['itr'] for record in replicates], dtype=float)
    replicate_mitr = np.asarray([record['marginal_itr'] for record in replicates], dtype=float)
    aggregate['replicate_itr_mean'] = float(replicate_itr.mean())
    aggregate['replicate_itr_std'] = float(replicate_itr.std(ddof=1)) if len(replicates) > 1 else 0.0
    aggregate['replicate_marginal_itr_mean'] = replicate_mitr.mean(axis=0).tolist()
    aggregate['replicate_marginal_itr_std'] = (
        replicate_mitr.std(axis=0, ddof=1).tolist() if len(replicates) > 1 else [0.0] * loops
    )

    result = {
        'model_path': os.path.abspath(args.model_path),
        'model_type': model.config.model_type,
        'layer': layer_index,
        'loop_count': loops,
        'seq_len': args.seq_len,
        'num_windows': args.num_windows,
        'probes_per_seed': args.probes,
        'probe_seeds': args.probe_seeds,
        'data_seed': args.data_seed,
        'covariance': covariance_info,
        'aggregate': aggregate,
        'replicates': replicates,
    }
    out_dir = Path(args.out_dir) if args.out_dir else Path(args.model_path) / 'itr_eval'
    write_results(out_dir, result)
    print(f'[itr] wrote {out_dir / "itr_eval.json"}')


if __name__ == '__main__':
    main()
