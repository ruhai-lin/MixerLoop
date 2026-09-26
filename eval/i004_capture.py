"""Frozen I004 target-block branches using the released accelerated modules."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import torch
import torch.nn.functional as F


def load_model(checkpoint_path, device="cuda", dtype=torch.float32):
    # These imports register the released checkpoint model types with Transformers.
    import fla.models  # noqa: F401
    import custom_models.mixerloop  # noqa: F401
    import custom_models.fullloop  # noqa: F401
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        str(Path(checkpoint_path)), torch_dtype=dtype, local_files_only=True,
    ).to(device).eval()
    model.requires_grad_(False)
    return model


def model_kind(model):
    return "mixerloop" if model.config.model_type == "mixerloop" else "gdn"


def layer_roles(model):
    depth = len(model.model.layers)
    return {(depth + 1) // 2: "mid", depth: "end", (3 * depth + 3) // 4: "transfer"}


def _normal_block(model, hidden, index, attention_mask=None):
    layer = model.model.layers[index]
    if model_kind(model) == "mixerloop":
        return layer(hidden, model.model.residual_weight, attention_mask=attention_mask)
    return layer(hidden, attention_mask=attention_mask, use_cache=False, output_attentions=False)[0]


@torch.inference_mode()
def prefix_to_layer(model, input_ids, layer, attention_mask=None):
    """Return the unchanged input to the requested one-based physical layer."""
    hidden = model.model.embeddings(input_ids)
    for index in range(layer - 1):
        hidden = _normal_block(model, hidden, index, attention_mask)
    return hidden


@torch.inference_mode()
def continue_suffix(model, block_post, layer, attention_mask=None):
    """Continue after one-based layer and apply the original final norm once."""
    hidden = block_post
    for index in range(layer, len(model.model.layers)):
        hidden = _normal_block(model, hidden, index, attention_mask)
    return model.model.norm(hidden)


def _observe(observer, metadata, name, value):
    if observer is not None:
        observer({**metadata, "position_name": name}, value.detach())


def _ffn_observed(ffn, normalized, observer, metadata):
    # Hooks observe the projections used by the original fused FFN, rather than
    # replacing its fused SwiGLU/down-projection path with an unfused network.
    if observer is None:
        return ffn(normalized)
    projections = {}

    def gate_hook(module, args, output):
        projections["gate"] = output

    def up_hook(module, args, output):
        projections["up"] = output

    handles = [ffn.gate_proj.register_forward_hook(gate_hook), ffn.up_proj.register_forward_hook(up_hook)]
    try:
        result = ffn(normalized)
    finally:
        for handle in handles:
            handle.remove()
    gate = projections["gate"]
    up = projections["up"]
    gate_silu = F.silu(gate)
    _observe(observer, metadata, "gate_pre", gate)
    _observe(observer, metadata, "up", up)
    _observe(observer, metadata, "gate_silu", gate_silu)
    _observe(observer, metadata, "product", gate_silu * up)
    return result


@torch.inference_mode()
def target_branches(model, hidden, layer, observer=None, metadata=None, attention_mask=None):
    """Original attention trajectory, with one independent FFN branch per pass.

    The returned dict maps one-based pass numbers to tensor dictionaries.
    Only the last branch is used when advancing the unchanged deeper prefix.
    """
    block = model.model.layers[layer - 1]
    kind = model_kind(model)
    base_metadata = {**(metadata or {}), "model": kind, "layer": layer}
    branches = {}
    if kind == "mixerloop":
        attention_post = hidden
        for index in range(block.loop_count):
            residual_input = attention_post
            attention_output = block.mixer(
                block.attn_norm(attention_post), attention_mask=attention_mask,
                use_cache=False, output_attentions=False,
            )[0]
            attention_post = attention_post + attention_output
            attention_post = attention_post + model.model.residual_weight[index].view(1, 1, -1) * residual_input
            normalized = block.ffn_norm(attention_post)
            meta = {**base_metadata, "pass": index + 1, "loop": index + 1}
            update = _ffn_observed(block.ffn, normalized, observer, meta)
            block_post = attention_post + update
            branch = {
                "attention_post": attention_post, "ffn_input": normalized,
                "ffn_update": update, "block_post": block_post,
                "head_input": model.model.norm(block_post),
            }
            branches[index + 1] = branch
            for name, value in branch.items():
                _observe(observer, meta, name, value)
    else:
        residual = hidden
        attention_output = block.attn(
            hidden_states=block.attn_norm(hidden), attention_mask=attention_mask,
            use_cache=False, output_attentions=False,
        )[0]
        if block.config.fuse_norm:
            normalized, attention_post = block.mlp_norm(attention_output, residual, True)
        else:
            attention_post = residual + attention_output
            normalized = block.mlp_norm(attention_post)
        meta = {**base_metadata, "pass": 1, "loop": 1}
        update = _ffn_observed(block.mlp, normalized, observer, meta)
        block_post = attention_post + update
        branch = {
            "attention_post": attention_post, "ffn_input": normalized,
            "ffn_update": update, "block_post": block_post,
            "head_input": model.model.norm(block_post),
        }
        branches[1] = branch
        for name, value in branch.items():
            _observe(observer, meta, name, value)
    return branches


@torch.inference_mode()
def _capture_fullloop(model, input_ids, targets, observer, metadata, attention_mask, observed):
    records = {}
    states = {}
    handles = []
    loop_index = [0]
    loop_input = [None]

    def loop_input_hook(module, args):
        loop_index[0] += 1
        loop_input[0] = args[0]

    handles.append(model.model.layers[0].register_forward_pre_hook(loop_input_hook))

    def meta_for(index):
        pass_index = states[index]["pass"]
        return {**(metadata or {}), "model": "fullloop", "layer": index + 1,
                "pass": pass_index, "loop": pass_index}

    def report(index, name, value):
        if observer is not None and index + 1 in observed:
            _observe(observer, meta_for(index), name, value)

    for index, block in enumerate(model.model.layers):
        if index + 1 not in targets:
            continue

        def block_input_hook(module, args, index=index):
            states[index] = {"pass": loop_index[0], "input": args[0], "projections": {}}

        def attention_hook(module, args, output, index=index):
            attention_post = states[index]["input"] + output[0]
            states[index]["attention_post"] = attention_post
            report(index, "attention_post", attention_post)

        def ffn_input_hook(module, args, output, index=index):
            states[index]["ffn_input"] = output
            report(index, "ffn_input", output)

        def ffn_hook(module, args, output, index=index):
            state = states[index]
            state["ffn_update"] = output
            if observer is not None and index + 1 in observed:
                gate = state["projections"]["gate"]
                up = state["projections"]["up"]
                gate_silu = F.silu(gate)
                report(index, "gate_pre", gate)
                report(index, "up", up)
                report(index, "gate_silu", gate_silu)
                report(index, "product", gate_silu * up)
            report(index, "ffn_update", output)

        def block_output_hook(module, args, output, index=index):
            state = states[index]
            layer = index + 1
            readout_post = output
            if layer == len(model.model.layers) and model.model.residual_weight is not None:
                readout_post = output + model.model.residual_weight[state["pass"] - 1].view(1, 1, -1) * loop_input[0]
            branch = {
                "attention_post": state["attention_post"],
                "ffn_input": state["ffn_input"],
                "ffn_update": state["ffn_update"],
                "block_post": output,
                "head_input": model.model.norm(readout_post),
            }
            records[(layer, state["pass"])] = branch
            report(index, "block_post", output)
            report(index, "head_input", branch["head_input"])
            states[index] = None

        handles.extend((
            block.register_forward_pre_hook(block_input_hook),
            block.mixer.register_forward_hook(attention_hook),
            block.ffn_norm.register_forward_hook(ffn_input_hook),
            block.ffn.register_forward_hook(ffn_hook),
            block.register_forward_hook(block_output_hook),
        ))
        if observer is not None and index + 1 in observed:
            handles.extend((
                block.ffn.gate_proj.register_forward_hook(
                    lambda module, args, output, index=index: states[index]["projections"].__setitem__("gate", output)),
                block.ffn.up_proj.register_forward_hook(
                    lambda module, args, output, index=index: states[index]["projections"].__setitem__("up", output)),
            ))

    try:
        model.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False,
                    output_hidden_states=False, return_dict=True)
    finally:
        for handle in handles:
            handle.remove()
    return records


@torch.inference_mode()
def capture_batch(model, input_ids, target_layers=None, observer: Callable | None = None,
                  metadata=None, attention_mask=None, observer_layers=None,
                  capture_full_loop=False):
    """Capture all requested physical layers from one unchanged full prefix."""
    targets = set(target_layers if target_layers is not None else layer_roles(model))
    observed = targets if observer_layers is None else set(observer_layers)
    if capture_full_loop:
        return _capture_fullloop(model, input_ids, targets, observer, metadata,
                                 attention_mask, observed)
    hidden = model.model.embeddings(input_ids)
    records = {}
    for index in range(max(targets)):
        layer = index + 1
        if layer in targets:
            branches = target_branches(model, hidden, layer, observer if layer in observed else None,
                                       metadata, attention_mask)
            records.update({(layer, loop): values for loop, values in branches.items()})
            hidden = branches[max(branches)]["block_post"]
        else:
            hidden = _normal_block(model, hidden, index, attention_mask)
    return records


@torch.inference_mode()
def head_metrics(head_input, head_weight, labels, token_chunk=512):
    """Exact full-vocabulary CE/top-1/correct-token probability per window.

    Vocabulary logits are temporary token chunks and are never persisted.
    """
    batch, positions, width = head_input.shape
    flattened = head_input.reshape(-1, width)
    targets = labels.reshape(-1).to(head_input.device)
    losses, correct, probabilities = [], [], []
    for start in range(0, len(flattened), token_chunk):
        target = targets[start:start + token_chunk]
        logits = F.linear(flattened[start:start + token_chunk], head_weight).float()
        log_z = torch.logsumexp(logits, dim=-1)
        log_true = logits.gather(1, target[:, None]).squeeze(1) - log_z
        losses.append(-log_true)
        correct.append(logits.argmax(dim=-1).eq(target))
        probabilities.append(log_true.exp())
    return {
        "nll_sum": torch.cat(losses).reshape(batch, positions).double().sum(1).cpu(),
        "correct_sum": torch.cat(correct).reshape(batch, positions).sum(1).cpu(),
        "p_correct_sum": torch.cat(probabilities).reshape(batch, positions).double().sum(1).cpu(),
        "count": positions,
    }
