from __future__ import annotations

import torch

from eval.itr_eval import ContextPrefix, effective_rank, marginal_itr, prediction_positions


def test_effective_rank_separates_aligned_and_orthogonal_trajectories():
    aligned = torch.ones(4, 4)
    orthogonal = torch.eye(4)

    assert effective_rank(aligned) == 1.0
    assert effective_rank(orthogonal) == 4.0


def test_marginal_itr_is_per_step_projection_residual():
    increments = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
        ]
    )
    gram = increments @ increments.T

    torch.testing.assert_close(
        torch.tensor(marginal_itr(gram)),
        torch.tensor([1.0, 1.0, 0.0]),
    )


def test_marginal_itr_marks_zero_increment_inactive():
    gram = torch.diag(torch.tensor([1.0, 0.0, 1.0]))

    torch.testing.assert_close(
        torch.tensor(marginal_itr(gram)),
        torch.tensor([1.0, 0.0, 1.0]),
    )


class CumulativeMixer(torch.nn.Module):
    def forward(self, hidden_states, attention_mask=None):
        del attention_mask
        return hidden_states.cumsum(dim=1), None, None


def test_context_prefix_removes_only_cross_token_computation():
    mixer = CumulativeMixer()
    hidden = torch.tensor([[[1.0], [2.0], [3.0]]])

    with ContextPrefix(mixer, native_prefix=0) as local:
        context_off = mixer(hidden)[0]
    with ContextPrefix(mixer, native_prefix=1) as native:
        context_on = mixer(hidden)[0]

    torch.testing.assert_close(context_off, hidden)
    torch.testing.assert_close(context_on, hidden.cumsum(dim=1))
    assert local.calls == native.calls == 1


def test_prediction_positions_are_unique_and_have_next_token_targets():
    positions = prediction_positions(seq_len=128, count=16)

    assert len(positions.unique()) == 16
    assert int(positions.min()) >= 64
    assert int(positions.max()) <= 126
