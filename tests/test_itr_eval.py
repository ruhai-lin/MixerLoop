from __future__ import annotations

import torch

from eval.itr_eval import effective_rank, marginal_itr


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
