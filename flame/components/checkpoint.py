# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field
from datetime import timedelta
from io import BytesIO
from pathlib import Path
import re
from typing import Any, Dict, List

import torch
from torch.distributed.checkpoint.stateful import Stateful
from torchtitan.components.checkpoint import CheckpointManager as TitanCheckpointManager


class CheckpointManager(TitanCheckpointManager):
    """Keep selected scientific milestones alongside the rolling recovery window."""

    def __init__(self, *args, job_config, **kwargs):
        self.milestone_steps = frozenset(
            int(step.strip())
            for step in job_config.checkpoint.milestone_steps.split(",")
            if step.strip()
        )
        if any(step <= 0 or step > job_config.training.steps for step in self.milestone_steps):
            raise ValueError("Checkpoint milestones must lie within the training budget")
        if self.milestone_steps and job_config.checkpoint.last_save_model_weights_only:
            raise ValueError("Milestones require full resumable checkpoints")
        super().__init__(*args, job_config=job_config, **kwargs)

    def save(self, curr_step: int, force: bool = False):
        return super().save(curr_step, force=force or curr_step in self.milestone_steps)

    def _purge_stale_checkpoints(self):
        if not self.milestone_steps:
            return super()._purge_stale_checkpoints()
        if (
            self.keep_latest_k <= 0
            or torch.distributed.get_rank() != 0
            or not Path(self.folder).is_dir()
            or (self.ft_manager and self.ft_manager.participating_rank() != 0)
        ):
            return
        checkpoints = []
        for path in Path(self.folder).iterdir():
            match = re.fullmatch(r"step-(\d+)", path.name)
            if match and path.is_dir():
                checkpoints.append((int(match[1]), path))
        checkpoints.sort()
        for step, path in checkpoints[:-self.keep_latest_k]:
            if step not in self.milestone_steps:
                self.purge_queue.put(str(path))


@dataclass
class TrainState(Stateful):
    step: int = 0
    skipped_step: int = 0
    token: int = 0
    elapsed: timedelta = timedelta(0)
    global_avg_losses: List[float] = field(default_factory=list)
    global_max_losses: List[float] = field(default_factory=list)
    log_steps: List[int] = field(default_factory=list)

    def state_dict(self) -> Dict[str, Any]:
        # Only checkpoint global_avg_losses and global_max_losses per log frequency
        # to avoid sync overhead in every iteration.
        global_avg_losses_bytes = BytesIO()
        torch.save(self.global_avg_losses, global_avg_losses_bytes)
        global_max_losses_bytes = BytesIO()
        torch.save(self.global_max_losses, global_max_losses_bytes)
        log_steps_bytes = BytesIO()
        torch.save(self.log_steps, log_steps_bytes)
        return {
            "step": torch.tensor(self.step, dtype=torch.int32),
            "skipped_step": torch.tensor(self.skipped_step, dtype=torch.int32),
            "token": torch.tensor(self.token, dtype=torch.int64),
            "elapsed": self.elapsed,
            "global_avg_losses": global_avg_losses_bytes,
            "global_max_losses": global_max_losses_bytes,
            "log_steps": log_steps_bytes,
        }

    def load_state_dict(self, state_dict) -> None:
        self.step = state_dict["step"].item()
        self.skipped_step = state_dict.get("skipped_step", 0).item()
        self.token = state_dict["token"].item()
        self.elapsed = state_dict["elapsed"]
        state_dict["global_avg_losses"].seek(0)
        self.global_avg_losses = torch.load(
            state_dict["global_avg_losses"], weights_only=False
        )
        state_dict["global_max_losses"].seek(0)
        self.global_max_losses = torch.load(
            state_dict["global_max_losses"], weights_only=False
        )
        state_dict["log_steps"].seek(0)
        self.log_steps = torch.load(state_dict["log_steps"], weights_only=False)
