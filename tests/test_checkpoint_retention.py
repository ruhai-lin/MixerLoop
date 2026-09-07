from pathlib import Path
from queue import Queue
from unittest.mock import patch

from flame.components.checkpoint import CheckpointManager
from torchtitan.components.checkpoint import CheckpointManager as TitanCheckpointManager


def test_milestones_survive_rolling_retention(tmp_path):
    manager = object.__new__(CheckpointManager)
    manager.milestone_steps = frozenset({10, 20})
    manager.keep_latest_k = 3
    manager.folder = str(tmp_path)
    manager.ft_manager = None
    manager.purge_queue = Queue()
    for step in [2, 4, 10, 12, 14, 20, 22, 24, 26]:
        (tmp_path / f"step-{step}").mkdir()
    (tmp_path / "manifest.json").write_text("{}")
    with patch("torch.distributed.get_rank", return_value=0):
        manager._purge_stale_checkpoints()
    queued = set()
    while not manager.purge_queue.empty():
        queued.add(Path(manager.purge_queue.get()).name)
    assert queued == {"step-2", "step-4", "step-12", "step-14"}
    assert (tmp_path / "manifest.json").exists()


def test_milestone_forces_save_between_intervals():
    manager = object.__new__(CheckpointManager)
    manager.milestone_steps = frozenset({76294})
    with patch.object(TitanCheckpointManager, "save") as save:
        manager.save(76294)
        save.assert_called_once_with(76294, force=True)


def test_no_milestones_uses_native_retention():
    manager = object.__new__(CheckpointManager)
    manager.milestone_steps = frozenset()
    with patch.object(TitanCheckpointManager, "_purge_stale_checkpoints") as purge:
        manager._purge_stale_checkpoints()
        purge.assert_called_once_with()
