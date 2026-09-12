import os
import shutil
import subprocess
from pathlib import Path

import pytest


@pytest.mark.parametrize("exit_code", [0, 7])
def test_launcher_passes_arguments_and_exports_only_on_success(tmp_path, exit_code):
    shutil.copyfile(Path(__file__).resolve().parents[1] / "train.sh", tmp_path / "train.sh")
    bin_dir = tmp_path / ".venv/bin"
    bin_dir.mkdir(parents=True)
    for name, body in {
        "torchrun": 'printf "%s\\n" "$WANDB_PROJECT" "$WANDB_NAME" "$@" > "$RECORD/train"\nexit "$TRAIN_EXIT"',
        "python": 'printf "%s\\n" "$@" > "$RECORD/export"',
    }.items():
        executable = bin_dir / name
        executable.write_text("#!/usr/bin/env bash\n" + body + "\n")
        executable.chmod(0o755)
    args = ["--job.dump_folder", "outputs/model/run with space", "--training.steps=7630",
            "--model.tokenizer_path", "assets/tokenizer", "--model.config", "configs/gdn_100m.json",
            "--training.data_files", "data/a b/*.parquet"]
    env = dict(os.environ, RECORD=str(tmp_path), TRAIN_EXIT=str(exit_code), NGPU="1",
               WANDB_PROJECT="torchtitan", WANDB_NAME="fineweb100m_gdn_1b_s1337")
    result = subprocess.run(["bash", str(tmp_path / "train.sh"), *args], env=env)
    assert result.returncode == exit_code
    recorded = (tmp_path / "train").read_text().splitlines()
    assert recorded[:2] == ["mixerloop", "fineweb100m_gdn_1b_s1337"]
    assert recorded[-len(args):] == args
    assert (tmp_path / "export").exists() == (exit_code == 0)
    if exit_code == 0:
        exported = (tmp_path / "export").read_text().splitlines()
        assert "outputs/model/run with space/config.json" in exported
        assert "7630" in exported
