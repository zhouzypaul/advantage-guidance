import hashlib
import os
import pathlib
import subprocess
import tempfile
from datetime import datetime

import absl.flags as flags
import ml_collections
import numpy as np
import wandb
from PIL import Image, ImageEnhance

WANDB_TAG_MAX_LENGTH = 64


def _wandb_group_tag(group):
    """Return a valid, deterministic W&B tag for a run group."""
    if len(group) <= WANDB_TAG_MAX_LENGTH:
        return group

    digest = hashlib.sha256(group.encode()).hexdigest()[:8]
    prefix_length = WANDB_TAG_MAX_LENGTH - len(digest) - 1
    return f"{group[:prefix_length]}-{digest}"


class CsvLogger:
    """CSV logger for logging metrics to a CSV file."""

    def __init__(self, path):
        self.path = path
        self.header = None
        self.file = None
        self.disallowed_types = (wandb.Image, wandb.Video, wandb.Histogram)

    def log(self, row, step):
        row["step"] = step
        # Per-sample arrays can contain newlines and must not become CSV cells.
        filtered_row = {
            k: v
            for k, v in row.items()
            if not isinstance(v, self.disallowed_types) and np.ndim(v) == 0
        }
        if self.file is None:
            self.file = open(self.path, "w")
            if self.header is None:
                self.header = list(filtered_row)
                self.file.write(",".join(self.header) + "\n")
        self.file.write(
            ",".join([str(filtered_row.get(k, "")) for k in self.header]) + "\n"
        )
        self.file.flush()

    def close(self):
        if self.file is not None:
            self.file.close()


def get_exp_name(seed):
    """Return the experiment name."""
    exp_name = ""
    exp_name += f"sd{seed:03d}_"
    if "SLURM_JOB_ID" in os.environ:
        exp_name += f"s_{os.environ['SLURM_JOB_ID']}."
    if "SLURM_PROCID" in os.environ:
        exp_name += f"{os.environ['SLURM_PROCID']}."
    exp_name += f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    return exp_name


def get_flag_dict():
    """Return the dictionary of flags."""
    flag_dict = {k: getattr(flags.FLAGS, k) for k in flags.FLAGS if "." not in k}
    for k in flag_dict:
        if isinstance(flag_dict[k], ml_collections.ConfigDict):
            flag_dict[k] = flag_dict[k].to_dict()
    return flag_dict


def setup_wandb(
    entity=None,
    project="project",
    group=None,
    name=None,
    mode="online",
    hyperparam_dict=None,
    log_code=False,
    tags=None,
    resume_id=None,
    resume_mode=None,
):
    """Set up Weights & Biases for logging."""
    wandb_output_dir = tempfile.mkdtemp()
    if tags is None:
        tags = [_wandb_group_tag(group)] if group is not None else None

    # Combine flag dict with hyperparameters (similar to utils/wandb.py)
    config_dict = get_flag_dict()
    if hyperparam_dict is not None:
        config_dict.update(hyperparam_dict)

    init_kwargs = dict(
        config=config_dict,
        project=project,
        entity=entity,
        tags=tags,
        group=group,
        dir=wandb_output_dir,
        name=name,
        settings=wandb.Settings(
            start_method="thread",
            _disable_stats=False,
        ),
        mode=mode,
        save_code=log_code,
    )
    if resume_id is not None:
        init_kwargs.update(dict(id=resume_id, resume=resume_mode or "allow"))

    run = wandb.init(**init_kwargs)

    if log_code:
        run.log_code(
            root=".",
            include_fn=lambda p: (
                not any(
                    part in {".git", "__pycache__", ".venv", "exp", ".onager"}
                    for part in pathlib.Path(p).parts
                )
            ),
        )

        try:
            sha = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip()
            diff = subprocess.check_output(["git", "diff"], text=True)
        except Exception:
            sha, diff = "no-git", ""
        wandb.config.update(
            {"git_commit": sha, "git_diff": diff}, allow_val_change=True
        )

    return run


def reshape_video(v, n_cols=None):
    """Helper function to reshape videos."""
    if v.ndim == 4:
        v = v[None,]

    _, t, h, w, c = v.shape

    if n_cols is None:
        # Set n_cols to the square root of the number of videos.
        n_cols = np.ceil(np.sqrt(v.shape[0])).astype(int)
    if v.shape[0] % n_cols != 0:
        len_addition = n_cols - v.shape[0] % n_cols
        v = np.concatenate((v, np.zeros(shape=(len_addition, t, h, w, c))), axis=0)
    n_rows = v.shape[0] // n_cols

    v = np.reshape(v, newshape=(n_rows, n_cols, t, h, w, c))
    v = np.transpose(v, axes=(2, 5, 0, 3, 1, 4))
    v = np.reshape(v, newshape=(t, c, n_rows * h, n_cols * w))

    return v


def get_wandb_video(renders=None, n_cols=None, fps=15):
    """Return a Weights & Biases video.

    It takes a list of videos and reshapes them into a single video with the specified number of columns.

    Args:
        renders: List of videos. Each video should be a numpy array of shape (t, h, w, c).
        n_cols: Number of columns for the reshaped video. If None, it is set to the square root of the number of videos.
    """
    # Pad videos to the same length.
    max_length = max([len(render) for render in renders])
    for i, render in enumerate(renders):
        assert render.dtype == np.uint8

        # Decrease brightness of the padded frames.
        final_frame = render[-1]
        final_image = Image.fromarray(final_frame)
        enhancer = ImageEnhance.Brightness(final_image)
        final_image = enhancer.enhance(0.5)
        final_frame = np.array(final_image)

        pad = np.repeat(final_frame[np.newaxis, ...], max_length - len(render), axis=0)
        renders[i] = np.concatenate([render, pad], axis=0)

        # Add borders.
        renders[i] = np.pad(
            renders[i],
            ((0, 0), (1, 1), (1, 1), (0, 0)),
            mode="constant",
            constant_values=0,
        )
    renders = np.array(renders)  # (n, t, h, w, c)

    renders = reshape_video(renders, n_cols)  # (t, c, nr * h, nc * w)

    return wandb.Video(renders, fps=fps, format="mp4")
