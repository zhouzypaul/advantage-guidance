"""Build SLURM submission scripts and local batch runners for experiment sweeps."""

from __future__ import annotations

import os
import shlex
from functools import partial

# Characters allowed unquoted in --flag=value arguments.
_SAFE_FLAG_CHARS = set("._:/%+-=@")


def _shell_double_quote(s: str) -> str:
    """Wrap *s* in double quotes for bash, escaping characters special inside them."""
    escaped = (
        s.replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("$", "\\$")
        .replace("`", "\\`")
    )
    return f'"{escaped}"'


def _format_flag(key: str, value) -> str:
    """Return ``--key=value``, shell-quoting the value when it contains unsafe characters."""
    s = str(value)
    if any(not c.isalnum() and c not in _SAFE_FLAG_CHARS for c in s):
        return f"--{key}={_shell_double_quote(s)}"
    return f"--{key}={s}"


def _flags_from_args(args: dict) -> list[str]:
    # ml_collections requires config-file flags before their field overrides.
    return [
        _format_flag(key, args[key]) for key in sorted(args, key=lambda k: k.count("."))
    ]


def _flag_value_for_sort(command: str, sort_by: str):
    """Extract ``--sort_by=...`` from a command for stable numeric/string sorting."""
    prefix = f"--{sort_by}="
    for part in command.split():
        if not part.startswith(prefix):
            continue
        value = part.split("=", 1)[1]
        try:
            return int(value)
        except ValueError:
            return value
    return ""


def _bash_commands_array(commands: list[str]) -> str:
    """Format commands as a 1-indexed bash array literal."""
    lines = (
        f"[{i}]={shlex.quote(command)}" for i, command in enumerate(commands, start=1)
    )
    return "\n  ".join(lines)


def _slurm_array_directive(num_array_tasks: int, concurrency_limit: int | None) -> str:
    base = f"#SBATCH --array=1-{num_array_tasks}"
    if concurrency_limit is None:
        return base
    return f"{base}%{concurrency_limit}"


_RUNNER_HEADER = """\
#!/usr/bin/env bash
set -euo pipefail

"""


class SbatchGenerator:
    """Collect CLI commands and render them as a SLURM array job or local runner script."""

    def __init__(
        self,
        prefix=("MUJOCO_GL=egl", "python main.py"),
        j=1,
        limit=None,
        ram_gb=24,
        job_name="advantage-guidance",
        time="4:00:00",
        log_dir=None,
        gres="gpu:1",
    ):
        self.prefix = list(prefix)
        self.commands: list[str] = []
        self.j = j  # commands executed in parallel per array task
        self.limit = limit  # max concurrent array tasks (% limit in SLURM)
        self.ram_gb = ram_gb
        self.job_name = job_name
        self.time = time
        self.gres = gres
        if log_dir is None:
            log_dir = os.environ.get("SLURM_LOG_DIR", "~/logs")
        self.log_dir = os.path.expanduser(log_dir)

    def add_common_prefix(self, args: dict) -> None:
        """Append flags shared by every subsequent ``add_run`` call."""
        self.prefix.extend(_flags_from_args(args))

    def add_run(self, args: dict) -> None:
        """Register one experiment command (common prefix + run-specific flags)."""
        self.commands.append(" ".join([*self.prefix, *_flags_from_args(args)]))

    def _sort_commands(self, sort_by: str | None) -> None:
        if sort_by is not None:
            self.commands.sort(key=partial(_flag_value_for_sort, sort_by=sort_by))

    def generate_str(self, sort_by=None, print_commands=False) -> str:
        """Return a bash script that writes and submits a SLURM array job."""
        self._sort_commands(sort_by)

        num_commands = len(self.commands)
        num_array_tasks = (num_commands - 1) // self.j + 1

        if print_commands:
            print("\n".join(self.commands))

        worker_script = f"""\
#!/bin/bash
#SBATCH --job-name={self.job_name}
#SBATCH --open-mode=append
#SBATCH -o {self.log_dir}/%A_%a.out
#SBATCH -e {self.log_dir}/%A_%a.err
#SBATCH --time={self.time}
#SBATCH --mem={self.ram_gb}G
#SBATCH --gres={self.gres}
#SBATCH --requeue
{_slurm_array_directive(num_array_tasks, self.limit)}

TASK_ID=$((SLURM_ARRAY_TASK_ID-1))
PARALLEL_N={self.j}
JOB_N={num_commands}

JOB_OFFSET=${{JOB_OFFSET:-0}}
COM_ID_S=$((JOB_OFFSET + TASK_ID * PARALLEL_N + 1))

declare -a commands=(
  {_bash_commands_array(self.commands)}
)

parallel --delay 20 --linebuffer -j {self.j} {{1}} ::: "${{commands[@]:$COM_ID_S:$PARALLEL_N}}"
"""

        print(f"Created {num_array_tasks} jobs")

        return "\n".join(
            [
                "#!/usr/bin/env bash",
                "set -euo pipefail",
                "",
                "# Run with: bash this_script.sh",
                "worker_file=$(mktemp)",
                "cat > \"$worker_file\" <<'SBATCH_WORKER'",
                worker_script,
                "SBATCH_WORKER",
                'sbatch "$worker_file"',
                'echo "Submitted. Worker script kept at: $worker_file"',
            ]
        )

    def generate_local_str(self, sort_by=None) -> str:
        """Return a bash script that runs all commands locally."""
        self._sort_commands(sort_by)
        return _RUNNER_HEADER + "\n".join(self.commands)
