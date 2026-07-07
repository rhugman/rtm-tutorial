"""
condor_deploy.env_builder
=========================
Build a minimal conda-pack worker environment for HTCondor slots.
"""

from __future__ import annotations

import json as _json
import logging
import subprocess
from pathlib import Path
from typing import List, Optional

from .config import get_config

log = logging.getLogger("condor_deploy")


def build_worker_env_zip(
    output_zip: "str | Path",
    env_name: str = "_worker",
    extra_packages: Optional[List[str]] = None,
    base_packages: Optional[List[str]] = None,
    python_version: Optional[str] = None,
    mamba: bool = True,
    channels: Optional[List[str]] = None,
    force_recreate: bool = False,
    pip_packages: Optional[List[str]] = None,
    pip_editable: Optional[List[str]] = None,
) -> Path:
    """
    Create a minimal conda environment for HTCondor workers and pack it
    with ``conda-pack`` into a relocatable tar.gz archive.

    The default package list comes from
    :attr:`~condor_deploy.config.CondorDeployConfig.worker_env_packages`
    (the live module config — override via :func:`~condor_deploy.config.configure`).

    Parameters
    ----------
    output_zip : str | Path
        Destination path for the tar.gz archive.
    env_name : str
        Conda environment name to create / reuse (default: ``"_worker"``).
    extra_packages : list[str] | None
        Extra packages appended to the default list.
    base_packages : list[str] | None
        Replace the default package list entirely (advanced use).
    python_version : str | None
        Python version for the new environment.  ``None`` (the default) resolves
        to :attr:`~condor_deploy.config.CondorDeployConfig.worker_python_version`.
    mamba : bool
        Prefer ``mamba`` over ``conda`` for faster solving.
    channels : list[str] | None
        Conda channels.  Defaults to ``["conda-forge"]``.
    force_recreate : bool
        Delete and recreate the env even if it already exists.
    pip_packages : list[str] | None
        Additional packages to install via ``pip`` after the conda env is
        created/reused.  Installed with ``conda run -n env_name pip install``.
    pip_editable : list[str] | None
        Paths (or ``vcs+url`` specs) to install in editable mode via
        ``pip install -e``.  Installed with
        ``conda run -n env_name pip install -e <path>``.

    Returns
    -------
    Path
        Absolute path to the created tar.gz archive.
    """
    output_zip = Path(output_zip).resolve()
    channels = channels or ["conda-forge"]
    cfg = get_config()
    python_version = python_version or cfg.worker_python_version
    packages = list(base_packages or cfg.worker_env_packages)
    if extra_packages:
        packages = packages + [p for p in extra_packages if p not in packages]
    resolved_pip_packages = pip_packages if pip_packages is not None else list(cfg.worker_pip_packages)
    resolved_pip_editable = pip_editable if pip_editable is not None else list(cfg.worker_pip_editable)

    solver = "mamba"
    if mamba:
        r = subprocess.run(["which", "mamba"], capture_output=True)
        if r.returncode != 0:
            log.warning("mamba not found, falling back to conda")
            solver = "conda"
    else:
        solver = "conda"
    # Check system prerequisites up-front with clear messages
    for tool in [solver]:
        if subprocess.run(["which", tool], capture_output=True).returncode != 0:
            raise RuntimeError(
                f"'{tool}' not found on PATH — mamba or conda must be installed on the submit node. "
                "See https://github.com/conda-forge/miniforge"
            )
    if subprocess.run(["which", "conda-pack"], capture_output=True).returncode != 0:
        raise RuntimeError(
            "conda-pack not found on PATH. Install it with:\n"
            "  conda install conda-pack\n"
            "or: pip install 'condor-deploy[build-env]'"
        )

    channel_args: List[str] = []
    for ch in channels:
        channel_args += ["-c", ch]

    env_list = subprocess.run(
        ["conda", "env", "list", "--json"],
        capture_output=True, text=True,
    )
    existing = [Path(p).name for p in
                _json.loads(env_list.stdout).get("envs", [])]

    if force_recreate and env_name in existing:
        log.info("Removing existing env '%s' …", env_name)
        subprocess.run(["conda", "env", "remove", "-n", env_name, "-y"], check=True)
        existing = []

    if env_name not in existing:
        log.info("Creating env '%s' (python=%s) with: %s",
                 env_name, python_version, ", ".join(packages))
        cmd = (
            [solver, "create", "-n", env_name, "-y", f"python={python_version}"]
            + channel_args
            + packages
        )
        log.info("Running: %s", " ".join(cmd))
        subprocess.run(cmd, check=True)
        if resolved_pip_packages:
            log.info("pip install into '%s': %s", env_name, ", ".join(resolved_pip_packages))
            subprocess.run(
                ["conda", "run", "-n", env_name, "pip", "install"] + resolved_pip_packages,
                check=True,
            )

        # Note: pip_editable installs are intentionally NOT added to the conda env here.
        # conda-pack cannot reliably pack editable installs.  Instead, the source
        # directories are bundled into the worker zip and installed via pip install -e
        # inside worker.sh at job start-up.  See pool.CondorWorkerPool.make_zip().

    else:
        log.info("Reusing existing env '%s'. Pass force_recreate=True to rebuild.", env_name)

    output_zip.parent.mkdir(parents=True, exist_ok=True)
    if output_zip.exists():
        output_zip.unlink()

    log.info("Packing env '%s' → %s …", env_name, output_zip)
    subprocess.run(
        ["conda-pack", "-n", env_name, "-o", str(output_zip), "--ignore-missing-files"],
        check=True,
    )
    size_mb = output_zip.stat().st_size / 1_048_576
    log.info("Worker env tar.gz ready: %.1f MB → %s", size_mb, output_zip)
    return output_zip

