"""
condor_deploy.local
===================
Local fallback — run pestpp-ies workers without an HTCondor pool,
using ``pyemu.os_utils.start_workers``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

log = logging.getLogger("condor_deploy")


def run_local(
    template_dir: "str | Path",
    pst_name: str,
    master_dir: "str | Path | None" = None,
    n_workers: int = 4,
    master_port: int = 4004,
    pestpp_exe: str = "pestpp-ies",
    worker_root: "str | Path | None" = None,
) -> None:
    """
    Run pestpp-ies locally using ``pyemu.os_utils.start_workers``.

    A convenience wrapper for testing without an HTCondor pool.
    ``pyemu.os_utils.start_workers`` handles copying the template to the
    master directory internally — no manual ``shutil.copytree`` needed.

    Parameters
    ----------
    template_dir : str | Path
        Clean PST template directory.
    pst_name : str
        PST control file name.
    master_dir : str | Path | None
        Where the master runs.  Defaults to a sibling directory derived
        from the template name.
    n_workers : int
        Number of local worker processes (default: 4).
    master_port : int
        TCP port for the pestpp panther master (default: 4004).
    pestpp_exe : str
        pestpp-ies executable name (default: ``"pestpp-ies"``).
    worker_root : str | Path | None
        Root directory under which per-worker subdirectories are created.
        Defaults to ``_local_workers`` alongside the template directory.
    """
    try:
        import pyemu
    except ImportError as exc:
        raise ImportError("pyemu is required for local worker mode") from exc

    template_dir = Path(template_dir).resolve()
    if master_dir is None:
        master_dir = template_dir.parent / f"{template_dir.name.replace('template', 'master')}"
    master_dir = Path(master_dir).resolve()

    if worker_root is None:
        worker_root = template_dir.parent / "_local_workers"
    worker_root = Path(worker_root).resolve()
    worker_root.mkdir(parents=True, exist_ok=True)

    log.info("[local] Starting %d local workers on port %d …", n_workers, master_port)
    pyemu.os_utils.start_workers(
        str(template_dir),
        pestpp_exe,
        pst_name,
        num_workers=n_workers,
        worker_root=str(worker_root),
        master_dir=str(master_dir),
        port=master_port,
        reuse_master=False,
    )
    log.info("[local] All workers finished.")

