"""
condor_deploy.entry_points
==========================
High-level entry points called by your workflow or the CLI.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

log = logging.getLogger("condor_deploy")


def submit_condor_workers_from_env(
    template_dir: "str | Path",
    pst_name: str,
    master_dir: "str | Path | None" = None,
    **condor_kwargs,
) -> "subprocess.Popen | None":
    """
    Entry point called by ``run_pestpp()`` in workflow.py when
    ``condor_submit`` is found on PATH.

    Parameters
    ----------
    template_dir : str | Path
        The clean PST template directory.
    pst_name : str
        PST control file name.
    master_dir : str | Path | None
        Where the master process runs.
    **condor_kwargs
        Forwarded to :class:`~condor_deploy.pool.CondorWorkerPool`.

        Special keys consumed here (not forwarded):

        ``force_local`` : bool
            Bypass HTCondor entirely and use
            :func:`~condor_deploy.local.run_local` (handy for debugging).
        ``env_zip`` : str | Path | None
            Path to conda-pack archive; ``None`` triggers auto-build.
    """
    force_local = condor_kwargs.pop("force_local", False)
    if force_local:
        from .local import run_local
        log.info("force_local=True — skipping HTCondor, using local workers.")
        run_local(
            template_dir=template_dir,
            pst_name=pst_name,
            master_dir=master_dir,
            n_workers=condor_kwargs.get("n_workers", 4),
            master_port=condor_kwargs.get("master_port", 4004),
            pestpp_exe=condor_kwargs.get("pestpp_exe", "pestpp-ies"),
        )
        return None

    from .pool import CondorWorkerPool
    env_zip = condor_kwargs.pop("env_zip", None)
    pool = CondorWorkerPool(
        template_dir=template_dir,
        env_zip=env_zip,
        master_dir=master_dir,
        pst_name=pst_name,
        **condor_kwargs,
    )
    return pool.submit_and_wait(block=True)

