"""
condor_deploy.pool
==================
:class:`CondorWorkerPool` — orchestrates zipping the template,
writing HTCondor submit files, starting the pestpp-ies master, and
submitting panther worker jobs to the HTCondor pool.
"""

from __future__ import annotations

import logging
import os
import signal
import subprocess
import time
import zipfile
from pathlib import Path
from typing import List, Optional

from .config import get_config
from ._utils import _utcnow_tag, _utcnow_iso

log = logging.getLogger("condor_deploy")


class CondorWorkerPool:
    """
    Zip the template directory, write HTCondor submit files, create the
    master directory, start the pestpp-ies master there as a local
    subprocess, and submit N panther agent jobs to the HTCondor pool.

    Mirrors the local ``pyemu.os_utils.start_workers`` pattern:

    * **template_dir** — the clean PST template; zipped and transferred
      to every worker slot so each gets an identical, independent copy.
    * **master_dir** — a separate copy of the template where the
      pestpp-ies master process runs and accumulates outputs.

    Designed to be called from within ``run_pestpp()`` in workflow.py
    when ``condor_submit`` is available on PATH, or directly from the CLI.

    Parameters
    ----------
    template_dir : str | Path
        The clean template directory containing ``pest.pst`` and all
        model files.  This is zipped for worker transfer AND copied to
        form the master directory.
    env_zip : str | Path | None
        Path to the conda-pack worker environment archive (tar.gz).
        If ``None`` (the default), :func:`~condor_deploy.env_builder.build_worker_env_zip`
        is called automatically to create ``<template_dir.parent>/worker_env.tar.gz``.
        Pass an explicit path to reuse a previously built archive and skip
        the (slow) environment build step.
    master_dir : str | Path | None
        Where the pestpp-ies master process runs.  Created as a fresh
        copy of ``template_dir`` at submit time.  Defaults to a sibling
        directory derived from the template name.
    pst_name : str
        Name of the PST control file (e.g. ``"pest.pst"``).
    n_workers : int
        Number of HTCondor worker jobs to submit.
    master_port : int
        TCP port the pestpp-ies master listens on for panther connections.
    master_host : str | None
        Hostname workers use to connect to the master.
    pestpp_exe : str
        Name (or relative path inside the unpacked zip) of the pestpp-ies
        executable.
    cpus_per_worker : int
        CPUs requested per HTCondor slot.
    memory_mb : int
        RAM (MB) requested per HTCondor slot.
    disk_mb : int
        Disk (MB) requested per HTCondor slot.
    condor_requirements : str | None
        Raw HTCondor Requirements ClassAd expression for this run.  Overrides
        the project-wide :attr:`~condor_deploy.config.CondorDeployConfig.platform_requirements`
        default.  ``None`` (the default) falls back to that config value; if it
        too is empty, no ``requirements`` line is written and the pool defaults apply.
    run_tag : str | None
        Short label used in filenames / log headers (auto-timestamped if None).
    zip_exclude : list[str] | None
        Glob patterns for files to omit from the worker zip.
        Defaults to :data:`~condor_deploy.config.CondorDeployConfig.zip_exclude`
        from the current module config (see :func:`~condor_deploy.config.configure`).
    timeout_h : float
        Maximum hours to wait for the master to finish.
    poll_interval_s : int
        Seconds between status polls in the wait loop.
    dry_run : bool
        If True, write submit files and print commands without starting
        the master or calling ``condor_submit``.
    """

    def __init__(
            self,
            template_dir: "str | Path",
            env_zip: "str | Path | None" = None,
            master_dir: "str | Path | None" = None,
            pst_name: str = "pest.pst",
            n_workers: int = 50,
            master_port: int = 4004,
            master_host: Optional[str] = None,
            pestpp_exe: str = "pestpp-ies",
            cpus_per_worker: int = 1,
            memory_mb: int = 2048,
            disk_mb: int = 5120,
            condor_requirements: Optional[str] = None,
            run_tag: Optional[str] = None,
            zip_exclude: Optional[List[str]] = None,
            timeout_h: float = 168.0,
            poll_interval_s: int = 10,
            dry_run: bool = False,
    ):
        self.template_dir = Path(template_dir).resolve()
        if env_zip is None:
            from .env_builder import build_worker_env_zip
            _default_zip = self.template_dir.parent / "worker_env.tar.gz"
            log.info("env_zip not provided — building worker env → %s", _default_zip)
            env_zip = build_worker_env_zip(_default_zip, force_recreate=True)
        self.env_zip = Path(env_zip).resolve()
        if not self.env_zip.exists():
            raise FileNotFoundError(f"env_zip not found: {self.env_zip}")
        self.master_dir = (
            Path(master_dir).resolve() if master_dir is not None
            else self.template_dir.parent / f"{self.template_dir.name.replace('template', 'master')}"
        )
        self.pst_name = pst_name
        self.n_workers = n_workers
        self.master_port = master_port
        import socket as _socket
        self.master_host = (master_host
                            or os.environ.get("HOSTNAME")
                            or _socket.gethostname()
                            or "localhost")
        self.pestpp_exe = pestpp_exe
        self.cpus_per_worker = cpus_per_worker
        self.memory_mb = memory_mb
        self.disk_mb = disk_mb
        self.condor_requirements = condor_requirements
        self.run_tag = run_tag or _utcnow_tag()
        # Resolve zip_exclude at instantiation time from the live config singleton
        self.zip_exclude: List[str] = (
            zip_exclude if zip_exclude is not None
            else list(get_config().zip_exclude)
        )
        self.timeout_h = timeout_h
        self.poll_interval_s = poll_interval_s
        self.dry_run = dry_run

        self.condor_cluster_id: Optional[str] = None
        self.master_proc: Optional[subprocess.Popen] = None

        self._staging = self.template_dir.parent / f"_condor_staging_{self.run_tag}"
        log.info("CondorWorkerPool | template=%s | master=%s | run_tag=%s | n_workers=%d",
                 self.template_dir.name, self.master_dir.name, self.run_tag, self.n_workers)

    # ------------------------------------------------------------------
    # Step 1: zip the template directory (sent to every worker)
    # ------------------------------------------------------------------

    def make_zip(self) -> Path:
        """Zip ``template_dir`` → ``<staging>/<template_dir.name>.zip``."""
        self._staging.mkdir(parents=True, exist_ok=True)
        zip_path = self._staging / f"{self.template_dir.name}.zip"
        log.info("Zipping template %s → %s", self.template_dir, zip_path)

        def _excluded(rel: Path) -> bool:
            for pat in self.zip_exclude:
                if Path(rel.name).match(pat):
                    return True
                for part in rel.parts:
                    if Path(part).match(pat):
                        return True
            return False

        n_files = 0
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED,
                             compresslevel=1) as zf:
            for root, dirs, files in os.walk(self.template_dir):
                dirs[:] = [
                    d for d in dirs
                    if not any(Path(d).match(p) for p in self.zip_exclude)
                    and d not in ("__pycache__", "figures", ".git")
                ]
                for fname in files:
                    fp = Path(root) / fname
                    rel = fp.relative_to(self.template_dir)
                    if _excluded(rel):
                        continue
                    zf.write(fp, arcname=str(rel))
                    n_files += 1

        size_mb = zip_path.stat().st_size / 1_048_576
        log.info("Template zip: %d files, %.1f MB → %s", n_files, size_mb, zip_path)

        # Bundle editable packages from config into _editable_pkgs/<pkg_name>/
        editable_paths: List[Path] = [Path(str(p)).resolve() for p in get_config().worker_pip_editable]
        if editable_paths:
            with zipfile.ZipFile(zip_path, "a", compression=zipfile.ZIP_DEFLATED,
                                 compresslevel=1) as zf:
                for src in editable_paths:
                    if not src.exists():
                        log.warning("Editable package path not found, skipping: %s", src)
                        continue
                    arc_base = Path("_editable_pkgs") / src.name
                    log.info("Bundling editable package %s → %s", src, arc_base)
                    for root, dirs, files in os.walk(src):
                        dirs[:] = [d for d in dirs if d not in ("__pycache__", ".git")]
                        for fname in files:
                            fp = Path(root) / fname
                            rel = fp.relative_to(src)
                            zf.write(fp, arcname=str(arc_base / rel))

        return zip_path

    # ------------------------------------------------------------------
    # Step 2: create the master directory (fresh copy of template)
    # ------------------------------------------------------------------

    def make_master_dir(self) -> Path:
        """Copy ``template_dir`` → ``master_dir``."""
        import shutil
        if self.master_dir.exists():
            shutil.rmtree(self.master_dir)
        shutil.copytree(self.template_dir, self.master_dir)
        log.info("Master dir: %s → %s", self.template_dir.name, self.master_dir)
        return self.master_dir

    # ------------------------------------------------------------------
    # Step 3: write worker.sh and condor_workers.sub
    # ------------------------------------------------------------------

    def write_submit_files(self, template_zip: Path) -> Path:
        """Write ``worker.sh`` and ``condor_workers.sub`` into the staging dir."""
        cfg = get_config()
        sd = self._staging
        sd.mkdir(parents=True, exist_ok=True)
        (sd / "log").mkdir(exist_ok=True)

        sh = sd / "worker.sh"
        editable_paths: List[Path] = [Path(str(p)).resolve() for p in cfg.worker_pip_editable]
        if editable_paths:
            lines = ["    # pip editable installs from _editable_pkgs/"]
            for src in editable_paths:
                pkg_dir = f"_editable_pkgs/{src.name}"
                lines.append(
                    f'    echo "[worker ${{WID}}] pip install -e {src.name} ..."\n'
                    f'    pip install -e "{pkg_dir}"'
                )
            editable_pkgs_sh = "\n".join(lines)
        else:
            editable_pkgs_sh = "    # (no editable packages)"

        chmod_exes = " ".join(f'"{e}"' for e in cfg.worker_chmod_exes)
        worker_sh_text = (
            cfg.worker_sh
            .replace("{env_zip}", self.env_zip.name)
            .replace("{template_zip}", template_zip.name)
            .replace("{editable_pkgs_sh}", editable_pkgs_sh)
            .replace("{chmod_exes}", chmod_exes)
        )
        sh.write_text(worker_sh_text)
        sh.chmod(0o755)

        transfer_files = f"worker.sh, {template_zip}, {self.env_zip}"

        # Resolve the HTCondor Requirements expression: a per-run
        # condor_requirements overrides the project-wide config default.
        # An empty expression omits the line entirely so pool defaults apply.
        req_expr = (self.condor_requirements
                    if self.condor_requirements is not None
                    else cfg.platform_requirements)
        requirements_line = (
            f"requirements            = {req_expr}" if req_expr
            else "# requirements: none set — HTCondor pool defaults apply"
        )

        sub = sd / "condor_workers.sub"
        sub.write_text(cfg.sub_template.format(
            timestamp=_utcnow_iso(),
            run_tag=self.run_tag,
            pestpp_exe=self.pestpp_exe,
            pst_name=self.pst_name,
            master_host=self.master_host,
            master_port=self.master_port,
            template_zip=template_zip,
            env_zip=self.env_zip,
            transfer_files=transfer_files,
            n_workers=self.n_workers,
            cpus_per_worker=self.cpus_per_worker,
            memory_mb=self.memory_mb,
            disk_mb=self.disk_mb,
            requirements_line=requirements_line,
            priority=0,
        ))
        log.info("Wrote worker.sh and condor_workers.sub → %s", sd)
        return sub

    # ------------------------------------------------------------------
    # Step 4: start the pestpp-ies master
    # ------------------------------------------------------------------

    def start_master(self, pipe_to_stdout: bool = False) -> "subprocess.Popen | None":
        """
        Launch pestpp-ies as a master (``/h :<port>``) from ``master_dir``.

        Parameters
        ----------
        pipe_to_stdout : bool
            If True, pestpp-ies stdout/stderr flow directly to the calling
            terminal.  If False (default), output is redirected to
            ``master_dir/master.out``.
        """
        exe = self.master_dir / self.pestpp_exe
        if not exe.exists():
            exe = Path(self.pestpp_exe)

        cmd = [str(exe), self.pst_name, "/h", f":{self.master_port}"]
        log.info("Starting master: %s (cwd=%s, port=%d)",
                 " ".join(cmd), self.master_dir, self.master_port)

        if self.dry_run:
            log.info("[dry-run] would run: %s", " ".join(cmd))
            return None

        if pipe_to_stdout:
            stdout_sink = None
            stderr_sink = None
        else:
            stdout_sink = open(self.master_dir / "master.out", "w")
            stderr_sink = subprocess.STDOUT

        proc = subprocess.Popen(
            cmd,
            cwd=str(self.master_dir),
            stdout=stdout_sink,
            stderr=stderr_sink,
        )
        self.master_proc = proc
        log.info("Master PID: %d", proc.pid)
        return proc

    # ------------------------------------------------------------------
    # Step 5: condor_submit
    # ------------------------------------------------------------------

    def submit(self, sub_path: Path) -> str:
        """Run ``condor_submit`` and return the cluster ID string."""
        if self.dry_run:
            log.info("[dry-run] condor_submit %s", sub_path)
            return "dry-run"

        r = subprocess.run(
            ["condor_submit", str(sub_path)],
            capture_output=True, text=True,
            cwd=str(self._staging),
        )
        if r.returncode != 0:
            # surface condor's actual complaint (check=True would swallow it as a bare exit code)
            log.error("condor_submit failed (exit %d) for %s", r.returncode, sub_path)
            if r.stdout.strip():
                log.error("condor_submit stdout:\n%s", r.stdout.strip())
            if r.stderr.strip():
                log.error("condor_submit stderr:\n%s", r.stderr.strip())
            raise RuntimeError(
                f"condor_submit failed (exit {r.returncode}): "
                f"{(r.stderr.strip() or r.stdout.strip() or 'no output').splitlines()[-1]}\n"
                f"submit file: {sub_path}")
        cluster_id = "unknown"
        for line in r.stdout.splitlines():
            if "cluster" in line.lower():
                cluster_id = line.strip().rstrip(".").split()[-1]
                break
        self.condor_cluster_id = cluster_id
        log.info("Submitted %d jobs → cluster %s", self.n_workers, cluster_id)
        return cluster_id

    # ------------------------------------------------------------------
    # Step 6: wait for master to finish
    # ------------------------------------------------------------------

    def wait(self) -> bool:
        """
        Block until the master process exits.

        Returns True on normal completion, False on timeout.
        """
        if self.dry_run or self.master_proc is None:
            return True

        deadline = time.monotonic() + self.timeout_h * 3600
        log.info("Blocking until master (PID %d) exits (timeout %.1fh) …",
                 self.master_proc.pid, self.timeout_h)
        while time.monotonic() < deadline:
            rc = self.master_proc.poll()
            if rc is not None:
                log.info("Master exited with code %d.", rc)
                return True
            time.sleep(self.poll_interval_s)
        log.warning("Timeout after %.1fh.", self.timeout_h)
        return False

    # ------------------------------------------------------------------
    # Convenience: full sequence
    # ------------------------------------------------------------------

    def submit_and_wait(self, block: bool = False) -> "subprocess.Popen | None":
        """
        Full sequence: zip template → make master dir → write submit files
        → condor_submit → start master.

        Parameters
        ----------
        block : bool
            If True, block until the master exits AND pipe pestpp-ies
            stdout/stderr directly to the terminal — the right choice when
            called from within a workflow script.  Defaults to False
            (return immediately, redirect output to ``master_dir/master.out``).

        Returns
        -------
        subprocess.Popen | None
            The master process handle (None in dry-run mode).
        """
        template_zip = self.make_zip()
        self.make_master_dir()
        sub_path = self.write_submit_files(template_zip)

        self.submit(sub_path)
        self.start_master(pipe_to_stdout=block)

        if self.dry_run:
            return None

        if block:
            log.info(
                "\n  Master PID : %d\n  Cluster    : %s\n  (stdout/stderr → terminal)\n",
                self.master_proc.pid,
                self.condor_cluster_id,
            )
            self.wait()
        else:
            log.info(
                "\n"
                "  Master PID : %d\n"
                "  Master log : %s\n"
                "  Cluster    : %s\n"
                "\n"
                "  Monitor with:\n"
                "    tail -f %s\n"
                "    condor_q %s\n"
                "    condor_status\n",
                self.master_proc.pid,
                self.master_dir / "master.out",
                self.condor_cluster_id,
                self.master_dir / "master.out",
                self.condor_cluster_id,
            )

        return self.master_proc

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def cancel(self) -> None:
        """Cancel the condor cluster and terminate the master process."""
        if self.condor_cluster_id and self.condor_cluster_id not in ("unknown", "dry-run"):
            subprocess.run(["condor_rm", self.condor_cluster_id], check=False)
            log.info("Cancelled condor cluster %s", self.condor_cluster_id)
        if self.master_proc and self.master_proc.poll() is None:
            self.master_proc.send_signal(signal.SIGTERM)
            log.info("Sent SIGTERM to master PID %d", self.master_proc.pid)

