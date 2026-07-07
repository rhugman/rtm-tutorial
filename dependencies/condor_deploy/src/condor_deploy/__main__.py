"""
condor_deploy.__main__
======================
CLI entry point — enables ``python -m condor_deploy`` and the
``condor-deploy`` console script installed by pyproject.toml.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "HTCondor deployment for pestpp-ies panther runs.\n\n"
            "Two modes:\n"
            "  (default)   Package template + submit workers + start master\n"
            "  --build-env Build and pack a minimal worker conda environment"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # ---- env build options ----
    p.add_argument("--build-env", action="store_true", dest="build_env",
                   help="Build and conda-pack the worker env tar.gz, then exit.")
    p.add_argument("--env-name", default="_worker", dest="env_name",
                   help="Conda env name to create for the worker (default: _worker).")
    p.add_argument("--python-version", default=None, dest="python_version",
                   help="Python version for the worker env "
                        "(default: from config, 3.13).")
    p.add_argument("--no-mamba", action="store_true", dest="no_mamba",
                   help="Use conda instead of mamba to create the worker env.")
    p.add_argument("--force-recreate", action="store_true", dest="force_recreate",
                   help="Delete and recreate the worker conda env even if it exists.")
    # ---- submit options ----
    p.add_argument("--template", default=None, dest="template",
                   help="Template directory containing pest.pst.")
    p.add_argument("--env-zip", default=None, dest="env_zip",
                   help="Path to conda-pack environment tar.gz. "
                        "If omitted, a worker env is built automatically.")
    p.add_argument("--master-dir", default=None, dest="master_dir",
                   help="Override master directory (default: derived from template name).")
    p.add_argument("--pst", default="pest.pst", dest="pst_name",
                   help="PST control file name (default: pest.pst).")
    p.add_argument("--pestpp-exe", default="pestpp-ies", dest="pestpp_exe",
                   help="pestpp executable name (default: pestpp-ies).")
    p.add_argument("--n-workers", type=int, default=50, dest="n_workers",
                   help="Number of HTCondor worker jobs (default: 50).")
    p.add_argument("--port", type=int, default=4004, dest="master_port",
                   help="TCP port for pestpp panther master (default: 4004).")
    p.add_argument("--host", default=None, dest="master_host",
                   help="Hostname workers connect to (default: $HOSTNAME).")
    p.add_argument("--cpus", type=int, default=1, dest="cpus_per_worker",
                   help="CPUs per worker slot (default: 1).")
    p.add_argument("--memory-mb", type=int, default=2048, dest="memory_mb",
                   help="RAM per worker slot in MB (default: 2048).")
    p.add_argument("--disk-mb", type=int, default=5120, dest="disk_mb",
                   help="Disk per worker slot in MB (default: 5120).")
    p.add_argument("--requirements", default=None, dest="condor_requirements",
                   help="HTCondor Requirements ClassAd expression.")
    p.add_argument("--timeout-h", type=float, default=168.0, dest="timeout_h",
                   help="Max hours to wait for master (default: 168).")
    p.add_argument("--run-tag", default=None, dest="run_tag",
                   help="Short label for this run (auto-timestamped if omitted).")
    p.add_argument("--dry-run", action="store_true", dest="dry_run",
                   help="Write files but do not start master or submit jobs.")
    p.add_argument("--block", action="store_true", dest="block",
                   help="Block until the master process exits (default: return immediately).")
    p.add_argument("--force-local", action="store_true", dest="force_local",
                   help="Skip HTCondor entirely and use pyemu.os_utils.start_workers locally.")
    p.add_argument("--local-workers", type=int, default=None, dest="local_workers",
                   help="Number of local workers when --force-local is used. "
                        "Defaults to --n-workers.")
    return p


def main(argv: Optional[List[str]] = None) -> None:
    args = _build_parser().parse_args(argv)

    # ---- build-env subcommand ----
    if args.build_env:
        if not args.env_zip:
            print("ERROR: --env-zip is required with --build-env", file=sys.stderr)
            sys.exit(1)
        from .env_builder import build_worker_env_zip
        build_worker_env_zip(
            output_zip=args.env_zip,
            env_name=args.env_name,
            python_version=args.python_version,
            mamba=not args.no_mamba,
            force_recreate=args.force_recreate,
        )
        return

    # ---- submit subcommand ----
    if not args.template:
        print("ERROR: --template is required when not using --build-env", file=sys.stderr)
        sys.exit(1)
    template_dir = Path(args.template).resolve()
    if not template_dir.is_dir():
        print(f"ERROR: template directory not found: {template_dir}", file=sys.stderr)
        sys.exit(1)

    # ---- force-local mode ----
    if args.force_local:
        from .local import run_local
        run_local(
            template_dir=template_dir,
            pst_name=args.pst_name,
            master_dir=args.master_dir,
            n_workers=args.local_workers if args.local_workers is not None else args.n_workers,
            master_port=args.master_port,
            pestpp_exe=args.pestpp_exe,
        )
        sys.exit(0)

    from .pool import CondorWorkerPool
    pool = CondorWorkerPool(
        template_dir=template_dir,
        env_zip=args.env_zip,
        master_dir=args.master_dir,
        pst_name=args.pst_name,
        n_workers=args.n_workers,
        master_port=args.master_port,
        master_host=args.master_host,
        pestpp_exe=args.pestpp_exe,
        cpus_per_worker=args.cpus_per_worker,
        memory_mb=args.memory_mb,
        disk_mb=args.disk_mb,
        condor_requirements=args.condor_requirements,
        run_tag=args.run_tag,
        timeout_h=args.timeout_h,
        dry_run=args.dry_run,
    )
    pool.submit_and_wait(block=args.block)
    sys.exit(0)


if __name__ == "__main__":
    main()

