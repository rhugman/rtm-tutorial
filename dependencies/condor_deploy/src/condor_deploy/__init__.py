"""
condor_deploy
=============
HTCondor deployment helper for pestpp-ies panther runs.

Quick start
-----------
Project-specific defaults can be set once at the top of your script::

    from condor_deploy import configure
    configure(
        zip_exclude=["*.hds", "*.cbc", "big_raster.tif"],
        worker_env_packages=["pyemu", "flopy", "numpy", "pandas", "conda-pack"],
        # sub_template=MY_CUSTOM_TEMPLATE,  # override the .sub file
        # worker_sh=MY_CUSTOM_WORKER_SH,    # override worker.sh
    )

Then use as before::

    from condor_deploy import CondorWorkerPool, submit_condor_workers_from_env

Submodules
----------
condor_deploy.config       — CondorDeployConfig dataclass + configure() / get_config()
condor_deploy.templates    — Default SUB_TEMPLATE and WORKER_SH strings
condor_deploy.pool         — CondorWorkerPool class
condor_deploy.env_builder  — build_worker_env_zip()
condor_deploy.local        — run_local()
condor_deploy.entry_points — submit_condor_workers_from_env()
condor_deploy.__main__     — CLI (python -m condor_deploy)
"""

import logging as _logging

_log = _logging.getLogger("condor_deploy")
if not _log.handlers:
    _h = _logging.StreamHandler()
    _h.setFormatter(_logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S"))
    _log.addHandler(_h)
    _log.setLevel(_logging.INFO)

# ---------------------------------------------------------------------------
# Config API — always available
# ---------------------------------------------------------------------------
from .config import (                          # noqa: E402
    CondorDeployConfig,
    configure,
    get_config,
    DEFAULT_ZIP_EXCLUDE,       # module-level constant (snapshot of the default)
    WORKER_ENV_PACKAGES,       # module-level constant (snapshot of the default)
)

# ---------------------------------------------------------------------------
# Core classes / functions — lazy-safe direct imports
# ---------------------------------------------------------------------------
from .pool import CondorWorkerPool             # noqa: E402
from .env_builder import build_worker_env_zip  # noqa: E402
from .local import run_local                   # noqa: E402
from .entry_points import (                    # noqa: E402
    submit_condor_workers_from_env,
)

# ---------------------------------------------------------------------------
# __all__ — controls ``from condor_deploy import *``
# ---------------------------------------------------------------------------
__all__ = [
    # config
    "CondorDeployConfig",
    "configure",
    "get_config",
    "DEFAULT_ZIP_EXCLUDE",
    "WORKER_ENV_PACKAGES",
    # core
    "CondorWorkerPool",
    "build_worker_env_zip",
    "run_local",
    "submit_condor_workers_from_env",
]

