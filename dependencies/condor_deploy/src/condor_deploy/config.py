"""
condor_deploy.config
====================
Project-level configuration for condor_deploy.

A module-level singleton :data:`_config` holds all tunable defaults.
Call :func:`configure` early in your script (before creating any
:class:`~condor_deploy.pool.CondorWorkerPool`) to override them::

    from condor_deploy import configure
    configure(
        zip_exclude=["*.hds", "*.cbc", "my_big_file.dat"],
        worker_env_packages=["pyemu", "flopy", "numpy", "pandas", "conda-pack"],
    )

Alternatively, swap out the submit template or worker shell script for a
custom one::

    configure(sub_template=my_custom_sub_template)
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace as _dc_replace
from typing import List

from .templates import SUB_TEMPLATE, WORKER_SH


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

#: File-pattern defaults for files excluded from the worker zip.
DEFAULT_ZIP_EXCLUDE: List[str] = [
    "*.hds", "*.cbc", "*.grb",
    "*.rec", "*.rmr", "*.log",
    "__pycache__", "*.pyc", ".DS_Store",
    "figures",
]

#: Conda packages installed in the worker environment by default.
WORKER_ENV_PACKAGES: List[str] = [
    "pyarrow",
    "pyemu",
    "flopy",
    "numpy",
    "pandas",
    "geopandas",
    "matplotlib",
    "conda-pack",
]

#: Pip packages installed in the worker environment by default.
WORKER_PIP_PACKAGES: List[str] = []

#: Editable pip installs for the worker environment by default.
WORKER_PIP_EDITABLE: List[str] = []

#: Default HTCondor ``Requirements`` ClassAd expression.  Empty string means
#: no platform restriction — the HTCondor pool's own defaults apply.  Override
#: per-project, e.g. ``'( (OpSys == "MacOS") )'`` or ``'( (OpSys == "LINUX") )'``.
PLATFORM_REQUIREMENTS: str = ""

#: Extra executables inside the template to ``chmod +x`` on each worker slot
#: (besides the pestpp executable, which is always made runnable).  The worker
#: zip does not preserve execute bits, so list any model binaries here,
#: e.g. ``["mf6"]``.
WORKER_CHMOD_EXES: List[str] = []

#: Python version for the auto-built worker conda environment.
WORKER_PYTHON_VERSION: str = "3.13"


# ---------------------------------------------------------------------------
# Config dataclass
# ---------------------------------------------------------------------------

@dataclass
class CondorDeployConfig:
    """
    All project-tunable defaults for condor_deploy.

    Attributes
    ----------
    zip_exclude : list[str]
        Glob patterns for files/dirs to omit from the worker template zip.
    worker_env_packages : list[str]
        Conda packages installed in the auto-built worker environment.
    worker_pip_packages : list[str]
        Pip packages installed in the worker environment after conda solve.
    worker_pip_editable : list[str]
        Editable pip installs (``pip install -e``) for the worker environment.
    platform_requirements : str
        Default HTCondor ``Requirements`` expression (empty = pool defaults).
        A per-run ``condor_requirements`` on :class:`~condor_deploy.pool.CondorWorkerPool`
        overrides it.
    worker_chmod_exes : list[str]
        Extra template executables to ``chmod +x`` on each worker (e.g. ``["mf6"]``).
    worker_python_version : str
        Python version for the auto-built worker environment.
    sub_template : str
        HTCondor submit description template (``str.format``-style).
    worker_sh : str
        Bash worker wrapper script template (``str.format``-style).
    """
    zip_exclude: List[str] = field(default_factory=lambda: list(DEFAULT_ZIP_EXCLUDE))
    worker_env_packages: List[str] = field(default_factory=lambda: list(WORKER_ENV_PACKAGES))
    worker_pip_packages: List[str] = field(default_factory=lambda: list(WORKER_PIP_PACKAGES))
    worker_pip_editable: List[str] = field(default_factory=lambda: list(WORKER_PIP_EDITABLE))
    platform_requirements: str = PLATFORM_REQUIREMENTS
    worker_chmod_exes: List[str] = field(default_factory=lambda: list(WORKER_CHMOD_EXES))
    worker_python_version: str = WORKER_PYTHON_VERSION
    sub_template: str = SUB_TEMPLATE
    worker_sh: str = WORKER_SH


# Module-level singleton.
_config = CondorDeployConfig()


def get_config() -> CondorDeployConfig:
    """Return the current module-level :class:`CondorDeployConfig` singleton."""
    return _config


def configure(**kwargs) -> CondorDeployConfig:
    """
    Update the module-level configuration in-place.

    Accepted keyword arguments mirror the fields of :class:`CondorDeployConfig`:
    ``zip_exclude``, ``worker_env_packages``, ``worker_pip_packages``,
    ``worker_pip_editable``, ``platform_requirements``, ``worker_chmod_exes``,
    ``worker_python_version``, ``sub_template``, ``worker_sh``.

    Returns the updated config object.

    Example
    -------
    ::

        from condor_deploy import configure
        configure(
            zip_exclude=["*.hds", "*.cbc", "big_raster.tif"],
            worker_env_packages=["pyemu", "flopy", "numpy", "pandas", "conda-pack"],
        )
    """
    global _config
    _config = _dc_replace(_config, **kwargs)
    return _config

