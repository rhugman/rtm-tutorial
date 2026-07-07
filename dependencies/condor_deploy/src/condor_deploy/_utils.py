"""
condor_deploy._utils
====================
Private timestamp helpers shared across submodules.
"""

import datetime


def _utcnow_tag() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%S")


def _utcnow_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")

