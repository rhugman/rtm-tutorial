"""
condor_deploy.templates
=======================
Default HTCondor submit description and worker shell-script templates.

These are the module-level defaults used by :class:`~condor_deploy.config.CondorDeployConfig`.
Override them project-wide via :func:`~condor_deploy.config.configure`.
"""

import textwrap

# HTCondor submit description file template.
SUB_TEMPLATE: str = textwrap.dedent("""\
    # HTCondor submit description — pestpp-ies panther workers
    # generated : {timestamp}
    # run_tag   : {run_tag}

    notification            = never
    universe                = vanilla
    # this will log all the worker stdout and stderr - make sure to mkdir a "./log" dir where ever
    # the condor_submit command is issued
    log                     = log/condor_$(Cluster).log
    output                  = log/worker_$(Cluster)_$(Process).out
    error                   = log/worker_$(Cluster)_$(Process).err
    stream_output           = true
    stream_error            = true
    
    # maybe not required in macmini htcondor environment 
    # run_as_owner            = true 
    # getenv                  = true

    executable              = worker.sh
    arguments               = {pestpp_exe} {pst_name} {master_host} {master_port} $(Process)
    # Ship the zipped template; worker.sh unpacks it into a scratch dir
    transfer_input_files    = {transfer_files}
    should_transfer_files   = YES
    when_to_transfer_output = ON_EXIT_OR_EVICT

    request_memory          = {memory_mb}MB
    request_cpus            = {cpus_per_worker}
    request_disk            = {disk_mb}MB

    max_retries             = 0

    {requirements_line}

    priority                = {priority}

    queue {n_workers}
""")

# Shell wrapper executed on each HTCondor worker slot.
# Placeholders use simple {key} tokens replaced via str.replace() in pool.py
# (not str.format()) so that bash ${VAR} syntax is preserved as-is.
WORKER_SH: str = textwrap.dedent("""\
    #!/usr/bin/env bash
    # worker.sh — pestpp-ies panther agent wrapper
    # Args: <pestpp_exe> <pst_name> <master_host> <master_port> <worker_id>
    # set -eo pipefail

    PESTPP="$1"
    PST="$2"
    MASTER_HOST="$3"
    MASTER_PORT="$4"
    WID="$5"
    ENVZIP="{env_zip}"
    ZIPFILE="{template_zip}"

    echo "[worker ${WID}] host=$(hostname) started=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "[worker ${WID}] PWD=$(pwd)  ENVZIP=${ENVZIP}  ZIPFILE=${ZIPFILE}"

    SCRATCH="worker_${WID}"
    mkdir -p "${SCRATCH}"

    # ---------------------------------------------------------------
    # 1. Unpack the conda-pack environment
    # ---------------------------------------------------------------
    mkdir -p "${SCRATCH}/env"
    echo "[worker ${WID}] unpacking env ..."
    tar -xf "${ENVZIP}" -C "${SCRATCH}/env"
    ls "${SCRATCH}/env"
    export PATH="${PWD}/${SCRATCH}/env/bin:$PATH"
    source ${SCRATCH}/env/bin/activate
    conda-unpack
    which python

    # ---------------------------------------------------------------
    # 2. Unpack the template directory (contains _editable_pkgs/)
    # ---------------------------------------------------------------
    unzip -q "${ZIPFILE}" -d "${SCRATCH}"
    cd "${SCRATCH}"

    # ---------------------------------------------------------------
    # 3. Install editable packages bundled in _editable_pkgs/
    # ---------------------------------------------------------------
    {editable_pkgs_sh}

    # Make executables runnable (zip does not preserve execute permission bits)
    chmod +x "${PESTPP}" {chmod_exes} 2>/dev/null || true

    echo "[worker ${WID}] connecting to ${MASTER_HOST}:${MASTER_PORT}"
    ./"${PESTPP}" "${PST}" /h "${MASTER_HOST}:${MASTER_PORT}" 2>&1
    EXIT=$?

    echo "[worker ${WID}] exit=${EXIT} finished=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    exit ${EXIT}
""")

