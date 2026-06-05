#!/usr/bin/env bash
#
# refresh_dependency.sh -- refresh one vendored dependency under dependencies/
#
# The packages under dependencies/ are source snapshots committed to this repo
# (not submodules, not PyPI releases): see dependencies/PROVENANCE.md. This
# script automates the mechanical part of refreshing one of them to a new
# upstream commit: clone the requested repo+branch, record the exact SHA,
# strip .git so it is a plain source tree, and replace the vendored copy.
#
# It deliberately does NOT update PROVENANCE.md or run an import smoke-test for
# you -- those are judgement steps. It prints the SHA you must record and
# reminds you to do both before committing.
#
# Usage:
#   etc/refresh_dependency.sh <name> <repo-url> <branch>
#
# Example:
#   etc/refresh_dependency.sh pyemu https://github.com/rhugman/pyemu.git feat_dsivc
#
set -euo pipefail

if [[ $# -ne 3 ]]; then
    echo "usage: $0 <name> <repo-url> <branch>" >&2
    echo "  e.g. $0 pyemu https://github.com/rhugman/pyemu.git feat_dsivc" >&2
    exit 2
fi

NAME="$1"
REPO_URL="$2"
BRANCH="$3"

# Resolve the repo root from this script's location (etc/ is one level down).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DEST="${REPO_ROOT}/dependencies/${NAME}"

SCRATCH="$(mktemp -d "${TMPDIR:-/tmp}/refresh_${NAME}.XXXXXX")"
cleanup() { rm -rf "${SCRATCH}"; }
trap cleanup EXIT

echo ">> cloning ${REPO_URL} (branch ${BRANCH}) into scratch ..."
git clone --depth 1 --branch "${BRANCH}" "${REPO_URL}" "${SCRATCH}/${NAME}"

SHA="$(git -C "${SCRATCH}/${NAME}" rev-parse HEAD)"
echo ">> upstream commit: ${SHA}"

echo ">> stripping .git from the snapshot ..."
rm -rf "${SCRATCH}/${NAME}/.git"

if [[ -d "${DEST}" ]]; then
    echo ">> replacing existing vendored tree at dependencies/${NAME} ..."
    rm -rf "${DEST}"
fi
mkdir -p "$(dirname "${DEST}")"
mv "${SCRATCH}/${NAME}" "${DEST}"

echo ""
echo "==================================================================="
echo " Refreshed dependencies/${NAME}"
echo "   upstream : ${REPO_URL}"
echo "   branch   : ${BRANCH}"
echo "   commit   : ${SHA}"
echo "   date     : $(date +%Y-%m-%d)"
echo "-------------------------------------------------------------------"
echo " TODO before you commit:"
echo "   1. Update dependencies/PROVENANCE.md (repo, branch, commit, date)"
echo "      with the SHA above."
echo "   2. Smoke-test the import in the project environment, e.g.:"
echo "        conda activate rtm_gmdsi"
echo "        python -c \"import ${NAME}; print(${NAME}.__file__)\""
echo "      (must resolve under dependencies/). For pyemu also confirm:"
echo "        python -c \"from pyemu.emulators import DSI, DSIVC\""
echo "==================================================================="
