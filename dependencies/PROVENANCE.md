# Vendored dependency provenance

The four packages under `dependencies/` are **source snapshots committed to this repository**, not git submodules and not PyPI releases. `environment.yml` pip-installs them editable (`-e ./dependencies/<pkg>`). The series depends on development-branch features — notably the `pyemu.emulators` DSI/DSIVC/DSIAE machinery and `mf6rtm` reactive-transport coupling — so do not substitute released versions.

Each snapshot has had its `.git` directory stripped, so upstream provenance is recorded here rather than recoverable from the tree.

## Snapshots

| Package | Upstream | Branch | Commit | Refreshed |
|---|---|---|---|---|
| `pyemu` | `rhugman/pyemu` | `feat_dsivc` | `e986b27e5ea0a83a0b8603cc167f481aaa120d08` | 2026-06-05 |
| `flopy` | unknown (TODO) | unknown (TODO) | unknown (TODO) | snapshot; vendored version `3.10.0.dev5` |
| `mf6rtm` | unknown (TODO) | unknown (TODO) | unknown (TODO) | snapshot |
| `vorflow` | unknown (TODO) | unknown (TODO) | unknown (TODO) | snapshot |

### Notes

- **`pyemu`** — `rhugman/pyemu@feat_dsivc`. This branch carries the DSI / DSIVC / DSIAE emulator additions the curriculum is built on (`pyemu.emulators`); the `feat_dsivc` DSIVC code is required by the optimization notebook (`part1_08`). The vendored copy was refreshed to `e986b27e5ea0a83a0b8603cc167f481aaa120d08` on 2026-06-05.
- **`flopy`, `mf6rtm`, `vorflow`** — snapshots whose upstream repository, branch and commit SHA were **not recorded** when first vendored. The version strings embedded in the trees (`flopy` reports `3.10.0.dev5`; `vorflow` reports `0.0.1`) are not a substitute for the upstream SHA. **TODO: record the upstream SHA for each on the next refresh.**

## Refresh procedure

To refresh a vendored dependency to a new upstream commit:

1. **Clone the target branch** into a scratch location, for example:
   ```bash
   git clone --branch feat_dsivc https://github.com/rhugman/pyemu.git /tmp/pyemu_refresh
   ```
2. **Record the exact commit** before stripping git history:
   ```bash
   git -C /tmp/pyemu_refresh rev-parse HEAD
   ```
   Update the table above with the upstream repo, branch, commit SHA and the refresh date.
3. **Strip the `.git` directory and the heavy non-package dirs** so the snapshot is a slim source tree (the editable install only needs the package + setup files; pyemu's `autotest/`+`verification/`+`examples/` alone are ~700 MB):
   ```bash
   rm -rf /tmp/pyemu_refresh/{.git,autotest,verification,examples,test,runner,misc,temp,bin,etc,.github,.vscode,.claude}
   ```
4. **Replace the vendored tree** under `dependencies/<pkg>/` with the stripped clone.
5. **Smoke-test the import** in the project environment before committing:
   ```bash
   conda activate rtm_gmdsi
   python -c "import pyemu; print(pyemu.__file__)"   # must resolve under dependencies/
   ```
   For `mf6rtm`, `flopy` and `vorflow`, do the equivalent import check; for `pyemu` also confirm the emulator entry points import (`from pyemu.emulators import DSI, DSIVC`).
6. **Commit** the refreshed tree together with the updated provenance table.
