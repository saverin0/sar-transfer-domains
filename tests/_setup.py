"""Shared start of every test here: sartransfer from THIS repo's src, temp files in one folder.

Run a test as a script, from any folder:   python tests/test_contract.py

- sys.path gets <repo>/src first, and the import is checked: a sartransfer from anywhere
  else (an installed copy, part one's repo) stops the test. The path is printed.
- Temp files go under <system temp>/sartransfer-part2-tests (the system temp folder comes
  from TMPDIR / TEMP / TMP; set them to choose a disk with a few GB free).
- Optional, only for the tests that use them:
    SARTRANSFER_TEST_PYLIB  folder(s) added at the END of sys.path (e.g. one holding h5py
                            when it is not installed; test_alpine needs h5py)
    SARTRANSFER_TEST_REF    folder(s) with reference files; test_snow section 6 looks for
                            snow_src/central_directory.bin (the real SnowSAR zip directory,
                            metadata only) and is skipped without it
"""
import os
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
sys.path.insert(0, str(SRC))
for _extra in filter(None, os.environ.get("SARTRANSFER_TEST_PYLIB", "").split(os.pathsep)):
    if _extra not in sys.path:
        sys.path.append(_extra)

import sartransfer  # noqa: E402

ORIGIN = Path(sartransfer.__file__).resolve()
if SRC.resolve() not in ORIGIN.parents:
    raise SystemExit(f"IMPORT GUARD: sartransfer imported from {ORIGIN}, not from {SRC}")
print(f"import guard ok: sartransfer {sartransfer.__version__} from {ORIGIN.parent}")

TMP_ROOT = Path(tempfile.gettempdir()).resolve()
TMP_BASE = TMP_ROOT / "sartransfer-part2-tests"
TMP_BASE.mkdir(parents=True, exist_ok=True)
REF = [Path(p) for p in os.environ.get("SARTRANSFER_TEST_REF", "").split(os.pathsep) if p]
