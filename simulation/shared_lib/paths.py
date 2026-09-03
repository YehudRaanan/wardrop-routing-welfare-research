"""Portable paths for the public research package.

Set ``WARDROP_PRIMARY_DB`` to use a database outside ``simulation/data``.
The working project's cloud storage and machine-specific paths are deliberately
not part of this publication copy.
"""

import os
from pathlib import Path
from typing import Final

PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parent.parent
DATA_DIR: Final[Path] = PROJECT_ROOT / "data"

_DEFAULT_PRIMARY_DB = DATA_DIR / "simulation.db"

_PRIMARY_DB_OVERRIDE = os.environ.get("WARDROP_PRIMARY_DB")
PRIMARY_DB_PATH: Final[Path] = (
    Path(_PRIMARY_DB_OVERRIDE) if _PRIMARY_DB_OVERRIDE else _DEFAULT_PRIMARY_DB
)
PRIMARY_DB_PATH_STR: Final[str] = str(PRIMARY_DB_PATH)

ARCHIVE_DB_PATH: Final[Path] = PRIMARY_DB_PATH.parent / "simulation_archive.db"
ARCHIVE_DB_PATH_STR: Final[str] = str(ARCHIVE_DB_PATH)

PRIMARY_DB_GUARD_PATH: Final[Path] = DATA_DIR / ".PRIMARY_DB_GUARD"

MIRROR_DB_PATH: Final[Path] = DATA_DIR / "simulation.db"
LEGACY_DRIVE_STREAM_DB_PATH: Final[Path] = MIRROR_DB_PATH

GCP_DIR: Final[Path] = PROJECT_ROOT / "gcp"
SHARED_LIB_DIR: Final[Path] = PROJECT_ROOT / "shared_lib"
TOOLS_DIR: Final[Path] = PROJECT_ROOT / "tools"
