from pathlib import Path
from typing import Dict

import pytest


def dummy_data_dir() -> Path:
    """Return the path to the dummy data directory."""
    dir_path = Path(__file__).parent
    return dir_path / "dummy_data"


@pytest.fixture
def dummy_data() -> Dict[str, Path]:
    """Return a dictionary mapping from filename to the path of each dummy data file."""
    return {
        "audio.tgz": dummy_data_dir() / "audio.tgz",
        "audio32.tgz": dummy_data_dir() / "audio32.tgz",
        "pmqd.csv": dummy_data_dir() / "pmqd.csv",
    }
