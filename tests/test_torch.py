import os
import shutil
from pathlib import Path
from typing import Dict
from unittest.mock import patch

import numpy as np
import pytest
from torch import Tensor

from pmqd.torch import PMQD


def test_torchaudio_no_dowload(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        PMQD(tmp_path, download=False)


def test_torchaudio_download(tmp_path: Path, dummy_data: Dict[str, Path]):
    def mock_download_url(url: str, download_folder: str, hash_value: str) -> None:
        filename = os.path.basename(url)
        shutil.copy(dummy_data[filename], download_folder)

    with patch("pmqd.torch.download_url", wraps=mock_download_url) as mocked:
        dataset = PMQD(root=tmp_path, download=True)
        mocked.assert_called()

    # Size of subset
    assert len(dataset) == 4

    # Check one instance
    example = dataset[0]
    assert isinstance(example["audio"], Tensor)
    assert isinstance(example["sample_start"], np.int64)
    assert isinstance(example["rating"], float)

    # Check that files are not downloaded again
    with patch("torchaudio.datasets.utils.download_url") as patched:
        dataset = PMQD(root=tmp_path, download=True)
        patched.assert_not_called()
