"""PyTorch Dataset for loading PMQD.

Based on torchaudio.datasets.librispeech.LIBRISPEECH.
"""
import os
from pathlib import Path
from typing import Any, Dict, Union

import pandas as pd
import torchaudio
from torch.utils.data import Dataset
from torchaudio.datasets.utils import download_url, extract_archive

from . import checksums

# Uses global constants for consistency with other torchaudio.datasets.
# Requires the 32-bit version for compatibility with torchaudio
AUDIO = checksums.query("filename == 'audio32.tgz'").iloc[0]
METADATA = checksums.query("filename == 'pmqd.csv'").iloc[0]
CHECKSUM_AUDIO = AUDIO["checksum"]
CHECKSUM_METADATA = METADATA["checksum"]
URL_AUDIO = AUDIO["url"]
URL_METADATA = METADATA["url"]
FOLDER_IN_ARCHIVE, _ = os.path.splitext(os.path.basename(URL_AUDIO))


class PMQD(Dataset):
    """Create a Dataset for PMQD.

    Args:
        root: Path to the directory where the dataset is found or downloaded.
        url_audio: The URL to download the dataset from, or the type of the
                   dataset to dowload.
        url_metadata: The URL to download the dataset from, or the type of the
                      dataset to dowload.
        folder_in_archive: The top-level directory of the dataset.
        download: Whether to download the dataset if it is not found at root path.
    """

    SAMPLE_RATE = 48000

    def __init__(
        self,
        root: Union[str, Path],
        url_audio: str = URL_AUDIO,
        url_metadata: str = URL_METADATA,
        folder_in_archive: str = FOLDER_IN_ARCHIVE,
        download: bool = False,
    ) -> None:
        root = Path(root)
        root.mkdir(exist_ok=True)  # Ensure the root directory exists
        archive = root / os.path.basename(url_audio)
        self._audio_path = root / folder_in_archive
        self._metadata_path = root / os.path.basename(url_metadata)

        if download:
            if not self._audio_path.exists():
                if not archive.is_file():
                    download_url(url_audio, root, hash_value=CHECKSUM_AUDIO)
                extract_archive(archive)
            if not self._metadata_path.is_file():
                download_url(url_metadata, root, hash_value=CHECKSUM_METADATA)

        if not self._metadata_path.is_file():
            raise FileNotFoundError(
                "Metadata CSV not found, use download=True to download the dataset."
            )
        if not self._audio_path.is_dir():
            raise FileNotFoundError(
                "Audio directory not found, use download=True to download the dataset."
            )
        self.metadata = pd.read_csv(self._metadata_path, index_col="id")

    def __getitem__(self, n: int) -> Dict[str, Any]:
        """Load the n-th example from the dataset.

        Args:
            n: The index of the example to be loaded

        Returns:
            A dictionary containing the metadata for the example and the loaded
            waveform in the key "audio" with shape [channels, samples].
        """
        row = self.metadata.loc[n]
        audio_path = os.path.join(self._audio_path, row["sample_filename"])
        audio, sample_rate = torchaudio.load(audio_path)
        assert sample_rate == self.SAMPLE_RATE

        return {
            "id": n,
            "genre": row["genre"],
            "artist": row["artist"],
            "title": row["title"],
            "degradation_type": row["degradation_type"],
            "degradation_intensity": row["degradation_intensity"],
            "rating": row["rating"],
            "sample_start": row["sample_start"],
            "audio": audio,
        }

    def __len__(self) -> int:
        return len(self.metadata)
