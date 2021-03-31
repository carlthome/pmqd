"""PMQD dataset."""
import resource
from pathlib import Path

import pandas as pd
import tensorflow as tf
import tensorflow_datasets as tfds

# TODO: Remove the following workaround for
#       https://github.com/tensorflow/datasets/issues/1441
low, high = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (high, high))

_DESCRIPTION = """
This is the dataset produced for the paper [Perceiving Music Quality with
GANs](https://arxiv.org/abs/2006.06287) in collaboration between
Peltarion and Epidemic Sound. The purpose is to evaluate methods for quality
rating music. It contains 975 segments from songs across 13 genres, with
degradations of various intensity applied. Each clip has an associated human
perceived quality rating, from 1 (`Bad`) to 5 (`Excellent`), which is the
median value of the rating assigned by 5 different people.

See [github.com/Peltarion/pmqd](https://github.com/Peltarion/pmqd) for more
details.
"""

_CITATION = """
@article{hilmkil2020perceiving,
  title={Perceiving Music Quality with GANs},
  author={Hilmkil, Agrin and Thom{\'e}, Carl and Arpteg, Anders},
  journal={arXiv preprint arXiv:2006.06287},
  year={2020}
}
"""

URL_AUDIO = "https://storage.googleapis.com/pmqd/audio.tgz"
URL_METADATA = "https://storage.googleapis.com/pmqd/pmqd.csv"
FOLDER_IN_ARCHIVE = "audio"


class PMQD(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for PMQD dataset."""

    VERSION = tfds.core.Version("1.0.0")
    RELEASE_NOTES = {
        "1.0.0": "Initial release.",
    }

    def _info(self) -> tfds.core.DatasetInfo:
        """Returns the dataset metadata."""
        return tfds.core.DatasetInfo(
            builder=self,
            description=_DESCRIPTION,
            features=tfds.features.FeaturesDict(
                {
                    "id": tfds.features.Tensor(shape=(), dtype=tf.int32),
                    "genre": tfds.features.Text(),
                    "artist": tfds.features.Text(),
                    "title": tfds.features.Text(),
                    "degradation_type": tfds.features.ClassLabel(
                        names=["original", "distortion", "limiter", "lowpass", "noise"]
                    ),
                    "degradation_intensity": tfds.features.Tensor(
                        shape=(), dtype=tf.float32
                    ),
                    "rating": tfds.features.Tensor(shape=(), dtype=tf.float32),
                    "sample_start": tfds.features.Tensor(shape=(), dtype=tf.int32),
                    "audio": tfds.features.Audio(
                        shape=(None, 2),
                        file_format="wav",
                        dtype=tf.float32,
                        sample_rate=48000,
                    ),
                }
            ),
            supervised_keys=("audio", "rating"),
            homepage="https://github.com/Peltarion/pmqd",
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Returns SplitGenerators."""
        path_audio = dl_manager.download_and_extract(URL_AUDIO)
        path_metadata = dl_manager.download(URL_METADATA)
        metadata = pd.read_csv(path_metadata, index_col="id")

        return {
            "test": self._generate_examples(metadata, path_audio / FOLDER_IN_ARCHIVE),
        }

    def _generate_examples(self, metadata: pd.DataFrame, path: Path):
        """Yields examples with loaded audio.

        Yields:
            A dictionary containing the metadata for the example and the loaded
            waveform in the key "audio" with shape [samples, channels].
        """
        for i, row in metadata.iterrows():
            yield i, {
                "id": i,
                "genre": row["genre"],
                "artist": row["artist"],
                "title": row["title"],
                "degradation_type": row["degradation_type"],
                "degradation_intensity": row["degradation_intensity"],
                "rating": row["rating"],
                "sample_start": row["sample_start"],
                "audio": path / row["sample_filename"],
            }
