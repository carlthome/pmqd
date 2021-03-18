import shutil
import tarfile

import conftest
import tensorflow_datasets as tfds

from pmqd.tfds import PMQD


class PMQDTest(tfds.testing.DatasetBuilderTestCase):
    """Tests for PMQD dataset."""

    EXAMPLE_DIR = conftest.dummy_data_dir()
    DL_DOWNLOAD_RESULT = "pmqd.csv"
    DATASET_CLASS = PMQD
    SPLITS = {
        "test": 4,
    }

    @classmethod
    def setUpClass(cls):
        tar = tarfile.open(cls.EXAMPLE_DIR / "audio.tgz")
        tar.extractall(cls.EXAMPLE_DIR)
        super().setUpClass()

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.EXAMPLE_DIR / "audio")
