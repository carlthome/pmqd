import pandas as pd
from pkg_resources import resource_stream

from . import degradation

with resource_stream("pmqd", "tfds/pmqd/checksums.tsv") as f:
    checksums = pd.read_csv(
        f, sep="\t", header=None, names=["url", "bytes", "checksum", "filename"]
    )

__all__ = [
    "checksums",
    "degradation",
]
