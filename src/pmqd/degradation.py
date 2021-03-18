"""JSFX for various degradation with intensity in [0, 100]."""
import os
import shutil
import subprocess
import tempfile
from multiprocessing import Pool, cpu_count
from typing import Callable, Collection, Dict, Tuple

from tqdm.auto import tqdm


def distortion(intensity: float = 100.0) -> str:
    """Simple waveshaping distortion."""
    assert 0.0 <= intensity <= 100.0

    min_percentage = 50.0
    percentage = (100 - min_percentage) * intensity / 100 + min_percentage

    fxchain = f"""
    <JS loser/waveShapingDstr
    {percentage}
    >
    """
    return fxchain


def limiter(intensity: float = 100.0) -> str:
    """Simple peak limiter."""
    assert 0.0 <= intensity <= 100.0

    floor = -30.0
    db = floor * intensity / 100.0

    fxchain = f"""
    <JS loser/MGA_JSLimiter ""
    {db} 0.000000 -0.100000
    >
    """
    return fxchain


def lowpass(intensity: float = 100.0) -> str:
    """Butterworth low-pass filter."""
    assert 0.0 <= intensity <= 100.0

    min_cutoff = 56.5  # 1000 Hz
    cutoff = 100 + (min_cutoff - 100) * intensity / 100

    fxchain = f"""
    <JS Liteon/butterworth24db
    0.000000 0.000000 {cutoff} 0.000000 0.000000 0.000000
    >
    """
    return fxchain


def noise(intensity: float = 100.0) -> str:
    """Linearly mixed in PURPLE noise."""
    assert 0.0 <= intensity <= 100.0

    floor = -25.0
    db = floor * (1 - intensity / 100.0)

    fxchain = f"""
    <JS Liteon/pinknoisegen
    0.000000 {db} 0.000000 0.000000
    >
    """
    return fxchain


DEGRADATIONS: Dict[str, Callable[[float], str]] = {
    "distortion": distortion,
    "limiter": limiter,
    "lowpass": lowpass,
    "noise": noise,
}


def degrade(
    degradation_type: str,
    degradation_intensity: float,
    original_path: str,
    target_path: str,
) -> subprocess.CompletedProcess:
    """Apply degradation and return result of subprocess execution."""
    degradation_fn = DEGRADATIONS[degradation_type]
    degradation = degradation_fn(degradation_intensity)
    temp_dir = tempfile.TemporaryDirectory()
    config = f"""
    <CONFIG
      <FXCHAIN {degradation}>
      SRATE 48000
      NCH 2
      DITHER 3
      PAD_START 0.0
      PAD_END 0.0
      OUTPATH '{temp_dir.name}'
    >
    """

    with temp_dir as temp_path:
        config_filename = os.path.join(temp_path, "config")
        with open(config_filename, "w") as f:
            print(original_path, file=f)
            print(config, file=f)

        result = subprocess.run(
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            args=["reaper", "-batchconvert", config_filename],
        )

        # Move to target location
        wavfile = os.path.basename(original_path)
        outpath = os.path.join(temp_path, wavfile)
        if not os.path.isfile(outpath):
            with open(os.path.join(temp_path, "config.log"), "r") as log_file:
                raise FileNotFoundError(
                    "Unable to find Reaper output file. Consult the Reaper log entry:\n"
                    "----------------------------------------------------------------\n"
                    + log_file.read()
                )
        shutil.copy2(outpath, target_path)
    return result


def degrade_unpack(item: Tuple[str, float, str, str]) -> subprocess.CompletedProcess:
    """Return the result of applying item as arguments to `degrade`."""
    degradation_type, degradation_intensity, original_path, target_path = item
    return degrade(
        degradation_type=degradation_type,
        degradation_intensity=degradation_intensity,
        original_path=original_path,
        target_path=target_path,
    )


def degrade_all(items: Collection[Tuple[str, float, str, str]]) -> bool:
    """Apply degradations multiple clips in parallel.

    Note:
        This is pretty inefficient since it starts up Reaper for each item. We
        do this to be able to avoid multiple input files with the same name
        overwriting each other. A more efficient method would be to produce a
        config with multiple conversions.

    Args:
        items: A collection of of arguments to `degrade` in order

    Returns:
        True if successful, False if not.
    """
    with Pool(cpu_count()) as pool:
        results = pool.imap_unordered(degrade_unpack, items)
        for result in tqdm(results, total=len(items)):
            if result.returncode != 0:
                raise RuntimeError("Failed: %s" % " ".join(result.args))

    return all(os.path.exists(item[3]) for item in items)
