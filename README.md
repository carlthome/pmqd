# Perceived Music Quality Dataset (PMQD)

## Intro
This repository contains the dataset produced for the paper [Perceiving Music
Quality with GANs](https://arxiv.org/abs/2006.06287) [[1]](#1) in collaboration
between Peltarion and Epidemic Sound. The purpose is to evaluate methods for
quality rating music. It contains 975 segments from songs across 13 genres,
with degradations of various intensities applied. Each clip has an associated
human perceived quality rating, from 1 (`Bad`) to 5 (`Excellent`), which is the
median value of the rating assigned by 5 different people.


## Usage
The dataset consists of a CSV file with metadata (rating and information about
each segment, including genre, artist, track) and the corresponding audio
segments. These may be downloaded and used directly. For simplicity, we provide
code to load the data in both PyTorch and Tensorflow.

### Contents
The following data is hosted in the
[releases](https://github.com/Peltarion/pmqd/releases/) of this GitHub repo:

| URL                                                                     | Description                                                                |
|:------------------------------------------------------------------------|:---------------------------------------------------------------------------|
| https://github.com/Peltarion/pmqd/releases/download/v1.0.0/audio.tgz    | Archive with all music segments at 48kHz / 24-bit.                         |
| https://github.com/Peltarion/pmqd/releases/download/v1.0.0/audio32.tgz  | Archive with all music segments at the increased bit depth 48kHz / 32-bit. |
| https://github.com/Peltarion/pmqd/releases/download/v1.0.0/pmqd.csv     | Metadata (ratings and song information).                                   |

### Torch
Install `pmqd` with additional [PyTorch](https://pytorch.org/) and
[torchaudio](https://pytorch.org/audio/stable/index.html) dependencies:

```console
> pip install git+https://github.com/Peltarion/pmqd#egg=pmqd[torch]
```

To download the dataset to `"download_directory"` and use it:
```python
from pmqd.torch import PMQD
from torch.utils.data import DataLoader


dataset = PMQD(root="download_directory", download=True)
dataloader = DataLoader(dataset, batch_size=32)
sample_rate = PMQD.SAMPLE_RATE
...

for batch in dataloader:
    audio, rating = batch["audio"], batch["rating"]
    ...
```

### TFDS
This requires [ffmpeg](https://ffmpeg.org/) on the target system, and has to be
installed manually (e.g. for MacOS do `brew install ffmpeg`
([formula](https://formulae.brew.sh/formula/ffmpeg))). Alternatively,
see the [docker instructions](#docker) below.

Install `pmqd` with additional [Tensorflow](http://tensorflow.org/) and
[Tensorflow Datasets](https://www.tensorflow.org/datasets) dependencies:

```console
> pip install git+https://github.com/Peltarion/pmqd#egg=pmqd[tfds]
```

To download the dataset to the default `tfds` location and use it:
```python
import pmqd.tfds
import tensorflow_datasets as tfds


dataset, info = tfds.load("PMQD", split="test", with_info=True)
dataset = dataset.batch(32)
sample_rate = info.features["audio"].sample_rate
...

for batch in dataset:
    audio, rating = batch["audio"], batch["rating"]
    ...
```

### Docker
The repository contains an appropriate docker image with all dependencies
required for both `tfds` and `torch`. It can be built directly from the
repository. To build and open a prompt inside it, do:

```console
docker build --target pmqd -t pmqd https://github.com/Peltarion/pmqd.git#main
docker run -it pmqd bash
```

### Examples
Example notebooks for usage with `tfds` and `torch:`

- [tfds.ipynb](./notebooks/tfds.ipynb)
- [torch.ipynb](./notebooks/torch.ipynb)


## Data

### Format

#### Audio
The rated music segments are available as 48kHz / 24-bit PCM stereo mixes. For
compatibility with e.g. `torchaudio` they are also available at an increased
bit depth of 32-bits.

#### Metadata
The metadata table contains the following columns:

- **id:** ID of sample.
- **genre:** Genre.
- **artist:** Artist.
- **title:** Title.
- **degradation_type:** Type of applied degradation (see [Degrading audio
  quality](#degrading-audio-quality)).
- **degradation_intensity:** Intensity of the applied degradation from 0 (none)
  to 100 (maximum)
- **sample_start:** The start location in seconds of this segment in the full
  song.
- **sample_filename:** Filename of the sample.

 Below is a random sample of the contents:

|   id | genre      | artist                       | title                                | degradation_type   |   degradation_intensity |   rating |   sample_start | sample_filename                      |
|-----:|:-----------|:-----------------------------|:-------------------------------------|:-------------------|------------------------:|---------:|---------------:|:-------------------------------------|
|  137 | Blues      | Martin Carlberg              | Bad Bad Blood (Instrumental Version) | original           |                    0.00 |     5.00 |            130 | 04694c6cb0cb4833906259ee961d53b8.wav |
|  448 | Rnb & Soul | Park Lane feat. Vincent Vega | I Don't Wanna Be You                 | limiter            |                   90.16 |     4.00 |            199 | 0b6856dacd8d4c19ad9f25e4f2fe3f02.wav |
|  850 | Blues      | Henrik Nagy                  | Strolling In New Orleans 1           | noise              |                   20.67 |     2.00 |             34 | 71982b5dbf1a4d6f86e9638fb3574f97.wav |
|   80 | Country    | Martin Carlberg              | Appalachian Trail 2                  | original           |                    0.00 |     4.00 |             98 | d4404cbff14244d58450fa5c73c97481.wav |
|  675 | Funk       | Teddy Bergström              | Godspel Groove                       | lowpass            |                   63.92 |     3.00 |             31 | 5df0c95dda78408aa7f73fbad7c029cc.wav |


### Source
The original audio is from the Epidemic Sound catalog, an online service with
professionally produced, high-quality music. It contains a wide range of music
and is curated to conform well to contemporary music, as it is intended for use
by content creators. Using their catalog, we created a balanced dataset of
mutually exclusive genres by randomly sampling 5 songs from each selected
genre. For each song, we then randomly sample 3 segments of constant length,
approximately 4 seconds in duration.


### Degrading audio quality
To include tracks of varying quality we used a set of signal degradations
with the following open-source [REAPER
JSFX](https://www.reaper.fm/sdk/js) audio plugins:

- Distortion (`loser/waveShapingDstr`): Waveshaping distortion with the
  waveshape going from a sine-like shape (50\%) to square (100\%).
- Lowpass (`Liteon/butterworth24db`): Low-pass filtering, a 24 dB Butterworth
  filter configured to have a frequency cutoff from 20 kHz down to 1000 Hz.
- Limiter (`loser/MGA\_JSLimiter`): Mastering limiter, having all settings
  fixed except for the threshold that was lowered from 0 dB to -30 dB
  (introduces clipping artifacts).
- Noise (`Liteon/pinknoisegen`): Additive noise on a range from -25 dB (subtly
  audible) to 0.0 dB (clearly audible).
- Original: The original music segment.

Plugins were applied separately to each segment without effects chaining. The
parameter of each plugin is rescaled to [0, 100] and considered the intensity
of the degradation.
[notebooks/degrade_audio.ipynb](notebooks/degrade_audio.ipynb) shows how to use
the code used to degrade the original music segments.

From each original segment described in [Source](#source), we produce degraded
versions of each type of degradation with a randomly sampled intensity from the
uniform distribution of the range. This yields 75 music segments per genre, and
our dataset thus consists of the following number of samples:

| Genre               |   Count |
|:--------------------|--------:|
| Acoustic            |      75 |
| Blues               |      75 |
| Classical           |      75 |
| Country             |      75 |
| Electronica & Dance |      75 |
| Funk                |      75 |
| Hip Hop             |      75 |
| Jazz                |      75 |
| Latin               |      75 |
| Pop                 |      75 |
| Reggae              |      75 |
| Rnb & Soul          |      75 |
| Rock                |      75 |
| **Total**           | **975** |


### Annotating for human perceived listening quality
To annotate the music segments with their human perceived listening quality we
turn to crowdsourcing the task on Amazon Mechanical Turk (_AMT_). This has the
advantage of allowing significantly larger scale than controlled tests, though
introduces some potential problems such as cheating and underperforming
participants, which we handle as described in this section.

#### Task assignment
Tasks to be completed by human participants are created to rate segments for
their listening quality. Segments are randomly assigned to tasks such that each
task contains 10 segments, never contains duplicates, and each segment occurs in
at least 5 tasks. Participants may only perform one task in order to avoid
individuals biases. In total, we produce 488 tasks resulting in 4880 individual
segment evaluations.

#### Task specification
During a task, each participant is asked to specify which type of device they
will use for listening from the list: `smartphone speaker`, `speaker`,
`headphones`, `other`, `will not listen`. If any other option than `speaker` or
`headphones` was selected, the submission was rejected and the task
re-assigned.  For each segment in the task, we ask the user for an assessment
of audio quality, not musical content[[2]](#2). The question is phrased as:
_"How do you rate the audio quality of this music segment?"_, and may be
answered on the ordinal scale: _Bad_, _Poor_, _Fair_, _Good_ and _Excellent_,
corresponding to the numerical values 1, 2, 3, 4, 5.

#### Rating aggregation
Once all tasks are completed, the ratings are aggregated to produce one
perceived quality rating per segment. Since participants are listening in their
own respective environments, we are concerned with lo-fi audio equipment or
scripted responses trying to game _AMT_. Thus we use the median over the mean
rating to discount outliers.

#### Cheating
The following schemes are applied in an attempt to reduce cheating or
participants not following instructions:

- Filtering out submissions on undesired devices also increases the chance of
  rejecting bots
- Multiple submissions by the same participant despite warnings that this will
  lead to rejection are all rejected
- Tasks completed in a shorter amount of time than the total duration of all
  segments in the task are rejected
- Tasks where all segments are given the same rating despite large variation in
  degradation intensity are rejected
- The number of tasks available at any moment is restricted to 50, as a smaller
  amount has been shown to decrease the prevalence of cheating[[3]](#3).


### Dataset summary
The following are the average ratings assigned by degradation intensity and
distortion type. As expected, some degradations (distortion and noise) have a
much larger impact on the quality than others. Furthermore, we note that the
original tracks are on average rated below the excellent quality mark, despite
being high-fidelity recordings. Part of this could be explained by the
annotators expectations.

| Degradation intensity   |   Distortion |   Limiter |   Lowpass |   Noise |   Original |
|:------------------------|-------------:|----------:|----------:|--------:|-----------:|
| [0.0, 20.0)             |         3.05 |      4.04 |      3.97 |    3.47 |       4.02 |
| [20.0, 40.0)            |         2.69 |      3.72 |      4.00 |    3.00 |       -    |
| [40.0, 60.0)            |         2.39 |      3.86 |      3.82 |    2.37 |       -    |
| [60.0, 80.0)            |         2.17 |      3.90 |      3.55 |    1.78 |       -    |
| [80.0, 100.0)           |         1.59 |      3.74 |      3.31 |    1.37 |       -    |


## Cite as
If you use this in your research, please cite our paper as

```bibtex
@article{hilmkil2020perceiving,
  title={Perceiving Music Quality with GANs},
  author={Hilmkil, Agrin and Thom{\'e}, Carl and Arpteg, Anders},
  journal={arXiv preprint arXiv:2006.06287},
  year={2020}
}
```

## References

<a id="1">[1]</a>
Hilmkil, A., Thomé, C. and Arpteg, A., 2020.
Perceiving Music Quality with GANs.
arXiv preprint arXiv:2006.06287.

<a id="2">[2]</a>
Wilson, A. and Fazenda, B.M., 2016.
Perception of audio quality in productions of popular music.
Journal of the Audio Engineering Society, 64(1/2), pp.23-34.

<a id="3">[3]</a>
Eickhoff, C. and de Vries, A.P., 2013.
Increasing cheat robustness of crowdsourcing tasks.
Information retrieval, 16(2), pp.121-137.
