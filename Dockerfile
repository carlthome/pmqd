FROM python:3.6.9 AS main

# Install binary dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg && \
    apt-get autoremove -y && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install tox
RUN pip install tox

# Copy source
COPY . /pmqd


# Run tests
FROM main AS tests
WORKDIR /pmqd
RUN tox


# Default target with PMQD installed
FROM main AS pmqd
RUN pip install /pmqd
