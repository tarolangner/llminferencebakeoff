"""Experiment configuration.

This is the single file to edit when switching experiments. Change MODEL_NAME
and GPU_CONFIG to match your target hardware, then set the experiment flags
below to select which experiment variant to run.

See README for the full experiment matrix.
"""

# -- Model / GPU ---------------------------------------------------------------
# Experiments 1-3 (single-user latency/throughput comparison via serve.py):
MODEL_NAME = "Qwen/Qwen3.5-4B"
GPU_CONFIG = "L4"
GPU_MEMORY_UTILIZATION = 0.8
MAX_CONTEXT_LENGTH = 4096

# Experiment 4 (concurrent benchmark via benchmark_concurrent.py):
# MODEL_NAME = "google/gemma-4-26B-A4B-it"
# GPU_CONFIG = "H100"
# GPU_MEMORY_UTILIZATION = 0.9

# -- Experiment flags ----------------------------------------------------------
# Experiment 1 - baseline:        DISABLE_PREFIX_CACHING=True,  ENABLE_MTP=False
# Experiment 2 - MTP:             DISABLE_PREFIX_CACHING=True,  ENABLE_MTP=True
# Experiment 3 - prefix caching:  DISABLE_PREFIX_CACHING=False, ENABLE_MTP=False
# Experiment 4 - concurrent:      DISABLE_PREFIX_CACHING=True,  ENABLE_MTP=False
DISABLE_PREFIX_CACHING = True
ENABLE_MTP = False
