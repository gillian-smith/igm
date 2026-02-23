import tensorflow as tf
import warnings
from itertools import cycle
from contextlib import contextmanager

_colors = cycle(["blue", "green", "yellow", "red", "magenta", "cyan", "white"])

try:
    import nvtx
except ImportError:
    warnings.warn("NVTX is not installed. Profiling will not be available.")


def srange(message, color):
    tf.test.experimental.sync_devices()
    return nvtx.start_range(message, color)


def erange(rng):
    tf.test.experimental.sync_devices()
    nvtx.end_range(rng)


@contextmanager
def profile_range(name: str, enabled: bool = True):
    rng = srange(name, next(_colors)) if enabled else None
    try:
        yield
    finally:
        if rng is not None:
            erange(rng)
