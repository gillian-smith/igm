import tensorflow as tf


def normalize_precision(p) -> tf.dtypes.DType:
    if isinstance(p, tf.dtypes.DType):
        if p not in (tf.float32, tf.float64):
            raise ValueError("precision dtype must be tf.float32 or tf.float64")
        return p
    s = str(p).lower()
    if s in ("single", "fp32", "float32"):
        return tf.float32
    if s in ("double", "fp64", "float64"):
        return tf.float64
    raise ValueError(
        f"Unknown precision '{p}' (use 'single'|'double' or tf.float32|tf.float64)."
    )
