from abc import ABC, abstractmethod
import tensorflow as tf
from ...emulate.utils.normalizations import StandardizationLayer, FixedAffineLayer


class Normalizer(ABC):

    @abstractmethod
    def set_stats(means: tf.Tensor, variances: tf.Tensor):
        pass

    @abstractmethod
    def compute_stats(inputs: tf.Tensor):
        pass


class IdentityNormalizer(Normalizer):

    def set_stats(self, means, variances):
        pass

    def compute_stats(self, inputs):
        return 0, 0  # use another interface...


class NetworkNormalizer(Normalizer):

    def __init__(
        self, method: tf.keras.layers.Layer
    ):  # make the layer a base class for typing
        self.method = method

    def set_stats(self, means: tf.Tensor, variances: tf.Tensor):
        self.method.set_stats(means, variances)

    def compute_stats(self, inputs: tf.Tensor):
        return self.method.compute_stats(inputs)

# TODO: Look into various data drift metrics
def mahalanobis_distance(
    previous_means, previous_variances, current_means, current_variances
):
    """Computes the equivalent mahalonbis distance between the previous and current distributions to detect data drift.
    After standardizing the data, this reduces to a simple eulcidean norm and further a single z-score if its a single point.
    """

    mean_drifts = tf.abs(previous_means - current_means) / tf.sqrt(previous_variances)

    return mean_drifts


def cosine_simularity(previous_means: tf.Tensor, current_means: tf.Tensor) -> float:

    A = previous_means
    B = current_means

    cosine_simularity = tf.tensordot(A, B) / (tf.linalg.norm(A) * tf.linalg.norm(B))

    return cosine_simularity
