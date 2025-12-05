from .metric import Metric
from .cost import MetricCost
from .grad_u_norm import MetricGradUNorm
from .grad_w_norm import MetricGradWNorm
from .u import MetricU
from .w import MetricW

Metrics = {
    "cost": MetricCost,
    "grad_u_norm": MetricGradUNorm,
    "grad_w_norm": MetricGradWNorm,
    "u": MetricU,
    "w": MetricW,
}
