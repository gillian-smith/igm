from .metric import Metric
from .metric_cost import MetricCost
from .metric_grad_u_norm import MetricGradUNorm
from .metric_grad_w_norm import MetricGradWNorm
from .metric_u import MetricU
from .metric_w import MetricW

Metrics = {
    "cost": MetricCost,
    "grad_u_norm": MetricGradUNorm,
    "grad_w_norm": MetricGradWNorm,
    "u": MetricU,
    "w": MetricW,
}
