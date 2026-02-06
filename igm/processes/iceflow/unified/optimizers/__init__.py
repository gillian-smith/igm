from .optimizer import Optimizer
from .adam import OptimizerAdam
from .adam_DA import OptimizerAdamDataAssimilation
from .lbfgs import OptimizerLBFGS
from .lbfgs_bounds import OptimizerLBFGSBounds
from .lbfgs_DA import OptimizerLBFGSDataAssimilation
from .cg import OptimizerCG
from .cg_newton import OptimizerCGNewton
from .trust_region import OptimizerTrustRegion
from .sequential import OptimizerSequential
from .hessian import OptimizerHessian

Optimizers = {
    "adam": OptimizerAdam,
    "adam_da": OptimizerAdamDataAssimilation,
    "lbfgs": OptimizerLBFGS,
    "lbfgs_bounds": OptimizerLBFGSBounds,
    "lbfgs_da": OptimizerLBFGSDataAssimilation,
    "hessian": OptimizerHessian,
    "cg": OptimizerCG,
    "cg_newton": OptimizerCGNewton,
    "trust_region": OptimizerTrustRegion,
    "sequential": OptimizerSequential,
}

from .interfaces import InterfaceOptimizer, InterfaceOptimizers, Status
from .utils import SyntheticCosts