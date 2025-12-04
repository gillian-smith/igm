from .optimizer import Optimizer
from .adam import OptimizerAdam
from .adam_DA import OptimizerAdamDataAssimilation
from .lbfgs import OptimizerLBFGS
from .lbfgs_bounds import OptimizerLBFGSBounds
from .lbfgs_DA import OptimizerLBFGSDataAssimilation
from .cg import OptimizerCG
from .sequential import OptimizerSequential


Optimizers = {
    "adam": OptimizerAdam,
    "adam_da": OptimizerAdamDataAssimilation,
    "lbfgs": OptimizerLBFGS,
    "lbfgs_bounds": OptimizerLBFGSBounds,
    "lbfgs_da": OptimizerLBFGSDataAssimilation,
    "cg": OptimizerCG,
    "sequential": OptimizerSequential,
}

from .interfaces import InterfaceOptimizers
from .interfaces.interface import InterfaceOptimizer, Status