from .optimizer import Optimizer
from .optimizer_adam import OptimizerAdam
from .optimizer_adam_DA import OptimizerAdamDataAssimilation
from .optimizer_lbfgs import OptimizerLBFGS
from .optimizer_lbfgs_bounds import OptimizerLBFGSBounds
from .optimizer_lbfgs_DA import OptimizerLBFGSDataAssimilation
from .optimizer_cg import OptimizerCG
from .interface import InterfaceOptimizer, Status
from .interface_adam import InterfaceAdam
from .interface_lbfgs import InterfaceLBFGS
from .interface_cg import InterfaceCG

Optimizers = {
    "adam": OptimizerAdam,
    "adam_da": OptimizerAdamDataAssimilation,
    "lbfgs": OptimizerLBFGS,
    "lbfgs_bounds": OptimizerLBFGSBounds,
    "lbfgs_da": OptimizerLBFGSDataAssimilation,
    "cg": OptimizerCG,
}

InterfaceOptimizers = {
    "adam": InterfaceAdam,
    "adam_da": InterfaceAdam,
    "lbfgs": InterfaceLBFGS,
    "lbfgs_bounds": InterfaceLBFGS,
    "lbfgs_da": InterfaceLBFGS,
    "cg": InterfaceCG,
}
