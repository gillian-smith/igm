from .optimizer import Optimizer
from .optimizer_adam import OptimizerAdam
from .optimizer_lbfgs import OptimizerLBFGS
from .interface import InterfaceOptimizer, Status
from .interface_adam import InterfaceAdam
from .interface_lbfgs import InterfaceLBFGS
from .optimizer_lbfgs_DA import OptimizerLBFGSDataAssimilation
from .optimizer_adam_DA import OptimizerAdamDataAssimilation
from .interface_cg import InterfaceCG
from .optimizer_cg import OptimizerCG

Optimizers = {
    "adam": OptimizerAdam,
    "lbfgs": OptimizerLBFGS,
    "lbfgs_da": OptimizerLBFGSDataAssimilation,
    "adam_da": OptimizerAdamDataAssimilation,
    "cg": OptimizerCG,
}

InterfaceOptimizers = {
    "adam": InterfaceAdam,
    "lbfgs": InterfaceLBFGS,
    "cg": InterfaceCG,
}
