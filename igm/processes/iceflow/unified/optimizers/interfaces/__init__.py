from .interface import InterfaceOptimizer, Status
from .adam import InterfaceAdam
from .lbfgs import InterfaceLBFGS
from .cg import InterfaceCG
from .cg_newton import InterfaceCGNewton
from .sequential import InterfaceSequential
from .hessian import InterfaceHessian
from .trust_region import InterfaceTrustRegion

InterfaceOptimizers = {
    "adam": InterfaceAdam,
    "adam_da": InterfaceAdam,
    "lbfgs": InterfaceLBFGS,
    "lbfgs_bounds": InterfaceLBFGS,
    "lbfgs_da": InterfaceLBFGS,
    "cg": InterfaceCG,
    "cg_newton": InterfaceCGNewton,
    "sequential": InterfaceSequential,
    "hessian": InterfaceHessian,
    "trust_region": InterfaceTrustRegion,
}
