from .bc import BoundaryCondition
from .frozen_bed import FrozenBed
from .periodic_ns import PeriodicNS, PeriodicNSGlobal
from .periodic_we import PeriodicWE, PeriodicWEGlobal

BoundaryConditions = {
    "frozen_bed": FrozenBed,
    "periodic_ns": PeriodicNS,
    "periodic_we": PeriodicWE,
    "periodic_ns_global": PeriodicNSGlobal,
    "periodic_we_global": PeriodicWEGlobal,
}

from .interfaces import InterfaceBoundaryCondition, InterfaceBoundaryConditions
