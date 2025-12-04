from .bcs import FrozenBed, PeriodicNS, PeriodicWE, PeriodicNSGlobal, PeriodicWEGlobal
from .bcs import BoundaryCondition

BoundaryConditions = {
    "frozen_bed": FrozenBed,
    "periodic_ns": PeriodicNS,
    "periodic_we": PeriodicWE,
    "periodic_ns_global": PeriodicNSGlobal,
    "periodic_we_global": PeriodicWEGlobal,
}
