from .bcs import FrozenBed, PeriodicNS, PeriodicWE

BoundaryConditions = {
    "frozen_bed": FrozenBed,
    "periodic_ns": PeriodicNS,
    "periodic_we": PeriodicWE,
}
