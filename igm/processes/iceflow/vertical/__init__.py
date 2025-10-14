from .vertical import VerticalDiscr
from .vertical_lagrange import LagrangeDiscr
from .vertical_legendre import LegendreDiscr
from .vertical_sia import SIADiscr
from .vertical_ssa import SSADiscr

VerticalDiscrs = {
    "lagrange": LagrangeDiscr,
    "legendre": LegendreDiscr,
    "sia": SIADiscr,
    "ssa": SSADiscr,
}
