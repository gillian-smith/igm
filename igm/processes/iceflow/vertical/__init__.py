from .vertical import VerticalDiscr
from .vertical_lagrange import LagrangeDiscr
from .vertical_legendre import LegendreDiscr
from .vertical_molho import MOLHODiscr
from .vertical_ssa import SSADiscr

VerticalDiscrs = {
    "lagrange": LagrangeDiscr,
    "legendre": LegendreDiscr,
    "molho": MOLHODiscr,
    "ssa": SSADiscr,
}
