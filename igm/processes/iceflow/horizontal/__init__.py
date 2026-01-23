from .horizontal import HorizontalDiscr
from .horizontal_central import CentralDiscr
from .horizontal_q1 import Q1Discr
from .horizontal_p1 import P1Discr
from .horizontal_mac import MACDiscr

HorizontalDiscrs = {
    "central": CentralDiscr,
    "q1": Q1Discr,
    "p1": P1Discr,
    "mac": MACDiscr,
}
