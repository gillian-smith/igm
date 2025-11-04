from .criterion import Criterion
from .criterion_abs_tol import CriterionAbsTol
from .criterion_rel_tol import CriterionRelTol
from .criterion_inf import CriterionInf
from .criterion_nan import CriterionNaN
from .criterion_threshold import CriterionThreshold


Criteria = {
    "abs_tol": CriterionAbsTol,
    "rel_tol": CriterionRelTol,
    "inf": CriterionInf,
    "nan": CriterionNaN,
    "threshold": CriterionThreshold,
}
