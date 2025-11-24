from .experiments import (
    ExperimentA,
    ExperimentB,
    ExperimentC,
    ExperimentE1,
    ExperimentE2,
    ExperimentF1,
    ExperimentF2,
)
from .plots import plot_comparison
from .runs import run_igm_unified

experiments_dict = {
    "A": ExperimentA,
    "B": ExperimentB,
    "C": ExperimentC,
    "E1": ExperimentE1,
    "E2": ExperimentE2,
    "F1": ExperimentF1,
    "F2": ExperimentF2
}