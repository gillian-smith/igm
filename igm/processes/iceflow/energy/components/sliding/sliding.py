#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

from typing import Any, Dict
from omegaconf import DictConfig

from ..energy import EnergyComponent


class SlidingComponent(EnergyComponent):
    """Energy component representing frictional energy."""

    pass


def get_sliding_params_args(cfg: DictConfig) -> Dict[str, Any]:
    """Extract friction parameters from configuration."""

    cfg_physics = cfg.processes.iceflow.physics

    law = cfg_physics.sliding.law

    return dict(cfg_physics.sliding[law])
