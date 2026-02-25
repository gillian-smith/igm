#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

from omegaconf import DictConfig

from igm.common import State


def checks(cfg: DictConfig, state: State) -> None:
    """
    Validate configuration compatibility for the enthalpy module.

    Ensures that the iceflow module is enabled and configured with the
    Weertman sliding law, which is required for enthalpy calculations.

    Raises:
        ValueError: If configuration requirements are not met.
    """
    if "iceflow" not in cfg.processes:
        raise ValueError("The 'iceflow' module is required for the 'enthalpy' module.")

    if cfg.processes.iceflow.physics.sliding.law != "weertman":
        raise ValueError(
            "The 'weertman' sliding law is required for the 'enthalpy' module."
        )
