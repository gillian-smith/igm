#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

from omegaconf import DictConfig

from igm.common import State


def checks(cfg: DictConfig, state: State) -> None:
    """
    Validate configuration compatibility for the enthalpy module.

    Ensures that the iceflow module is enabled with compatible settings,
    including Weertman sliding law and matching vertical discretization
    parameters between iceflow and enthalpy modules.

    Raises:
        ValueError: If configuration requirements are not met.
    """
    if "iceflow" not in cfg.processes:
        raise ValueError("The 'iceflow' module is required for the 'enthalpy' module.")

    if cfg.processes.iceflow.physics.sliding.law != "weertman":
        raise ValueError(
            "The 'weertman' sliding law is required for the 'enthalpy' module."
        )

    # TODO: allow other vertical discretizations
    cfg_iceflow_numerics = cfg.processes.iceflow.numerics
    cfg_enthalpy_numerics = cfg.processes.enthalpy.numerics

    if cfg_iceflow_numerics.Nz != cfg_enthalpy_numerics.Nz:
        raise ValueError("The 'Nz' parameter should be the same in iceflow & enthalpy.")

    if cfg_iceflow_numerics.vert_spacing != cfg_enthalpy_numerics.vert_spacing:
        raise ValueError(
            "The 'vert_spacing' parameter should be the same in iceflow & enthalpy."
        )

    if cfg_iceflow_numerics.vert_basis.lower() != "lagrange":
        raise ValueError("The 'vert_basis' parameter should be Lagrange.")
