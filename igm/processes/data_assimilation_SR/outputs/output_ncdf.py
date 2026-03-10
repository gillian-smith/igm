#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors 
# Published under the GNU GPL (Version 3), check at the LICENSE file

import numpy as np
import tensorflow as tf
from scipy import stats
from netCDF4 import Dataset
from igm.utils.math.getmag import getmag

def update_ncdf_optimize(cfg, state, it):
    """
    Initialize and write the ncdf optimze file
    """

        
    if hasattr(state, "logger"):
        state.logger.info("Initialize  and write NCDF output Files")

    has_costs = hasattr(state, "da_cost_total")
        
    if "velbase_mag" in cfg.processes.data_assimilation_SR.output.vars_to_save:
        state.velbase_mag = getmag(state.uvelbase, state.vvelbase)

    if "velsurf_mag" in cfg.processes.data_assimilation_SR.output.vars_to_save:
        state.velsurf_mag = getmag(state.uvelsurf, state.vvelsurf)

    if "velsurfobs_mag" in cfg.processes.data_assimilation_SR.output.vars_to_save:
        state.velsurfobs_mag = getmag(state.uvelsurfobs, state.vvelsurfobs)
    
    if "sliding_ratio" in cfg.processes.data_assimilation_SR.output.vars_to_save:
        state.sliding_ratio = tf.where(state.velsurf_mag > 10, state.velbase_mag / state.velsurf_mag, np.nan)

    if it == 0:
        nc = Dataset(
            "optimize.nc",
            "w",
            format="NETCDF4",
        )

        nc.createDimension("iterations", None)
        E = nc.createVariable("iterations", np.dtype("float32").char, ("iterations",))
        E.units = "None"
        E.long_name = "iterations"
        E.axis = "ITERATIONS"
        E[0] = it

        nc.createDimension("y", len(state.y))
        E = nc.createVariable("y", np.dtype("float32").char, ("y",))
        E.units = "m"
        E.long_name = "y"
        E.axis = "Y"
        E[:] = state.y.numpy()

        nc.createDimension("x", len(state.x))
        E = nc.createVariable("x", np.dtype("float32").char, ("x",))
        E.units = "m"
        E.long_name = "x"
        E.axis = "X"
        E[:] = state.x.numpy()

        if has_costs:
            C = nc.createVariable("da_cost_total", np.dtype("float32").char, ("iterations",))
            C.long_name = "DA total cost"
            C[0] = state.da_cost_total
            C = nc.createVariable("da_cost_data", np.dtype("float32").char, ("iterations",))
            C.long_name = "DA data misfit cost"
            C[0] = state.da_cost_data
            C = nc.createVariable("da_cost_reg", np.dtype("float32").char, ("iterations",))
            C.long_name = "DA regularization cost"
            C[0] = state.da_cost_reg

        for var in cfg.processes.data_assimilation_SR.output.vars_to_save:
            E = nc.createVariable(
                var, np.dtype("float32").char, ("iterations", "y", "x")
            )
            E[0, :, :] = vars(state)[var].numpy()

        nc.close()

    else:
        nc = Dataset("optimize.nc", "a", format="NETCDF4", )

        d = nc.variables["iterations"][:].shape[0]

        if has_costs:
            nc.variables["da_cost_total"][d] = state.da_cost_total
            nc.variables["da_cost_data"][d] = state.da_cost_data
            nc.variables["da_cost_reg"][d] = state.da_cost_reg

        nc.variables["iterations"][d] = it

        for var in cfg.processes.data_assimilation_SR.output.vars_to_save:
            nc.variables[var][d, :, :] = vars(state)[var].numpy()

        nc.close()
