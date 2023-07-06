#!/usr/bin/env python3

# Copyright (C) 2021-2023 Guillaume Jouvet <guillaume.jouvet@unil.ch>
# Published under the GNU GPL (Version 3), check at the LICENSE file 

"""
This IGM module write time serie variables (ice glaciated area and volume)
into the ncdf output file ts.nc
==============================================================================
Input: self.thk, self.dx
Output: ts.nc
"""

import numpy as np
import os, sys, shutil
import matplotlib.pyplot as plt
import tensorflow as tf
from netCDF4 import Dataset

def params_write_ncdf_ts(self):
    pass


def init_write_ncdf_ts(params, self):
    os.system("echo rm " + os.path.join(params.working_dir, "ts.nc") + " >> clean.sh")

    self.var_info_ncdf_ts = {}
    self.var_info_ncdf_ts["vol"] = ["Ice volume", "km^3"]
    self.var_info_ncdf_ts["area"] = ["Glaciated area", "km^2"]


def update_write_ncdf_ts(params, self):

    if self.saveresult:
        vol = np.sum(self.thk) * (self.dx**2) / 10**9
        area = np.sum(self.thk > 1) * (self.dx**2) / 10**6

        if not hasattr(self, "already_called_update_write_ncdf_ts"):
            self.already_called_update_write_ncdf_ts = True

            self.logger.info("Initialize NCDF ts output Files")

            nc = Dataset(
                os.path.join(params.working_dir, "ts.nc"),
                "w",
                format="NETCDF4",
            )

            nc.createDimension("time", None)
            E = nc.createVariable("time", np.dtype("float32").char, ("time",))
            E.units = "yr"
            E.long_name = "time"
            E.axis = "T"
            E[0] = self.t.numpy()

            for var in ["vol", "area"]:
                E = nc.createVariable(var, np.dtype("float32").char, ("time"))
                E[0] = vars()[var].numpy()
                E.long_name = self.var_info_ncdf_ts[var][0]
                E.units = self.var_info_ncdf_ts[var][1]
            nc.close()

            # os.system('echo rm '+ os.path.join(params.working_dir, "ts.nc") + ' >> clean.sh')

        else:
            self.logger.info("Write NCDF ts file at time : " + str(self.t.numpy()))

            nc = Dataset(
                os.path.join(params.working_dir, "ts.nc"),
                "a",
                format="NETCDF4",
            )
            d = nc.variables["time"][:].shape[0]

            nc.variables["time"][d] = self.t.numpy()
            for var in ["vol", "area"]:
                nc.variables[var][d] = vars()[var].numpy()
            nc.close()
