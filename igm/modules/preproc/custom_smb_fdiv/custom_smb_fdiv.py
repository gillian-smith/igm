#!/usr/bin/env python3

# Copyright (C) 2024 Gillian Smith <g.smith@ed.ac.uk>
# Published under the GNU GPL (Version 3), check at the LICENSE file

import numpy as np
import matplotlib.pyplot as plt
import os, glob, shutil, scipy
from netCDF4 import Dataset
import tensorflow as tf
import pandas as pd

from igm.modules.utils import *
#from igm.modules.utils import str2bool

#from igm.modules.utils import complete_data

def params(parser):
    parser.add_argument(
        "--csmb_path_ncfile",
        type=str,
        default="input_saved_thkobs.nc", # assuming we ran custom_thkobs first
        help="Path to netCDF file created by oggm_shop"
    )
    parser.add_argument(
        "--csmb_path_smbobs",
        type=str,
        default="/home/s1639117/Documents/igm_folder/DATA/SMB/5119153/compressed_RGI131415_ALL2km_13-May-2020/",
        help="Path to SMB and FDIV .tif files"
    )
    parser.add_argument(
        "--csmb_shift_coords",
        type=str2bool,
        default=False,
        help="Shift coordinates"
    )
    # Always run ["oggm_shop", "custom_thkobs", "custom_smb_fdiv"]
    # Then ["load_ncdf"] with "lncd_input_file" : "input_saved_with_smb.nc"
    # You must have oggm_save_in_ncdf=True
    # If intending to use individual RGI outline as downloaded from oggm_shop you MUST set oggm_remove_RGI_folder=False

def initialize(params, state):

    import rioxarray as rioxr
    import geopandas as gpd
    import xarray as xr
    import shapely
    #from pyproj import Transformer
    from scipy.interpolate import RectBivariateSpline

    nc = xr.open_dataset(params.csmb_path_ncfile)

    # IGM x and y coordinates

    x = np.squeeze(nc.variables["x"]).astype("float32").values
    #y = np.flip(np.squeeze(nc.variables["y"]).astype("float32"))
    y = np.squeeze(nc.variables["y"]).astype("float32").values

    short_rgi_id = params.oggm_RGI_ID[-8:]
    csmb_path_smbobs_dir = params.csmb_path_smbobs + short_rgi_id +"/"

    fdiv = xr.open_dataarray(csmb_path_smbobs_dir+short_rgi_id+"_FDIV.tif") # flux div in m/yr
    smb = xr.open_dataarray(csmb_path_smbobs_dir+short_rgi_id+"_SMB.tif") # SMB in m w.e. /yr

    if params.csmb_shift_coords:
        fdiv = shift_coords(fdiv)
        smb = shift_coords(smb)

    nc["divfluxobs"] = fdiv.interp(x=x,y=y,method='linear')
    nc["smb"] = smb.interp(x=x,y=y,method='linear')

    nc.to_netcdf(params.csmb_path_ncfile[:-3]+"_smb.nc",mode="w",format="NETCDF4")

def update(params, state):
    pass

def finalize(params, state):
    pass

def shift_coords(ds):
    ds_new = ds.copy()
    dx = ds_new.x[1] - ds_new.x[0]
    ds_new = ds_new.assign_coords({"x":ds.x-dx,"y":ds.y+dx})
    return ds_new