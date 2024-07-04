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
        "--cthk_path_ncfile",
        type=str,
        default="input_saved.nc",
        help="Path to netCDF file created by oggm_shop"
    )
    parser.add_argument(
        "--cthk_rgi_folder",
        type=str,
        default="/home/s1639117/Documents/igm_folder/RGI/RGI2000-v7.0-G-15_south_asia_east", # TODO: test with oggm_shop downloaded RGI folder
        help="Filepath to RGI outline/s for this glacier/the whole region"
    )
    parser.add_argument(
        "--cthk_path_thkobs",
        type=str,
        default="/home/s1639117/Documents/igm_folder/hamish_data/Nepal2019_24Jan24_icethick_heli_and_ground_merge",
        help="Filepath to thkobs dataframe"
    )
    # Parameters like oggm_RGI_ID are also used, for this reason you must run custom_thkobs AFTER oggm_shop
    # You must have oggm_save_in_ncdf=True
    # If intending to use individual RGI outline as downloaded from oggm_shop you MUST set oggm_remove_RGI_folder=False

def initialize(params, state):

    import geopandas as gpd
    import xarray as xr
    import shapely
    #from pyproj import Transformer
    from scipy.interpolate import RectBivariateSpline

    nc = xr.open_dataset("input_saved.nc")

    rgi_folder = params.cthk_rgi_folder    
    if rgi_folder=="/home/s1639117/Documents/igm_folder/RGI/RGI2000-v7.0-G-15_south_asia_east":
        rgi_filepath = rgi_folder
    else: # we use individual RGI outline as downloaded from oggm_shop
        # TODO: if oggm_remove_RGI_folder is True throw an error
        rgi_folder = params.oggm_RGI_ID
        os.system('mkdir ' + rgi_folder + '/outlines');
        os.system('tar -xvzf '+ rgi_folder + '/outlines.tar.gz -C ' + rgi_folder + '/outlines');
        rgi_filepath = rgi_folder + "/outlines"
    
    rgi = gpd.read_file(rgi_filepath)
    rgi = rgi.to_crs(32645) # TODO: this crs is UTM45N for Everest area, may need to generalise for other regions

    outline = rgi[rgi["rgi_id"]==params.oggm_RGI_ID]
    outline.geometry = shapely.force_2d(outline.geometry)

    # x and y coordinates

    x = np.squeeze(nc.variables["x"]).astype("float32").values
    #y = np.flip(np.squeeze(nc.variables["y"]).astype("float32"))
    y = np.squeeze(nc.variables["y"]).astype("float32").values
    #x = state.x
    #y = state.y

    # surface elevation
    #usurf = state.usurf
    usurf = np.flipud(np.squeeze(nc.variables["usurf"]).astype("float32"))
    # interpolated surface elevation
    fsurf = RectBivariateSpline(x, y, np.transpose(usurf))

    # load thickness observations from file
    df = gpd.read_file(params.cthk_path_thkobs)
    df.geometry = shapely.force_2d(df.geometry)
    df = df.to_crs(outline.crs)

    # spatial join - find data points inside outline
    df = gpd.sjoin(df,outline)

    xx = df.geometry.x
    yy = df.geometry.y

    #bedrock = df["surfaceDEM"] - df["thick_m"]
    #elevation_normalized = fsurf(xx, yy, grid=False)
    #thickness_normalized = np.maximum(elevation_normalized - bedrock, 0)
    thickness_normalized = df["thick_m"].copy() 
    # TODO: look into whether thickness needs to be normalized or not - where do DEM values come from
    # TODO: param for if thickness column was called something other than "thick_m"

    # Rasterize thickness
    thickness_gridded = (
    pd.DataFrame(
        {
            "col": np.floor((xx - np.min(x)) / (x[1] - x[0])).astype(int),
            "row": np.floor((yy - np.min(y)) / (y[1] - y[0])).astype(int),
            "thickness": thickness_normalized,
        }
    )
    .groupby(["row", "col"])["thickness"]
    .mean() # mean over each grid cell
    )

    thkobs = np.full((y.shape[0], x.shape[0]), np.nan) # fill array with nans

    thickness_gridded[thickness_gridded == 0] = np.nan # nans where we have zero thickness / no observations

    thkobs[tuple(zip(*thickness_gridded.index))] = thickness_gridded

    thkobs_xr = xr.DataArray(thkobs,coords={'y':y,'x':x},attrs={'long_name':"Ice Thickness",'units':"m",'standard_name':"thkobs"})

    nc["thkobs"] = thkobs_xr

    #vars(state)["thkobs"] = tf.Variable(thkobs.astype("float32"))

    nc.to_netcdf("input_saved_with_thkobs.nc",mode="w",format="NETCDF4")

def update(params, state):
    pass

def finalize(params, state):
    pass