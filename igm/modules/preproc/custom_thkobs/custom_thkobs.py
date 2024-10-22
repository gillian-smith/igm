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
        default="", # TODO: test with oggm_shop downloaded RGI folder
        help="Filepath to RGI outline/s for this glacier/the whole region"
    )
    parser.add_argument(
        "--cthk_path_thkobs",
        type=str,
        #default="/home/s1639117/Documents/igm_folder/hamish_data/Nepal2019_24Jan24_icethick_heli_and_ground_merge",
        default="/home/s1639117/Documents/igm_folder/DATA/Nepal_survey_thickness_final/Nepal2019_new_GS.shp",
        help="Filepath to thkobs dataframe"
    )
    parser.add_argument(
        "--cthk_thkobs_column",
        type=str,
        default="thick_m",
        help="Column in dataframe where thkobs are stored"
    )
    parser.add_argument(
        "--cthk_thkobs_find_method",
        type=str,
        default="outline", # 'outline' or 'RGIId'
        help="Method for finding thkobs for this glacier - RGI outline or RGI ID"
    )
    parser.add_argument(
        "--cthk_profiles_to_use",
        type=str,
        default=[], # empty list means use all profiles, otherwise we can pass a list e.g. ["A","B","C"] to only use those profiles 
        help="Profiles to use as thkobs"
    )
    # Always run ["oggm_shop", "custom_thkobs"]
    # Then ["load_ncdf"] with "lncd_input_file" : "input_saved_with_thkobs.nc"
    # You must have oggm_save_in_ncdf=True
    # If intending to use individual RGI outline as downloaded from oggm_shop you MUST set oggm_remove_RGI_folder=False

def initialize(params, state):

    import geopandas as gpd
    import xarray as xr
    import shapely
    #from pyproj import Transformer
    from scipy.interpolate import RectBivariateSpline

    nc = xr.open_dataset(params.cthk_path_ncfile)

    rgi_folder = params.cthk_rgi_folder    
    
    if rgi_folder == "": # we use individual RGI outline as downloaded from oggm_shop
        # TODO: if oggm_remove_RGI_folder is True throw an error
        rgi_folder = params.oggm_RGI_ID
        os.system('mkdir ' + rgi_folder + '/outlines');
        os.system('tar -xvzf '+ rgi_folder + '/outlines.tar.gz -C ' + rgi_folder + '/outlines');
        rgi_filepath = rgi_folder + "/outlines"
        rgi = gpd.read_file(rgi_filepath)
        # crs should already be local one
    else: # usually import the file for the entire RGI region
        rgi_filepath = rgi_folder 
        rgi = gpd.read_file(rgi_filepath)        
        rgi = rgi.to_crs(32645) # TODO: this crs is UTM45N for Everest area, may need to generalise for other regions

    if params.oggm_RGI_version==6:
        outline = rgi[rgi["RGIId"]==params.oggm_RGI_ID]
    else: # RGI 7
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
    #fsurf = RectBivariateSpline(x, y, np.transpose(usurf))

    # load thickness observations from file
    df = gpd.read_file(params.cthk_path_thkobs)
    df.geometry = shapely.force_2d(df.geometry)
    df = df.to_crs(outline.crs)

    if params.cthk_thkobs_find_method=="outline":
        # spatial join - find data points inside outline
        df = gpd.sjoin(df,outline)
    elif params.cthk_thkobs_find_method=="RGIId":
        # find points with same RGI ID as this glacier (even if they are outside the outline)
        df = df[df["RGIId"]==params.oggm_RGI_ID]

    # Filter by profiles we are interested in
    if params.cthk_profiles_to_use: # is not the empty list
        df = df[df["profile_id"].str.endswith(tuple(params.cthk_profiles_to_use))]

    xx = df.geometry.x
    yy = df.geometry.y

    #bedrock = df["surfaceDEM"] - df["thick_m"]
    #elevation_normalized = fsurf(xx, yy, grid=False)
    #thickness_normalized = np.maximum(elevation_normalized - bedrock, 0)
    thickness_normalized = df[params.cthk_thkobs_column].copy() 
    # TODO: look into whether thickness needs to be normalized or not - where do DEM values come from

    # Rasterize
    gridded = (
    pd.DataFrame(
        {
            "col": np.floor((xx - np.min(x)) / (x[1] - x[0])).astype(int),
            "row": np.floor((yy - np.min(y)) / (y[1] - y[0])).astype(int),
            "thickness": thickness_normalized,
        }
    )
    .groupby(["row", "col"])["thickness"])

    # Thickness - mean over each grid cell
    thickness_gridded = gridded.mean() # mean over each grid cell
    thkobs = np.full((y.shape[0], x.shape[0]), np.nan) # fill array with nans
    thickness_gridded[thickness_gridded == 0] = np.nan # put nans where we have zero thickness / no observations
    thkobs[tuple(zip(*thickness_gridded.index))] = thickness_gridded
    thkobs_xr = xr.DataArray(thkobs,coords={'y':y,'x':x},attrs={'long_name':"Ice Thickness",'units':"m",'standard_name':"thkobs"})
    nc["thkobs"] = thkobs_xr

    # Std dev of each grid cell
    thickness_stdev_gridded = gridded.std() # mean over each grid cell
    thkobs_stdev = np.full((y.shape[0], x.shape[0]), np.nan) # fill array with nans
    thickness_stdev_gridded[thickness_stdev_gridded == 0] = np.nan # put nans where we have zero thickness / no observations
    thkobs_stdev[tuple(zip(*thickness_stdev_gridded.index))] = thickness_stdev_gridded
    thkobs_stdev_xr = xr.DataArray(thkobs_stdev,coords={'y':y,'x':x},attrs={'long_name':"Ice Thickness standard deviation",'units':"m",'standard_name':"thkobs_std"})
    nc["thkobs_std"] = thkobs_stdev_xr

    # Count of thkobs in each grid cell
    thickness_count_gridded = gridded.count() # mean over each grid cell
    thkobs_count = np.full((y.shape[0], x.shape[0]), np.nan) # fill array with nans
    thickness_count_gridded[thickness_count_gridded == 0] = np.nan # put nans where we have zero thickness / no observations
    thkobs_count[tuple(zip(*thickness_count_gridded.index))] = thickness_count_gridded
    thkobs_count_xr = xr.DataArray(thkobs_count,coords={'y':y,'x':x},attrs={'long_name':"Ice Thickness count",'units':"none",'standard_name':"thkobs_count"})
    nc["thkobs_count"] = thkobs_count_xr

    #vars(state)["thkobs"] = tf.Variable(thkobs.astype("float32"))

    nc.to_netcdf(params.cthk_path_ncfile[:-3]+"_thkobs.nc",mode="w",format="NETCDF4")
    os.system( "echo rm -r " + params.cthk_path_ncfile + " >> clean.sh" )

def update(params, state):
    pass

def finalize(params, state):
    pass