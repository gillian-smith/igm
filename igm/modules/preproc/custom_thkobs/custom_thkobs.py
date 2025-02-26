#!/usr/bin/env python3

# Copyright (C) 2024 Gillian Smith <g.smith@ed.ac.uk>
# Published under the GNU GPL (Version 3), check at the LICENSE file

import numpy as np
import matplotlib.pyplot as plt
import os, glob, shutil, scipy
from netCDF4 import Dataset
import tensorflow as tf
import pandas as pd
import xarray as xr

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
        help="Method for finding thkobs for this glacier - RGI 6/7 outline or RGI 6 ID"
    )
    parser.add_argument(
        "--cthk_profiles_constrain",
        type=str,
        default=[], # a list e.g. ["A","B","C"]
        help="Profiles to use as thkobs cost constraint"
    )
    parser.add_argument(
        "--cthk_profiles_test",
        type=str,
        default=[], # a list e.g. ["D","E","F"]
        help="Profiles to use as thkobs test"
    )
    parser.add_argument(
        "--cthk_exclude_test_cells_from_constraint",
        type=str2bool,
        default=True,
        help="Exclude cells in thkobs_test from thkobs to ensure no intersection"
    )
    parser.add_argument(
        "--cthk_shift_thkobs",
        type=str2bool,
        default=True,
        help="Shift thkobs cells up and to the right",
    )
    parser.add_argument(
        "--cthk_mask_thkobs",
        type=str2bool,
        default=True,
        help="Exclude thkobs cells outside icemask",
    )
    # Always run ["oggm_shop", "custom_thkobs"]
    # Then ["load_ncdf"] with "lncd_input_file" : "input_saved_with_thkobs.nc"
    # You must have oggm_save_in_ncdf=True
    # If intending to use individual RGI outline as downloaded from oggm_shop you MUST set oggm_remove_RGI_folder=False

def initialize(params, state):

    import geopandas as gpd
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

    #profiles = list(map(lambda x: x[-1], df["profile_id"].unique().tolist()))
    profiles_constrain = params.cthk_profiles_constrain
    profiles_test = params.cthk_profiles_test

    # if not profiles_test: # profiles_test == []
    #     # no test set specified, use all profiles for constraint
    #     profiles_constrain = profiles
    # elif not profiles_constrain: # profiles_constrain == []
    #     # no constraining set specified, use all profiles not already in test set
    #     # think this would even work if profiles_test == []?
    #     profiles_constrain = [profile for profile in profiles if profile not in profiles_test]

    # TODO this assumes every profile is identified by one character - allow for multiple characters by splitting on "_"
    df_constrain = df[df["profile_id"].str.strip().str[-1].isin(profiles_constrain)]
    df_test = df[df["profile_id"].str.strip().str[-1].isin(profiles_test)]
    
    # TODO test case where either df empty
    if df_constrain.empty:
        print("No profiles for constraint")
        nc["thkobs"] = xr.full_like(nc["thkinit"],np.nan)
        nc["thkobs_std"] = xr.full_like(nc["thkinit"],np.nan)
        nc["thkobs_count"] = xr.full_like(nc["thkinit"],np.nan)
    else:
        nc["thkobs"]     , nc["thkobs_std"], nc["thkobs_count"] = rasterize(params,df_constrain,x,y,params.cthk_thkobs_column)
    
    if df_test.empty:
        print("No profiles for test")
        nc["thkobs_test"] = xr.full_like(nc["thkinit"],np.nan)
    else:
        nc["thkobs_test"], _               , _                  = rasterize(params,df_test,x,y,params.cthk_thkobs_column)
    
    if params.cthk_exclude_test_cells_from_constraint:
        # Exclude thkobs_test cells from thkobs
        MASK = nc["thkobs_test"].isnull() & ~nc["thkobs"].isnull()
        nc["thkobs"] = xr.where(MASK, nc["thkobs"], np.nan)

    # Exclude cells outside outline
    if params.cthk_mask_thkobs:
        nc["thkobs"] = xr.where(nc["icemaskobs"], nc["thkobs"], np.nan)

    #vars(state)["thkobs"] = tf.Variable(thkobs.astype("float32"))

    count_cells_constraining = np.count_nonzero(~np.isnan(nc["thkobs"].values))
    count_cells_test = np.count_nonzero(~np.isnan(nc["thkobs_test"].values))

    print(f"# Grid cells in constraining set = {count_cells_constraining}")
    print(f"# Grid cells in test set = {count_cells_test}")

    nc.to_netcdf(params.cthk_path_ncfile[:-3]+"_thkobs.nc",mode="w",format="NETCDF4")
    os.system( "echo rm -r " + params.cthk_path_ncfile + " >> clean.sh" )

def update(params, state):
    pass

def finalize(params, state):
    pass
    #TODO remove input_saved.nc

def rasterize(params,df,x,y,thkobs_column):

    xx = df.geometry.x
    yy = df.geometry.y

    #bedrock = df["surfaceDEM"] - df["thick_m"]
    #elevation_normalized = fsurf(xx, yy, grid=False)
    #thickness_normalized = np.maximum(elevation_normalized - bedrock, 0)
    thickness_normalized = df[thkobs_column].copy() 
    #thickness_normalized = df["thick_m"].copy() 
    # TODO: look into whether thickness needs to be normalized or not - where do DEM values come from

    dx = x[1]-x[0]
    dy = y[1]-y[0]

    # Rasterize
    gridded = (
    pd.DataFrame(
        {
            "col": np.floor((xx - np.min(x) + dx/2) / dx).astype(int),
            "row": np.floor((yy - np.min(y) + dy/2) / dy).astype(int),
            "thickness": thickness_normalized,
        }
    )
    .groupby(["row", "col"])["thickness"])

    # TODO lots of repetition of code here - use function if possible?

    # Thickness - mean over each grid cell
    thickness_gridded = gridded.mean() # mean over each grid cell
    thkobs = np.full((y.shape[0], x.shape[0]), np.nan) # fill array with nans
    thickness_gridded[thickness_gridded == 0] = np.nan # put nans where we have zero thickness / no observations
    thkobs[tuple(zip(*thickness_gridded.index))] = thickness_gridded
    thkobs_xr = xr.DataArray(thkobs,coords={'y':y,'x':x},attrs={'long_name':"Ice Thickness",'units':"m",'standard_name':"thkobs"})

    # Std dev of each grid cell
    thickness_stdev_gridded = gridded.std() # mean over each grid cell
    thkobs_stdev = np.full((y.shape[0], x.shape[0]), np.nan) # fill array with nans
    thickness_stdev_gridded[thickness_stdev_gridded == 0] = np.nan # put nans where we have zero thickness / no observations
    thkobs_stdev[tuple(zip(*thickness_stdev_gridded.index))] = thickness_stdev_gridded
    thkobs_stdev_xr = xr.DataArray(thkobs_stdev,coords={'y':y,'x':x},attrs={'long_name':"Ice Thickness standard deviation",'units':"m",'standard_name':"thkobs_std"})

    # Count of thkobs in each grid cell
    thickness_count_gridded = gridded.count() # mean over each grid cell
    thkobs_count = np.full((y.shape[0], x.shape[0]), np.nan) # fill array with nans
    thickness_count_gridded[thickness_count_gridded == 0] = np.nan # put nans where we have zero thickness / no observations
    thkobs_count[tuple(zip(*thickness_count_gridded.index))] = thickness_count_gridded
    thkobs_count_xr = xr.DataArray(thkobs_count,coords={'y':y,'x':x},attrs={'long_name':"Ice Thickness count",'units':"none",'standard_name':"thkobs_count"})

    return thkobs_xr, thkobs_stdev_xr, thkobs_count_xr