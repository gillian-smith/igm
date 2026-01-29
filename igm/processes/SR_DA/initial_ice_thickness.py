import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter


def initial_thickness(
    s,                 # surface elevation [m], shape (Ny, Nx)
    u, v,              # surface velocities (same shape), units set by speed_units
    mask,              # boolean ice mask (True = ice)
    dx, dy,            # grid spacing [m]
    *,
    speed_units="m/yr",   # "m/s" or "m/yr"
    A=1e-24,              # Pa^-3 s^-1 (tune 1e-24 .. 5e-24)
    n=3,
    rho=917.0, g=9.81,
    kappa=0.8,            # surface->depth-avg factor
    slope_floor=1e-3,     # dimensionless slope floor
    U0=100.0,             # m/yr threshold for blending
    L=5000.0,             # [m] distance length scale
    smooth_sigma_cells=2, # Gaussian sigma in grid cells
    Hmin=10.0, Hmax=3000.0
):
    s = np.asarray(s); u = np.asarray(u); v = np.asarray(v); mask = np.asarray(mask, dtype=bool)
    Ny, Nx = s.shape

    # --- speed magnitude
    speed = np.hypot(u, v)
    if speed_units.lower() in ["m/yr","m/years","m/year"]:
        speed = speed / (365.25*24*3600.0)
    elif speed_units.lower() not in ["m/s","mps"]:
        raise ValueError("speed_units must be 'm/s' or 'm/yr'.")

    # --- surface slope |∇s|
    ds_dy, ds_dx = np.gradient(s, dy, dx)  
    slope = np.hypot(ds_dx, ds_dy)
    slope = np.maximum(slope, slope_floor)

    # --- SIA-based thickness (depth-avg speed ~ k H^{n+1} slope^n)
    k = (2.0*A/(n+2.0)) * (rho*g)**n
    H_sia = np.power(np.maximum(kappa*speed, 0.0) / (k*np.power(slope, n)), 1.0/(n+1.0))
    H_sia = np.clip(H_sia, Hmin, Hmax)
    H_sia[~mask] = 0.0

    # distance inside ice to nearest non-ice
    d = distance_transform_edt(mask, sampling=(dy, dx))
    sia_vals = H_sia[mask]
    beta = np.percentile(sia_vals[sia_vals>0], 90) if np.any(sia_vals>0) else 500.0
    H_dist = beta * (1.0 - np.exp(-d / L))
    H_dist = np.clip(H_dist, Hmin, Hmax)

    # --- Blend by speed reliability
    U0_mps = U0 / (365.25*24*3600.0)  # convert to m/s
    w = speed / (speed + U0_mps)
    H0 = w*H_sia + (1.0 - w)*H_dist

    # --- Smooth & clamp
    if smooth_sigma_cells and smooth_sigma_cells > 0:
        H0 = gaussian_filter(H0, sigma=smooth_sigma_cells, mode="nearest")
    H0 = np.clip(H0, Hmin, Hmax)
    H0[~mask] = 0.0
    return H0
