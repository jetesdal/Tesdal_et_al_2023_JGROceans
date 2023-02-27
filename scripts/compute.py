
import xarray as xr
import xgcm
import warnings

def zonal_mean(da, metrics):
    num = (da * metrics['areacello'] * metrics['wet']).sum(dim=['x'])
    denom = (da/da * metrics['areacello'] * metrics['wet']).sum(dim=['x'])
    return num/denom

def detrend_dim(da, dim, deg=1):
    # detrend along a single dimension
    p = da.polyfit(dim=dim, deg=deg)
    fit = xr.polyval(da[dim], p.polyfit_coefficients)
    return da - fit


def detrend(da, dims, deg=1):
    # detrend along multiple dimensions
    # only valid for linear detrending (deg=1)
    da_detrended = da
    for dim in dims:
        da_detrended = detrend_dim(da_detrended, dim, deg=deg)
    return da_detrended

def horizontal_mean(da, ds):
    num = (da * ds['areacello'] * ds['wet']).sum(dim=['x', 'y'])
    denom = (da/da * ds['areacello'] * ds['wet']).sum(dim=['x', 'y'])
    return num / denom

def mean_bottom_age(ds,var,depth):
    num = (ds[var]*ds.areacello*ds.wet).where(ds.deptho>depth).sum(dim=['x','y'])
    denom = (ds.areacello * ds.wet).where(ds.deptho>depth).sum(dim=['x','y'])
    return num/denom


def get_xgcm_grid(ds,ds_grid,grid='z',metrics=True, **kwargs):
    """
    Get the xgcm grid object from non-static and static grid information.
    
    Parameters
    ----------
    ds : xarray.Dataset
        Contains non-static grid information (e.g., volcello)
    ds_grid : xarray.Dataset
        Contains static grid information (e.g., dxt, dyt)
    metrics : boolean, optional
        
    Returns
    -------
    An object with multiple :class:`xgcm.Axis` objects representing different
    independent axes
    """
    
    # Copy static grid dataset
    ds_g = ds_grid.copy()
    
    # Add vertical coordinate
    ds_g['lev'] = ds['lev']
    ds_g['lev_outer'] = ds['lev_outer']
    
    # Define coordinates
    coords = {'X': {'center': 'x', 'right': 'xq'},
              'Y': {'center': 'y', 'right': 'yq'},
              'Z': {'center': 'lev', 'outer': 'lev_outer'}
             }
    
    if metrics:
        
        # Define a nominal layer thickness
        ds_g['dzt'] = xr.DataArray(data=ds['lev_outer'].diff('lev_outer').values, 
                                   coords={'lev': ds['lev']}, dims=('lev'))
        
        # Replace all NaNs with zeros
        ds_g['dxt'] = ds_g['dxt'].fillna(0.)
        ds_g['dyt'] = ds_g['dyt'].fillna(0.)
        ds_g['dzt'] = ds_g['dzt'].fillna(0.)
        ds_g['areacello'] = ds_g['areacello'].fillna(0.)
        
        metrics={('X',): ['dxt'], ('Y',): ['dyt'], ('Z',): ['dzt'], ('X', 'Y'): ['areacello']}
        
        # The volume of the cell can be different from areacello * dzt 
        # We need to add volcello to the metrics
        if "volcello" in ds.keys():
            ds_g['volcello'] = ds['volcello'].fillna(0.)
            metrics[('X', 'Y', 'Z')] = ['volcello']
        else:
            warnings.warn("'volcello' is missing")
            
        xgrid = xgcm.Grid(ds_g, coords=coords, metrics=metrics, **kwargs)
        
    else:
        xgrid = xgcm.Grid(ds_g, coords=coords, **kwargs)

    return xgrid


def get_xgcm_grid_mom6(ds,ds_grid,grid='z',metrics=True, **kwargs):
    """
    Get the xgcm grid object from non-static and static grid information.
    Specfic naming as in MOM6 (e.g., CM4 or ESM4).

    Parameters
    ----------
    ds : xarray.Dataset
        Contains non-static grid information (e.g., volcello)
    ds_grid : xarray.Dataset
        Contains static grid information (e.g., dxt, dyt)
    grid : str
        Specifies the diagnostic grid ['native','z','rho2']
    metrics : boolean, optional
        
    Returns
    -------
    An object with multiple :class:`xgcm.Axis` objects representing different
    independent axes
    """
    
    # Copy static grid dataset
    ds_g = ds_grid.copy()

    # Specify vertical index name in grid (center '_l' and outer '_i')
    vertind = {'native':['zl','zi'],'z':['z_l','z_i'],'rho2':['rho2_l','rho2_i']}
    
    # Add vertical coordinate
    ds_g[vertind[grid][0]] = ds[vertind[grid][0]]
    ds_g[vertind[grid][1]] = ds[vertind[grid][1]]
    
    # Define coordinates
    coords = {'X': {'center': 'xh', 'right': 'xq'},
              'Y': {'center': 'yh', 'right': 'yq'},
              'Z': {'center': vertind[grid][0], 'outer': vertind[grid][1]}
             }
    
    if metrics:
        
        # Define a nominal layer thickness
        ds_g['dzt'] = xr.DataArray(data=ds[vertind[grid][1]].diff(vertind[grid][1]).values, 
                                   coords={vertind[grid][0]: ds[vertind[grid][0]]}, dims=(vertind[grid][0]))
        
        # Replace all NaNs with zeros
        ds_g['dxt'] = ds_g['dxt'].fillna(0.)
        ds_g['dyt'] = ds_g['dyt'].fillna(0.)
        ds_g['dzt'] = ds_g['dzt'].fillna(0.)
        ds_g['areacello'] = ds_g['areacello'].fillna(0.)
        
        metrics={('X',): ['dxt'], ('Y',): ['dyt'], ('Z',): ['dzt'], ('X', 'Y'): ['areacello']}
        
        # The volume of the cell can be different from areacello * dzt 
        # We need to add volcello to the metrics
        if "volcello" in ds.keys():
            ds_g['volcello'] = ds['volcello'].fillna(0.)
            metrics[('X', 'Y', 'Z')] = ['volcello']
        else:
            warnings.warn("'volcello' is missing")
            
        xgrid = xgcm.Grid(ds_g, coords=coords, metrics=metrics, **kwargs)
        
    else:
        xgrid = xgcm.Grid(ds_g, coords=coords, **kwargs)

    return xgrid
