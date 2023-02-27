import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy
import matplotlib.path as mpath
import matplotlib.ticker as mticker
from matplotlib.offsetbox import AnchoredText
import calendar
from compute import zonal_mean
import shapely
#import geopandas as gpd
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable

def hovmoeller(da, cmap='RdBu_r', vrange= [None, None], xlimrange = [None, None], ylimrange = [None, None],
               fsize = [15, 7], ylabel = '', xlabel='', title='', cb_label = '',
               xticks=[], yticks=[], tpls = 12, axlfs = 14, tfs = 16, cblfs = 14, 
               hline = None, vline = None, **kwargs):
    
    fig, ax = plt.subplots(figsize=fsize)
    p = da.T.plot(ax=ax,vmin=vrange[0], vmax=vrange[1], cmap=cmap, add_colorbar=False,**kwargs)
    ax.set_xlim([xlimrange[0], xlimrange[1]])
    ax.set_ylim([ylimrange[0], ylimrange[1]])
    ax.tick_params(axis='both', which='major', labelsize=tpls)
    ax.set_ylabel(ylabel,fontsize=axlfs)
    ax.set_xlabel(xlabel,fontsize=axlfs)
    ax.set_title(title,fontsize=tfs)
    if len(xticks)>0:
        ax.set_xticks(xticks)
    if len(yticks)>0:
        ax.set_yticks(yticks)
        
    if hline:
        for val in hline:
            ax.axhline(y=val, xmin=0, xmax=1, c = 'k', lw=1.0, ls='--')
    if vline:
        for val in vline:
            ax.axvline(x=val, ymin=0, ymax=1, c = 'k', lw=1.0, ls='--')

    cb = plt.colorbar(p, orientation='vertical', extend='both', pad=0.02)
    cb.set_label(label=cb_label, fontsize=cblfs, weight='bold')
    cb.ax.tick_params(labelsize='large')
    return fig, ax

def hovmoeller_clim(da, vrange, cb_label = '', fsize = [10, 5], **kwargs):
    fig, ax = plt.subplots(figsize=fsize)
    da.T.plot(ax=ax, cmap='RdBu_r', vmin=vrange[0], vmax=vrange[1], add_labels=False,
              cbar_kwargs={'label': cb_label}, **kwargs)
    ax.set_xticks(np.arange(1,13))
    plt.setp(ax, 'xticklabels',calendar.month_abbr[1:13])
    return fig

def wmt_plot(G_exp, G_ctrl, G_ctr_ann, var = 'total', panel_txt = 'Total over the Antarctic shelf',
             fsize=[10,5], add_legend=False, ylimrange=[None,None],xticks=[], yticks=[], 
             lcn = 1, lloc='lower right',lfs = 14, **kwargs):

    fig, ax = plt.subplots(figsize=fsize)
    ax.axhline(y=0, xmin=0, xmax=1, linewidth=1.0, color = 'k')
    ax.fill_between(G_ctrl.sigma1,
                    (G_ctrl[var+'_mean']-G_ctrl[var+'_std'])*1e-6,
                    (G_ctrl[var+'_mean']+G_ctrl[var+'_std'])*1e-6,
                    alpha=0.1, edgecolor='k', facecolor='k')

    G_ctr_ann.mean('year')[var].plot(ax=ax,lw=2,c='k',label='Control (0251-0420)',_labels=False)
    G_ctr_ann.sel(year=slice(385,394)).mean('year')[var].plot(ax=ax,c='k', lw=2, ls='--',alpha=0.5,
                                                               label='Control (0385-0394)',_labels=False)
    G_ctr_ann.sel(year=slice(333,342)).mean('year')[var].plot(ax=ax,c='k', lw=2, ls=':',alpha=0.5,
                                                               label='Control (0333-0342, polynya)',_labels=False)
    ax.plot(G_exp.sigma1, G_exp[var+'_antw']*1e-6, color='b', linestyle='-', lw=3, label='Antwater (0051-0071)')
    ax.plot(G_exp.sigma1, G_exp[var+'_strs']*1e-6, color='r', linestyle='-', lw=3, label='Stress (0051-0071)')
    ax.plot(G_exp.sigma1, G_exp[var+'_anst']*1e-6, color='g', linestyle='-', lw=3,
            label='Antwater-Stress (0051-0071)')
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.set_xlim([31.1,33.1])
    ax.set_ylim([ylimrange[0],ylimrange[1]])

    if len(xticks)>0:
        ax.set_xticks(xticks)
    if len(yticks)>0:
        ax.set_yticks(yticks)
    ax.set_ylabel('Mean transformation [Sv]',fontsize=16)
    ax.text(0.01, 0.97, panel_txt, horizontalalignment='left', verticalalignment='top',
            transform=ax.transAxes, fontsize=16, fontweight='bold')
    if add_legend:
        ax.legend(frameon=False, fancybox=False, loc=lloc, ncol=lcn, fontsize=lfs, **kwargs)

    return fig


def paraplot(y1,y2,y3,ylabel,llabel,fsize=[12,5],zero_line=False,add_legend=False,
             xlimrange=[None,None],ylimrange=[None,None],xticks=[], yticks=[]):
    '''
    Parasite axis plot. Plots three variables with three independent y axes onto one single plot.
    Function uses twinx meaning variables are tied directly to the x-axis.
    From there, each y axis can behave separately from each other.
    Each can take on separate values from themselves as well as the x-axis.
    
    '''
    fig, host = plt.subplots(figsize=fsize)
    fig.subplots_adjust(right=0.75)
    
    par1 = host.twinx()
    par2 = host.twinx()
    par2.spines['right'].set_position(('axes', 1.15))
    
    if zero_line:
        host.axhline(y=0, xmin=0, xmax=1, linewidth=2, color = 'k',zorder=1)
        
    p2 = par1.plot(y2.year, y2, label=llabel[1], lw=2, c='r')
    p3 = par2.plot(y3.year, y3, label=llabel[2], lw=2, c='b')
    p1 = host.plot(y1.year, y1, label=llabel[0], lw=2, c='k')
    
    # Set y axis limits
    #host.set_ylim([ylimrange[0],ylimrange[1]])
    #par1.set_ylim([ylimrange[0],ylimrange[1]])
    #par2.set_ylim([ylimrange[0],ylimrange[1]])
    
    # Format y axis
    tkw = dict(size=5, width=1.5, labelsize=12)
    host.tick_params(axis='y', colors='k', **tkw)
    par1.tick_params(axis='y', colors='k', **tkw)
    par2.tick_params(axis='y', colors='k', **tkw)
    host.tick_params(axis='x', **tkw)
    
    # Set x axis limit
    host.set_xlim([xlimrange[0],xlimrange[1]])
    par1.set_xlim([xlimrange[0],xlimrange[1]])
    par2.set_xlim([xlimrange[0],xlimrange[1]])
    
    # Format x axis
    if len(xticks)>0:
        host.set_xticks(xticks)
        
    # Lable y axes
    host.set_ylabel(ylabel[0], fontsize=12, fontweight='bold', color='k')
    par1.set_ylabel(ylabel[1], fontsize=12, fontweight='bold', color='r')
    par2.set_ylabel(ylabel[2], fontsize=12, fontweight='bold', color='b')
    
    if add_legend:
        host.legend([p1,p2,p3],[llabel[0],llabel[1],llabel[2]], frameon=False, fancybox=False,
                    bbox_to_anchor=(0.5, 1.13),loc='upper center', ncol=3, fontsize=14)
    return fig

def corrplot(X, Y, xlabel='', ylabel='', title='',fsize=[10,5], 
             zero_line=True, hline = None, vline = None, add_legend=True, drawGrid=True,
             lcn = 1, lloc='lower right',lfs = 14, **kwargs):
    
    fig, ax = plt.subplots(figsize=fsize)
    
    if zero_line:
        ax.axhline(y=0, c = 'k', ls='-', lw=1)
        
    # Shelf
    xr.corr(Y[0].onshlf, X[0].sel(year=Y[0].year),dim='year').plot(ax=ax, lw=2, _labels=False, label='Shelf (CM4)')
    xr.corr(Y[1].onshlf, X[1].sel(year=Y[1].year),dim='year').plot(ax=ax, lw=2, _labels=False, label='Shelf (ESM4)')
    
    # Off-shore
    xr.corr(Y[0].offshlf, X[0].sel(year=Y[0].year),dim='year').plot(ax=ax,lw=2,ls='--',c='tab:blue',
                                                                    _labels=False, label='Open ocean (CM4)')
    xr.corr(Y[1].offshlf, X[1].sel(year=Y[1].year),dim='year').plot(ax=ax,lw=2,ls='--',c='tab:orange',
                                                                    _labels=False, label='Open ocean (ESM4)')
    
    if hline:
        for val in hline:
            ax.axhline(y=val, xmin=0, xmax=1, c = 'k', lw=1.0, ls='--')
    if vline:
        for val in vline:
            ax.axvline(x=val, ymin=0, ymax=1, c = 'k', lw=1.0, ls='--')

    ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
    
    if drawGrid:
        ax.grid()
        
    if add_legend:
        ax.legend(frameon=True, fancybox=True, loc=lloc, ncol=lcn, fontsize=lfs, **kwargs)
        
    return fig


def create_topomask(ds_section):
    deptho= ds_section['deptho']
    y = deptho.y
    depth = np.where(np.isnan(deptho.to_masked_array()), 0.0, deptho)
    
    topomask = depth.max(axis=-1)
    
    _y = y
    _z = np.arange(0, 7100, 1)
    _yy, _zz = np.meshgrid(_y, _z)
    topomask = np.tile(topomask[None, :], (len(_z), 1))
    topomask = np.ma.masked_where(_zz < topomask, topomask)
    topomask = topomask * 0.0
    
    topomask = xr.DataArray(topomask, coords = [_z,_y], dims = ['lev','y'], name = 'topomask')
    return topomask


def zm_section(ds, var, varc=None, varc2=None, xslice=[None, None], yslice=[None, None],
               fsize = [10, 4], cmap='viridis',vrange= [None, None],
               xlimrange = [None, None], ylimrange = [None, None],ylimrange_top = [None, None],
               clevels=None, ccol = 'k', clw=2, clfmt='%1.3f', clf=12, cloc=None, clab=True,
               clevels2=None, ccol2 = 'k', clw2=2, clfmt2='%1.3f', clf2=12, cloc2=None, clab2=True,
               ylabel = '', xlabel='', title='', cb_label = '',
               tpls = 12, axlfs = 14, tfs = 16, cblfs = 14,
               add_top=False,da_top=None,topcol='k'):
    
    ds_section = ds.sel(x=slice(xslice[0],xslice[1]),y=slice(yslice[0],yslice[1]))
    topomask = create_topomask(ds_section)
    da = zonal_mean(ds_section[var], ds_section).squeeze()
    if varc:
        dac = zonal_mean(ds_section[varc], ds_section).squeeze()
    if varc2:
        dac2 = zonal_mean(ds_section[varc2], ds_section).squeeze()
    
    if add_top:
        fig = plt.figure(figsize=fsize)
        gs = gridspec.GridSpec(2, 1, height_ratios = [1,3],hspace=0.05)
        ax = plt.subplot(gs[1,0])
    else:
        fig, ax = plt.subplots(figsize=fsize)
    
    p = da.plot(ax=ax, yincrease=False, x='y', y='lev', cmap=cmap, vmin=vrange[0], vmax=vrange[1],
                add_colorbar=False, add_labels=False)
    if varc:
        cs = dac.plot.contour(ax=ax, yincrease=False, x='y', y='lev', 
                              levels=clevels, linewidths=clw, colors=ccol)
        if cloc:
            ax.clabel(cs, cs.levels, inline=True, fmt=clfmt, fontsize=clf, manual=cloc)
        elif clab:
            ax.clabel(cs, cs.levels, inline=True, fmt=clfmt, fontsize=clf)
        
    if varc2:
        cs = dac2.plot.contour(ax=ax, yincrease=False, x='y', y='lev', 
                               levels=clevels2, linewidths=clw2, colors=ccol2)
        if cloc2:
            ax.clabel(cs, cs.levels, inline=True, fmt=clfmt2, fontsize=clf2,manual=cloc2)
        elif clab2:
            ax.clabel(cs, cs.levels, inline=True, fmt=clfmt2, fontsize=clf2)
        
    topomask.plot(ax=ax,x='y', y='lev', yincrease=False, cmap='gray', shading='auto',
                  add_colorbar=False, add_labels=False)
    ax.set_xlim([xlimrange[0], xlimrange[1]])
    ax.set_ylim([ylimrange[0], ylimrange[1]])
    ax.tick_params(axis='both', which='major', labelsize=tpls)
    ax.set_ylabel(ylabel,fontsize=axlfs)
    ax.set_xlabel(xlabel,fontsize=axlfs)
    ax.set_title(title,fontsize=tfs)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad=.1)
    cb = plt.colorbar(p, cax=cax, orientation='vertical', extend='both', pad=0.02)
    cb.set_label(label=cb_label, fontsize=cblfs, weight='bold')
    cb.ax.tick_params(labelsize='large')
    
    if add_top:
        axt = plt.subplot(gs[0,0],sharex=ax)
        axt.axhline(y=0, xmin=0, xmax=1, linewidth=1.0, color = 'k')
        da_top.plot(ax=axt,c=topcol,lw=2,_labels=False)
        axt.set_xlim([xlimrange[0], xlimrange[1]])
        axt.set_ylim([ylimrange_top[0], ylimrange_top[1]])
        axt.tick_params(axis='both', which='major', labelsize=tpls)
        axt.tick_params(labelbottom=False)
        dividert = make_axes_locatable(axt)
        cax2 = dividert.append_axes("right", size="3%", pad=.1)
        cax2.remove()
    
    return fig, ax


def decomp_plot(G, title):
    '''
    Plot watermass transformation (WMT) decomposed by thermal and haline part and freshwater components
    
    Parameters
    ----------
    G : list [xarray.Dataset, xarray.Dataset]
        xarray datasets containing WMT for each component.
        First dataset includes the thermal and haline part.
        Second dataset incluces the different freshwater components.
    title : str
        Title for the plot
        
    Returns
    -------
    fig: Figure instance
    '''
    
    G_fw_comp = G[1][['total','rain_and_ice','snow']]\
                    .rename({'total':'Total FW','rain_and_ice':'Precip. + Sea ice','snow':'Snow'})
    G_fw_comp_minor = G[1][['evaporation','rivers','icebergs']]\
                        .rename({'evaporation':'Evaporation','rivers':'Runoff','icebergs':'Icebergs'})

    fig, ax = plt.subplots(figsize=(14,5))
    ax.axhline(y=0, xmin=0, xmax=1, linewidth=1.0, color = 'k')
    ax.plot(G[0]['sigma1'], (G[0].heat+G[0].salt)*1e-6, color='k', linestyle='-', lw=3, label='Total')
    ax.plot(G[0]['sigma1'], G[0].heat*1e-6, color='r', linestyle=':', lw=3, label='Thermal')
    ax.plot(G[0]['sigma1'], G[0].salt*1e-6, color='b', linestyle='--', lw=3, label='Haline')
    
    for var in G_fw_comp.keys():
        ax.plot(G_fw_comp.sigma1, G_fw_comp[var]*1e-6, lw=2, linestyle='-', label=var)
    
    for var, color in zip(G_fw_comp_minor.keys(), 'ycm'):
        ax.plot(G_fw_comp_minor.sigma1, G_fw_comp_minor[var]*1e-6, lw=1, ls='-', c=color, label=var)
    
    #ax.set_ylim([-47,33])
    ax.set_xlim([31,33.3])
    ax.set_xlabel(r'Potential density ($\sigma_1$) [kg m$^{-3}$ - 1000]',fontsize=14)
    ax.set_ylabel('Mean transformation [Sv]',fontsize=14)
    ax.grid(True)
    ax.set_title(title,fontsize=15,fontweight='bold')
    ax.legend(loc='center left',bbox_to_anchor=(1, 0.55))
    return fig

def comparison_map(v, grid, t, vmin, vmax, titles,cb_label=''):
    fig = plt.figure(figsize=[12,4])
    fig.subplots_adjust(left=0.05, right=0.95, bottom=0.1, top=0.9, hspace=0.01)
    
    ax1 = fig.add_subplot(1, 3, 1,facecolor='grey')
    v[0].isel(time=t).where(grid.wet==1).plot(ax=ax1,vmin=vmin, vmax=vmax, cmap='RdBu_r', 
                                                add_colorbar=False, add_labels=False)
    ax1.set_title(titles[0], fontsize=10)
    
    ax2 = fig.add_subplot(1, 3, 2,facecolor='grey')
    p = v[1].isel(time=t).where(grid.wet==1).plot(ax=ax2, vmin=vmin, vmax=vmax, cmap='RdBu_r', 
                                              add_colorbar=False, add_labels=False)
    ax2.set_title(titles[1], fontsize=10)
    
    pos2 = ax2.get_position()
    ax2.set_position([pos2.x0 - 0.025, pos2.y0, (pos2.x1 - pos2.x0), (pos2.y1 - pos2.y0)])
    cax = fig.add_axes([0.62, 0.1, 0.02, 0.8])
    cb = fig.colorbar(p, cax=cax, orientation='vertical', shrink=0.9, pad=0.02)
    cb.set_label(cb_label, fontsize=12)
    cb.ax.tick_params(labelsize=10)
    
    ax3 = fig.add_subplot(1, 3, 3,facecolor='grey')
    p = (v[0].isel(time=t)-v[1].isel(time=t)).where(grid.wet==1).plot(ax=ax3, robust=True, add_colorbar=False, 
                                                                        add_labels=False)
    ax3.set_title('Difference', fontsize=10)
    pos3 = ax3.get_position()
    ax3.set_position([pos3.x0 + 0.04, pos3.y0, (pos3.x1 - pos3.x0), (pos3.y1 - pos3.y0)])
    cax = fig.add_axes([1.0, 0.1, 0.02, 0.8])
    cb = fig.colorbar(p, cax=cax, orientation='vertical', shrink=0.9, pad=0.02)
    cb.set_label(cb_label, fontsize=12)
    cb.ax.tick_params(labelsize=10)
    plt.show()

def get_so_map_boundary():
    '''
    Compute a circle in axes coordinates, which we can use as a boundary
    for the map. We can pan/zoom as much as we like - the boundary will be
    permanently circular.
    '''
    # Parameters for cartopy map
    theta = np.linspace(0, 2*np.pi, 100)
    center, radius = [0.5, 0.5], 0.5
    verts = np.vstack([np.sin(theta), np.cos(theta)]).T
    circle = mpath.Path(verts * radius + center)
    return circle

def get_gdf_patch(coords_patch):
    # Replace each pair of points by 10 points for projected map
    geometry = []
    segments = zip(coords_patch[:-2:2], coords_patch[1:-1:2], coords_patch[2::2], 
                   coords_patch[3::2])
    
    for line in [shapely.geometry.LineString([(x1, y1), (x2, y2)]) for (x1,y1,x2,y2) in segments]:
        for length in np.linspace(0,1,11):
            geometry.append( line.interpolate(length, normalized=True) )
        
    gdf_patch = gpd.GeoDataFrame([], geometry=geometry)
    
    # Convert Points to Polygo
    gdf_patch['geometry'] = gdf_patch['geometry'].apply(lambda x: x.coords[0])
    
    gdf_patch['shape_id'] = 0
    gdf_patch = gdf_patch.groupby('shape_id')['geometry'].apply(lambda x: shapely.geometry.Polygon(x.tolist())).reset_index()
    gdf_patch = gpd.GeoDataFrame(gdf_patch, geometry = 'geometry')
    
    # Salem uses this attribute:
    gdf_patch.crs = {'init': 'epsg:4326'}
    
    return gdf_patch


def cartopy_map(ds, var, varc = None, lat_range = [-90,90], lon_range = [-180,180], central_longitude=0,
                cmap = 'viridis', vrange = [None, None],levels=None, fsize = [12,6],
                drawLand=False, drawGrid=False, meridians=None, parallels=None,  
                patch=None, pcol='k', palpha=0.5, plw=3,clevels=None, 
                ccol = 'k', clw=2, clfmt='%1.3f', clf=12, 
                drawBath=True, blevels = [1000.0], bcol = 'k', balpha=1.0,
                cb_orientation = 'vertical', cb_ticks = None, cb_label='', title=''):

    '''
    Generates simple cartopy map using SouthPolarStereo cartopy projection.
    
    Parameters
    ----------
    ds : xarray.Dataset
        Input data for map
    var : str
        Variable name
    lat : int
        Latitude of the northern edge of the map
    cmap : str
        Colormap
    vrange : list
        Minimum and maximum value of colorscale

    Returns
    -------
    fig: Figure instance
    '''

    fig = plt.figure(figsize=fsize)
    ax = plt.axes(projection=cartopy.crs.PlateCarree(central_longitude=central_longitude),facecolor='grey')
    ax.set_extent([lon_range[0], lon_range[1], lat_range[1], lat_range[0]], cartopy.crs.PlateCarree())

    p =  ds[var].where(ds.wet==1).plot(ax=ax, x='x', y='y', vmin=vrange[0], vmax=vrange[1], cmap=cmap,levels=levels,
                                       transform=cartopy.crs.PlateCarree(), add_colorbar=False)

    if drawBath:
        ds.deptho.plot.contour(ax=ax, x='x', y='y', levels=blevels, linewidths=1, colors=bcol, alpha=balpha,
                               transform=cartopy.crs.PlateCarree(), add_labels=False)

    if varc:
        cs = ds[varc].where(ds.wet==1).plot.contour(ax=ax, x='x', y='y', levels=clevels,
                                                    linewidths=clw, colors=ccol,
                                                    transform=cartopy.crs.PlateCarree())
        ax.clabel(cs, cs.levels, inline=True, fmt=clfmt, fontsize=clf)

    if drawLand:
        ax.add_feature(cartopy.feature.LAND, color='lightgrey')
        ax.coastlines(resolution='50m', linewidth=1.0, color='grey')

    if drawGrid:
        gl = ax.gridlines(cartopy.crs.PlateCarree(), draw_labels=False, linewidth=1.0,
                          linestyle=':', color='k', alpha=0.5)

        if meridians:
            gl.xlocator = mticker.FixedLocator(meridians)

        if parallels:
            gl.ylocator = mticker.FixedLocator(parallels)

    if patch:
        for region in patch.keys():
            gdf_patch = get_gdf_patch(patch[region])
            ax.add_geometries([gdf_patch['geometry'][0]], cartopy.crs.PlateCarree(),
                              facecolor='none', edgecolor=pcol, linewidth=plw, alpha=palpha,
                              linestyle='-', zorder=3)

    ax.set_title(title, fontsize=14, fontweight='bold')

    # Add separate colorbar
    cb = plt.colorbar(p, ticks=cb_ticks, shrink=0.8, pad=0.05, orientation=cb_orientation)
    cb.set_label(cb_label, fontsize=13)
    cb.ax.tick_params(labelsize=12)

    return fig, ax




def cartopy_map_so(ds, var, varc = None, lat = -50, cmap = 'viridis', vrange = [None, None], levels=None, fsize = [10,8],
                   drawLand=False, drawGrid=False, meridians=None, parallels=None,  patch=None, pcol='k', palpha=0.5, plw=3,
                   clevels=None, ccol = 'k', clw=2, clfmt='%1.3f',clf=12, drawBath=True, blevels = [1000.0], bcol = 'k', balpha=1.0,
                   cb_orientation = 'vertical', cb_ticks = None, cb_label='', title=''):

    '''
    Generates simple cartopy map using SouthPolarStereo cartopy projection.
    
    Parameters
    ----------
    ds : xarray.Dataset
        Input data for map
    var : str
        Variable name
    lat : int
        Latitude of the northern edge of the map
    cmap : str
        Colormap
    vrange : list
        Minimum and maximum value of colorscale

    Returns
    -------
    fig: Figure instance
    '''

    circle = get_so_map_boundary()
    fig = plt.figure(figsize=fsize)
    ax = plt.axes(projection=cartopy.crs.SouthPolarStereo(central_longitude=0.),facecolor='grey')
    ax.set_boundary(circle, transform=ax.transAxes)
    ax.set_extent([-300, 60, lat, -90], cartopy.crs.PlateCarree())
    
    p =  ds[var].where(ds.wet==1).plot(ax=ax, x='x', y='y', vmin=vrange[0], vmax=vrange[1], cmap=cmap, levels=levels,
                                       transform=cartopy.crs.PlateCarree(), add_colorbar=False)
    
    if drawBath:
        ds.deptho.plot.contour(ax=ax, x='x', y='y', levels=blevels, linewidths=1, colors=bcol, alpha=balpha, 
                               transform=cartopy.crs.PlateCarree(), add_labels=False)
    
    if varc:
        cs = ds[varc].where(ds.wet==1).plot.contour(ax=ax, x='x', y='y', levels=clevels,
                                                    linewidths=clw, colors=ccol,
                                                    transform=cartopy.crs.PlateCarree())
        ax.clabel(cs, cs.levels, inline=True, fmt=clfmt, fontsize=clf)

    if drawLand:
        ax.add_feature(cartopy.feature.LAND, color='lightgrey')
        ax.coastlines(resolution='50m', linewidth=1.0, color='grey')
        
    if drawGrid:
        gl = ax.gridlines(cartopy.crs.PlateCarree(), draw_labels=False, linewidth=1.0, 
                          linestyle=':', color='k', alpha=0.5)
        
        if meridians:
            gl.xlocator = mticker.FixedLocator(meridians)
            
        if parallels:
            gl.ylocator = mticker.FixedLocator(parallels)

    if patch:
        for region in patch.keys():
            gdf_patch = get_gdf_patch(patch[region])
            ax.add_geometries([gdf_patch['geometry'][0]], cartopy.crs.PlateCarree(), 
                              facecolor='none', edgecolor=pcol, linewidth=plw, alpha=palpha,
                              linestyle='-', zorder=3)
            
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # Add separate colorbar
    cb = plt.colorbar(p, ticks=cb_ticks, shrink=0.8, pad=0.05, orientation=cb_orientation)
    cb.set_label(cb_label, fontsize=13)
    cb.ax.tick_params(labelsize=12)

    return fig, ax


def cartopy_map_so_2by1(datal, datar, titles, cmap = 'RdBu_r', lat = -50, vrangel = [None, None], vranger = [None, None]):

    '''
    Generates a 2-by-1 figure using SouthPolarStereo cartopy projection.
    
    Parameters
    ----------
    datal : list [xarray.DataArray, xarray.Dataset]
        Input data for left map
    datar : list [xarray.DataArray, xarray.Dataset]
        Input data for right map
    titles : list [str, str]
        Title for each map
    cb_label : str
        Label for colorbar
        
    Returns
    -------
    fig: Figure instance
    '''

    circle = get_so_map_boundary()

    fig = plt.figure(figsize=(15,6))
    fig.subplots_adjust(left=0.1, right=0.88, bottom=0.1, top=0.9, hspace=0.1, wspace=0.02)

    ax1 = fig.add_subplot(1, 2, 1, projection=cartopy.crs.SouthPolarStereo(central_longitude=0.))
    ax1.set_boundary(circle, transform=ax1.transAxes)
    ax1.set_extent([-300, 60, lat, -90], cartopy.crs.PlateCarree())
    ax1.add_feature(cartopy.feature.LAND, color='lightgrey')
    ax1.coastlines(resolution='50m', linewidth=1.0, color='grey')
    datal[0].where(datal[1].wet==1).plot(ax=ax1, x='x', y='y', vmin=vrangel[0], vmax=vrangel[1], cmap=cmap, 
                                         transform=cartopy.crs.PlateCarree())
    datal[1].deptho.plot.contour(ax=ax1, x='x', y='y', levels=[1000.0], linewidths=1, colors='k', 
                                 add_labels=False, transform=cartopy.crs.PlateCarree())
    ax1.set_title(titles[0], fontsize=14, fontweight='bold')

    ax2 = fig.add_subplot(1, 2, 2, projection=cartopy.crs.SouthPolarStereo(central_longitude=0.))
    ax2.set_boundary(circle, transform=ax2.transAxes)
    ax2.set_extent([-300, 60, lat, -90], cartopy.crs.PlateCarree())
    ax2.add_feature(cartopy.feature.LAND, color='lightgrey')
    ax2.coastlines(resolution='50m', linewidth=1.0, color='grey')
    p = datar[0].where(datar[1].wet==1).plot(ax=ax2, x='x', y='y', vmin=vranger[0], vmax=vranger[1], cmap=cmap,
                                             transform=cartopy.crs.PlateCarree())
    datar[1].deptho.plot.contour(ax=ax2, x='x', y='y', levels=[1000.0], linewidths=1, colors='k',
                                 add_labels=False, transform=cartopy.crs.PlateCarree())
    ax2.set_title(titles[1], fontsize=14, fontweight='bold')
    return fig


def map_transformation_rate(F, ds, l, lval, month=None, draw_Mean=True, 
                            lat_range=[-90, 90], lon_range=[-180, 180], central_longitude=0,
                            cmap = 'RdBu_r', vrange = [None, None], fsize = [8,6],
                            drawLand=False, drawGrid=False, meridians=None, parallels=None, 
                            ccolmean = 'k', ccolmonth = 'k', clw_mean=2, clw_month=1, clf_mean=12, clf_month=10,
                            patch=None, pcol='k', palpha=0.5,plw=3,
                            cb_orientation='vertical', cb_ticks=None, cb_label='', title=''):
                               
    '''
    Transformation map with lambda contours using PlateCarree cartopy projection.
    
    Parameters
    ----------
    F : xarray.DataArray
        Input data for trandformation map
    ds: xarray.Dataset
        Input data for map
    l : xarray.DataArray
        Tracer (lambda) field
    lval : float
        Lambda value for which contours are plotted
    month : list (optional)
        Month(s) for which contours are plotted
    draw_Mean : bool (optional)
        Option to draw the annual mean contour (default is True)
    lat_range : int (optional)
        Latitude bounds of the map (default is -90 to 90)
    lon_range : int (optional)
        Longitude bounds of the map (default is -180 to 180)
    cmap : str (optional)
        Colormap (default is 'RdBu_r')
    vrange : list (optional)
        Minimum and maximum value of colorscale
    cb_label : str (optional)
        Label for colorbar
    title : str (optional)
        Title for the plot
    
    Returns
    -------
    fig: Figure instance
    '''
    
    fig = plt.figure(figsize=fsize)
    ax = plt.axes(projection=cartopy.crs.PlateCarree(central_longitude=central_longitude),facecolor='grey')
    ax.set_extent([lon_range[0], lon_range[1], lat_range[1], lat_range[0]], cartopy.crs.PlateCarree())
    
    p = F.where(ds.wet==1).plot(ax=ax, x='x', y='y', vmin=vrange[0], vmax=vrange[1], cmap=cmap, 
                                transform=cartopy.crs.PlateCarree(), add_colorbar=False)
    if draw_Mean:
        cs = l.mean('time').where(ds.wet==1).plot.contour(ax=ax, x='x', y='y', levels=np.linspace(lval,lval,1),
                                                          linewidths=clw_mean, colors=ccolmean, 
                                                          transform=cartopy.crs.PlateCarree())
        ax.clabel(cs, cs.levels, inline=True, fmt={lval: '%.2f (Mean)'%lval}, fontsize=clf_mean)
    
    if month:
        for m in month:
            lmon = l[l.time.dt.month==m].mean('time')
            cs_mon = lmon.plot.contour(ax=ax, x='x', y='y', levels=np.linspace(lval,lval,1),
                                       linewidths=clw_month, colors=ccolmonth, transform=cartopy.crs.PlateCarree())
            ax.clabel(cs_mon, cs_mon.levels, inline=True, 
                      fmt={lval: '%.2f ('%lval+calendar.month_abbr[m]+')'}, fontsize=clf_month)

    
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    if drawLand:
        ax.add_feature(cartopy.feature.LAND, color='lightgrey')
        ax.coastlines(resolution='50m', linewidth=1.0, color='grey')
    
    if drawGrid:
        gl = ax.gridlines(cartopy.crs.PlateCarree(), draw_labels=False, linewidth=1.0, 
                          linestyle=':', color='k', alpha=0.5)
        
        if meridians:
            gl.xlocator = mticker.FixedLocator(meridians)
            
        if parallels:
            gl.ylocator = mticker.FixedLocator(parallels)
            
    if patch:
        for region in patch.keys():
            gdf_patch = get_gdf_patch(patch[region])
            ax.add_geometries([gdf_patch['geometry'][0]], cartopy.crs.PlateCarree(), 
                              facecolor='none', edgecolor=(0.5,0.5,0.5,1), linewidth=2, 
                              linestyle='-', zorder=3, alpha=0.5)

    
    cax = fig.add_axes([0.9, 0.2 , 0.02, 0.6])
    cb = fig.colorbar(p, cax=cax, ticks=cb_ticks, orientation=cb_orientation, shrink=0.5, pad=0.2)
    cb.set_label(cb_label, fontsize=13)
    cb.ax.tick_params(labelsize=12)
    
    return fig, ax



def map_transformation_rate_so(F, ds, l, lval, month=None, draw_Mean=True, lat = -50, title='', cb_label='',
                               cmap = 'RdBu_r', vrange = [None, None], fsize = [8,6],
                               drawGrid=False, meridians=None, parallels=None, patch=None):
    '''
    Transformation map with lambda contours using SouthPolarStereo cartopy projection.
    
    Parameters
    ----------
    F : xarray.DataArray
        Input data for trandformation map
    ds: xarray.Dataset
        Input data for map
    l : xarray.DataArray
        Tracer (lambda) field
    lval : float
        Lambda value for which contours are plotted
    month : list (optional)
        Month(s) for which contours are plotted
    draw_Mean : bool (optional)
        Option to draw the annual mean contour (default is True)
    lat : int (optional)
        Latitude of the northern edge of the map (default is -50)
    title : str (optional)
        Title for the plot
    cb_label : str (optional)
        Label for colorbar
    cmap : str (optional)
        Colormap (default is 'RdBu_r')
    vrange : list (optional)
        Minimum and maximum value of colorscale

    Returns
    -------
    fig: Figure instance
    '''
    
    circle = get_so_map_boundary()
    
    fig = plt.figure(figsize=fsize)
    ax = plt.axes(projection=cartopy.crs.SouthPolarStereo(central_longitude=0.),facecolor='grey')
    ax.set_boundary(circle, transform=ax.transAxes)
    ax.set_extent([-300, 60, lat, -90], cartopy.crs.PlateCarree())
    
    p = F.where(ds.wet==1).plot(ax=ax, x='x', y='y', vmin=vrange[0], vmax=vrange[1], cmap='RdBu_r', 
                                transform=cartopy.crs.PlateCarree(), add_colorbar=False)
    if draw_Mean:
        cs = l.mean('time').where(ds.wet==1).plot.contour(ax=ax, x='x', y='y', levels=np.linspace(lval,lval,1),
                                                          linewidths=2, colors='k', 
                                                          transform=cartopy.crs.PlateCarree())
        ax.clabel(cs, cs.levels, inline=True, fmt={lval: '%.2f (Mean)'%lval}, fontsize=12)
    
    if month:
        for m in month:
            lmon = l[l.time.dt.month==m].mean('time')
            cs_mon = lmon.plot.contour(ax=ax, x='x', y='y', levels=np.linspace(lval,lval,1),
                                       linewidths=1, colors='k', transform=cartopy.crs.PlateCarree())
            ax.clabel(cs_mon, cs_mon.levels, inline=True, 
                      fmt={lval: '%.2f ('%lval+calendar.month_abbr[m]+')'}, fontsize=10)

    
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    if drawGrid:
        gl = ax.gridlines(cartopy.crs.PlateCarree(), draw_labels=False, linewidth=1.0, 
                          linestyle=':', color='k', alpha=0.5)
        
        if meridians:
            gl.xlocator = mticker.FixedLocator(meridians)
            
        if parallels:
            gl.ylocator = mticker.FixedLocator(parallels)
            
    if patch:
        for region in patch.keys():
            gdf_patch = get_gdf_patch(patch[region])
            ax.add_geometries([gdf_patch['geometry'][0]], cartopy.crs.PlateCarree(), 
                              facecolor='none', edgecolor=(0.5,0.5,0.5,1), linewidth=2, 
                              linestyle='-', zorder=3, alpha=0.5)

    
    cax = fig.add_axes([0.9, 0.2 , 0.02, 0.6])
    cb = fig.colorbar(p, cax=cax, orientation='vertical', shrink=0.5, pad=0.2)
    cb.set_label(cb_label, fontsize=13)
    cb.ax.tick_params(labelsize=12)
    
    return fig, ax

def map_transformation_rate_3by1(F, ds, l, lval, month=None, draw_Mean=True, 
                                 cb_label='',sub_titles=[''], sup_title='',extent = [-180, 180, -90, 90],
                                 cmap = 'RdBu_r', vrange = [None, None], fsize = [14,14]):
    '''
    Generates a 3-by-1 figure with maps using cartopy.
    
    Parameters
    ----------

    F : list [xarray.DataArrays]
        Input data for trandformation map
    ds: xarray.Dataset
        Input data for map
    l : xarray.DataArray
        Tracer (lambda) field
    lval : list float
        Lambda values for which contours are plotted
    month : list (optional)
        Month(s) for which contours are plotted
    draw_Mean : bool (optional)
        Option to draw the annual mean contour (default is True)
    cb_label : str (optional)
        Label for colorbar
    sup_title : str (optional)
        Title for the entire plot
    sup_title : list [str] (optional)
        Titles for each of the three subplots
    cmap : str (optional)
        Colormap (default is 'RdBu_r')
    vrange : list (optional)
        Minimum and maximum value of colorscale
    fsize : list [width,height] (optional)
        Figure size
        
    Returns
    -------
    fig: Figure instance
    '''

    fig = plt.figure(figsize=fsize)
    fig.suptitle(sup_title, fontsize=14, fontweight='bold', y=0.95)
    
    ax1 = fig.add_subplot(3, 1, 1, projection=cartopy.crs.PlateCarree(),facecolor='grey')
    ax1.set_extent(extent, cartopy.crs.PlateCarree())
    p = F[0].where(ds.wet==1).plot(ax=ax1, x='x', y='y', vmin=vrange[0], vmax=vrange[1], cmap='RdBu_r', 
                                   transform=cartopy.crs.PlateCarree(), add_colorbar=False)
    
    if draw_Mean:
        cs = l.mean('time').where(ds.wet==1).plot.contour(ax=ax1, x='x', y='y', 
                                                          levels=np.linspace(lval[0],lval[0],1),
                                                          linewidths=2, colors='k',
                                                          transform=cartopy.crs.PlateCarree())
        ax1.clabel(cs, cs.levels, inline=True, fmt={lval[0]: '%.2f (Mean)'%lval[0]}, fontsize=12)
        
    if month:
        for m in month:
            lmon = l[l.time.dt.month==m].mean('time')
            cs_mon = lmon.plot.contour(ax=ax1, x='x', y='y', levels=np.linspace(lval[0],lval[0],1),
                                       linewidths=1, colors='k', transform=cartopy.crs.PlateCarree())
            ax1.clabel(cs_mon, cs_mon.levels, inline=True,
                       fmt={lval[0]: '%.2f ('%lval[0]+calendar.month_abbr[m]+')'}, fontsize=10)
            
    ax1.set_title(sub_titles[0], fontsize=16)
    
    ax2 = fig.add_subplot(3, 1, 2, projection=cartopy.crs.PlateCarree(),facecolor='grey')
    ax2.set_extent(extent, cartopy.crs.PlateCarree())
    p = F[1].where(ds.wet==1).plot(ax=ax2, x='x', y='y', vmin=vrange[0], vmax=vrange[1], cmap='RdBu_r', 
                                   transform=cartopy.crs.PlateCarree(), add_colorbar=False)
    
    if draw_Mean:
        cs = l.mean('time').where(ds.wet==1).plot.contour(ax=ax2, x='x', y='y', 
                                                          levels=np.linspace(lval[1],lval[1],1),
                                                          linewidths=2, colors='k',
                                                          transform=cartopy.crs.PlateCarree())
        ax2.clabel(cs, cs.levels, inline=True, fmt={lval[1]: '%.2f (Mean)'%lval[1]}, fontsize=12)
        
    if month:
        for m in month:
            lmon = l[l.time.dt.month==m].mean('time')
            cs_mon = lmon.plot.contour(ax=ax2, x='x', y='y', levels=np.linspace(lval[1],lval[1],1),
                                       linewidths=1, colors='k', transform=cartopy.crs.PlateCarree())
            ax2.clabel(cs_mon, cs_mon.levels, inline=True,
                       fmt={lval[1]: '%.2f ('%lval[1]+calendar.month_abbr[m]+')'}, fontsize=10)
            
    ax2.set_title(sub_titles[1], fontsize=16)
    
    ax3 = fig.add_subplot(3, 1, 3, projection=cartopy.crs.PlateCarree(),facecolor='grey')
    ax3.set_extent(extent, cartopy.crs.PlateCarree())
    p = F[2].where(ds.wet==1).plot(ax=ax3, x='x', y='y', vmin=vrange[0], vmax=vrange[1], cmap='RdBu_r', 
                                   transform=cartopy.crs.PlateCarree(), add_colorbar=False)
    
    if draw_Mean:
        cs = l.mean('time').where(ds.wet==1).plot.contour(ax=ax3, x='x', y='y', 
                                                          levels=np.linspace(lval[2],lval[2],1),
                                                          linewidths=2, colors='k',
                                                          transform=cartopy.crs.PlateCarree())
        ax2.clabel(cs, cs.levels, inline=True, fmt={lval[2]: '%.2f (Mean)'%lval[2]}, fontsize=12)
        
    if month:
        for m in month:
            lmon = l[l.time.dt.month==m].mean('time')
            cs_mon = lmon.plot.contour(ax=ax3, x='x', y='y', levels=np.linspace(lval[2],lval[2],1),
                                       linewidths=1, colors='k', transform=cartopy.crs.PlateCarree())
            ax3.clabel(cs_mon, cs_mon.levels, inline=True,
                       fmt={lval[2]: '%.2f ('%lval[2]+calendar.month_abbr[m]+')'}, fontsize=10)
    
    ax3.set_title(sub_titles[2], fontsize=16)
    
    fig.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9, hspace=0.1)
    cax = fig.add_axes([0.80, 0.25, 0.017, 0.5])
    cb = fig.colorbar(p, cax=cax, orientation='vertical', shrink=0.9, pad=0.02)
    cb.set_label(cb_label, fontsize=14)
    cb.ax.tick_params(labelsize=12)
    
    return fig


def map_transformation_rate_so_1by3(F, ds, l, lval, month=None, draw_Mean=True, 
                                    cb_label='',sub_titles=['','',''], sup_title='',lat = -40,
                                    cmap = 'RdBu_r', vrange = [None, None], fsize = [15,6],
                                    drawGrid=False, meridians=None, parallels=None, add_integral=True,
                                    add_wmt_plot=False, G={'mean':[],'std':[]},wmt_label=''):
    '''
    Generates a 1-by-3 figure with lambda contours using SouthPolarStereo cartopy projection.
    
    Parameters
    ----------

    F : list [xarray.DataArrays]
        Input data for trandformation map
    ds: xarray.Dataset
        Input data for map
    l : xarray.DataArray
        Tracer (lambda) field
    lval : list float
        Lambda values for which contours are plotted
    month : list (optional)
        Month(s) for which contours are plotted
    draw_Mean : bool (optional)
        Option to draw the annual mean contour (default is True)
    cb_label : str (optional)
        Label for colorbar
    sup_title : str (optional)
        Title for the entire plot
    sup_title : list [str] (optional)
        Titles for each of the three subplots
    lat : int
        Latitude of the northern edge of the map
    cmap : str (optional)
        Colormap (default is 'RdBu_r')
    vrange : list (optional)
        Minimum and maximum value of colorscale
    fsize : list [width,height] (optional)
        Figure size
    drawGrid : bool
        Option to draw gridlines
    meridians : list (optional)
        Specify longitude values for meridians
    parallels : list (optional)
        Specify latitude values for parallels
    add_integral : bool
        Option to add area-integrated values
    add_wmt_plot : bool
        Option to add a figure that shows the WMT line plot
    G : dictionary {'mean':[xarray.DataArray],'std':[xarray.DataArray]}
        Includes the WMT curve with mean and standard deviation
        
    Returns
    -------
    fig: Figure instance
    '''

    # Title string for central box
    box_titel_str = 'net = %.1f Sv\npos. = %.1f Sv\nneg. = %.1f Sv'
    
    circle = get_so_map_boundary()

    fig = plt.figure(figsize=fsize)
    fig.suptitle(sup_title, fontsize=14, fontweight='bold', y=0.95)
    gs = gridspec.GridSpec(2, 3, width_ratios =[1,1,1])

    if add_wmt_plot:
        ax1 = plt.subplot(gs[0,0], projection=cartopy.crs.SouthPolarStereo(central_longitude=0.),facecolor='gray')
    else:
        ax1 = fig.add_subplot(1, 3, 1, projection=cartopy.crs.SouthPolarStereo(central_longitude=0.),facecolor='grey')
        
    ax1.set_boundary(circle, transform=ax1.transAxes)
    ax1.set_extent([-300, 60, lat, -90], cartopy.crs.PlateCarree())
    p = F[0].where(ds.wet==1).plot(ax=ax1, x='x', y='y', vmin=vrange[0], vmax=vrange[1], cmap='RdBu_r', 
                                   transform=cartopy.crs.PlateCarree(), add_colorbar=False)
    
    if draw_Mean:
        cs = l.mean('time').where(ds.wet==1).plot.contour(ax=ax1, x='x', y='y', 
                                                          levels=np.linspace(lval[0],lval[0],1),
                                                          linewidths=2, colors='k',
                                                          transform=cartopy.crs.PlateCarree())
        ax1.clabel(cs, cs.levels, inline=True, fmt={lval[0]: '%.2f (Mean)'%lval[0]}, fontsize=12)
        
    if month:
        for m in month:
            lmon = l[l.time.dt.month==m].mean('time')
            cs_mon = lmon.plot.contour(ax=ax1, x='x', y='y', levels=np.linspace(lval[0],lval[0],1),
                                       linewidths=1, colors='k', transform=cartopy.crs.PlateCarree())
            ax1.clabel(cs_mon, cs_mon.levels, inline=True,
                       fmt={lval[0]: '%.2f ('%lval[0]+calendar.month_abbr[m]+')'}, fontsize=10)
            
    ax1.set_title(sub_titles[0], fontsize=16)
    
    if add_wmt_plot:
        ax2 = plt.subplot(gs[0,1], projection=cartopy.crs.SouthPolarStereo(central_longitude=0.),facecolor='gray')
    else:
        ax2 = fig.add_subplot(1, 3, 2, projection=cartopy.crs.SouthPolarStereo(central_longitude=0.),facecolor='grey')
        
        
    
    ax2.set_boundary(circle, transform=ax2.transAxes)
    ax2.set_extent([-300, 60, lat, -90], cartopy.crs.PlateCarree())
    p = F[1].where(ds.wet==1).plot(ax=ax2, x='x', y='y', vmin=vrange[0], vmax=vrange[1], cmap='RdBu_r', 
                                   transform=cartopy.crs.PlateCarree(), add_colorbar=False)
    
    if draw_Mean:
        cs = l.mean('time').where(ds.wet==1).plot.contour(ax=ax2, x='x', y='y', 
                                                          levels=np.linspace(lval[1],lval[1],1),
                                                          linewidths=2, colors='k',
                                                          transform=cartopy.crs.PlateCarree())
        ax2.clabel(cs, cs.levels, inline=True, fmt={lval[1]: '%.2f (Mean)'%lval[1]}, fontsize=12)
        
    if month:
        for m in month:
            lmon = l[l.time.dt.month==m].mean('time')
            cs_mon = lmon.plot.contour(ax=ax2, x='x', y='y', levels=np.linspace(lval[1],lval[1],1),
                                       linewidths=1, colors='k', transform=cartopy.crs.PlateCarree())
            ax2.clabel(cs_mon, cs_mon.levels, inline=True,
                       fmt={lval[1]: '%.2f ('%lval[1]+calendar.month_abbr[m]+')'}, fontsize=10)
            
    ax2.set_title(sub_titles[1], fontsize=16)
    
    if add_wmt_plot:
        ax3 = plt.subplot(gs[0,2], projection=cartopy.crs.SouthPolarStereo(central_longitude=0.),facecolor='gray')
    else:
        ax3 = fig.add_subplot(1, 3, 3, projection=cartopy.crs.SouthPolarStereo(central_longitude=0.),facecolor='grey')
    
    ax3.set_boundary(circle, transform=ax3.transAxes)
    ax3.set_extent([-300, 60, lat, -90], cartopy.crs.PlateCarree())
    p = F[2].where(ds.wet==1).plot(ax=ax3, x='x', y='y', vmin=vrange[0], vmax=vrange[1], cmap='RdBu_r', 
                                   transform=cartopy.crs.PlateCarree(), add_colorbar=False)
    
    if draw_Mean:
        cs = l.mean('time').where(ds.wet==1).plot.contour(ax=ax3, x='x', y='y', 
                                                          levels=np.linspace(lval[2],lval[2],1),
                                                          linewidths=2, colors='k',
                                                          transform=cartopy.crs.PlateCarree())
        ax2.clabel(cs, cs.levels, inline=True, fmt={lval[2]: '%.2f (Mean)'%lval[2]}, fontsize=12)
        
    if month:
        for m in month:
            lmon = l[l.time.dt.month==m].mean('time')
            cs_mon = lmon.plot.contour(ax=ax3, x='x', y='y', levels=np.linspace(lval[2],lval[2],1),
                                       linewidths=1, colors='k', transform=cartopy.crs.PlateCarree())
            ax3.clabel(cs_mon, cs_mon.levels, inline=True,
                       fmt={lval[2]: '%.2f ('%lval[2]+calendar.month_abbr[m]+')'}, fontsize=10)
    
    ax3.set_title(sub_titles[2], fontsize=16)


    if drawGrid:
        
        for ax in [ax1,ax2,ax3]:
            gl = ax.gridlines(cartopy.crs.PlateCarree(), draw_labels=False, linewidth=1.0,
                              linestyle=':', color='k', alpha=0.5)
            if meridians:
                gl.xlocator = mticker.FixedLocator(meridians)
            if parallels:
                gl.ylocator = mticker.FixedLocator(parallels)


    if add_integral:
        for idx, ax in enumerate([ax1,ax2,ax3]):
            ax.add_artist(AnchoredText(box_titel_str% \
                                        (np.round((F[idx]*1e-6*ds['areacello']).sum(['x','y']).values,1),
                                         np.round((F[idx].where(F[idx]>0)*1e-6*ds['areacello'])\
                                                  .sum(['x','y']).values,1),
                                         np.round((F[idx].where(F[idx]<0)*1e-6*ds['areacello'])\
                                                  .sum(['x','y']).values,1)),
                                       loc='center', prop={'size': 10, 'weight': 'bold'}, frameon=False,
                                       bbox_to_anchor=(0.54, 0.51), bbox_transform=ax.transAxes))

   
    fig.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9, wspace=0.02)
    if add_wmt_plot:
        cax = fig.add_axes([0.91, 0.565, 0.01, 0.3])
    else:
        cax = fig.add_axes([0.91, 0.2, 0.017, 0.6])
    cb = fig.colorbar(p, cax=cax, orientation='vertical', shrink=0.9, pad=0.05)
    cb.set_label(cb_label, fontsize=14)
    cb.ax.tick_params(labelsize=12)
    
    if add_wmt_plot:
        ax4 = plt.subplot(gs[1,:])
        ax4.axhline(y=0, xmin=0, xmax=1, linewidth=1.0, color = 'k')
        ax4.fill_between(G['mean'].sigma1, (G['mean']-G['std'])*1e-6, (G['mean']+G['std'])*1e-6, alpha=0.1,
                         edgecolor='k', facecolor='k')
        ax4.plot(G['mean'].sigma1, G['mean']*1e-6, ls='-', lw=2, c='k',marker='o', ms=3, label=wmt_label)
        ax4.plot([lval[0],lval[1]],
                 [(F[0]*ds['areacello']).sum(['x','y'])*1e-6,(F[1]*ds['areacello']).sum(['x','y'])*1e-6],
                 ls='-', lw=4, marker='o', ms=6, c='r')
        ax4.tick_params(axis='both', which='major', labelsize=12)
        ax4.set_ylabel('Mean transformation [Sv]',fontsize=16)

        pos = ax4.get_position()
        ax4.set_position([pos.x0 + 0.1, pos.y0+0.05, (pos.x1 - pos.x0)*0.8, (pos.y1 - pos.y0)*0.98])
    
    return fig, ax1, ax2, ax3



def map_comparison_so(datal, datar, titles, cb_label, cmap = 'RdBu_r', lat = -50, vrange = [None, None]):
    
    '''
    Generates a 2-by-1 figure using SouthPolarStereo cartopy projection.
    
    Parameters
    ----------
    datal : list [xarray.DataArray, xarray.Dataset]
        Input data for left map
    datar : list [xarray.DataArray, xarray.Dataset]
        Input data for right map
    titles : list [str, str]
        Title for each map
    cb_label : str
        Label for colorbar
        
    Returns
    -------
    fig: Figure instance
    '''

    circle = get_so_map_boundary()

    
    fig = plt.figure(figsize=(12,6))
    fig.subplots_adjust(left=0.1, right=0.88, bottom=0.1, top=0.9, hspace=0.1, wspace=0.02)

    ax1 = fig.add_subplot(1, 2, 1, projection=cartopy.crs.SouthPolarStereo(central_longitude=0.))
    ax1.set_boundary(circle, transform=ax1.transAxes)
    ax1.set_extent([-300, 60, lat, -90], cartopy.crs.PlateCarree())
    ax1.add_feature(cartopy.feature.LAND, color='lightgrey')
    ax1.coastlines(resolution='50m', linewidth=1.0, color='grey')
    datal[0].where(datal[1].wet==1).plot(ax=ax1, x='x', y='y', vmin=vrange[0], vmax=vrange[1], cmap=cmap, 
                                         add_labels=False, add_colorbar=False, transform=cartopy.crs.PlateCarree())
    datal[1].deptho.plot.contour(ax=ax1, x='x', y='y', levels=[1000.0], linewidths=1, colors='k', 
                                 add_labels=False, transform=cartopy.crs.PlateCarree())
    ax1.set_title(titles[0], fontsize=14, fontweight='bold')
    
    ax2 = fig.add_subplot(1, 2, 2, projection=cartopy.crs.SouthPolarStereo(central_longitude=0.))
    ax2.set_boundary(circle, transform=ax2.transAxes)
    ax2.set_extent([-300, 60, lat, -90], cartopy.crs.PlateCarree())
    ax2.add_feature(cartopy.feature.LAND, color='lightgrey')
    ax2.coastlines(resolution='50m', linewidth=1.0, color='grey')
    p = datar[0].where(datar[1].wet==1).plot(ax=ax2, x='x', y='y', vmin=vrange[0], vmax=vrange[1], cmap=cmap, 
                                             add_labels=False, add_colorbar=False, 
                                             transform=cartopy.crs.PlateCarree())
    datar[1].deptho.plot.contour(ax=ax2, x='x', y='y', levels=[1000.0], linewidths=1, colors='k', 
                                 add_labels=False, transform=cartopy.crs.PlateCarree())
    ax2.set_title(titles[1], fontsize=14, fontweight='bold')
    
    cax = fig.add_axes([0.9, 0.2 , 0.02, 0.6])
    cb = fig.colorbar(p, cax=cax, orientation='vertical', shrink=0.5, pad=0.2)
    cb.set_label(cb_label, fontsize=13)
    cb.ax.tick_params(labelsize=12)
    
    return fig


def map_comparison_so_wmt(datal, datar, titles, cb_label, lat = -50, vrange = [-1e-5, 1e-5]):
    
    '''
    Generates a 2-by-1 figure using SouthPolarStereo cartopy projection.
    
    Parameters
    ----------
    datal : list [xarray.DataArray, xarray.Dataset]
        Input data for left map
    datar : list [xarray.DataArray, xarray.Dataset]
        Input data for right map
    titles : list [str, str]
        Title for each map
    cb_label : str
        Label for colorbar
        
    Returns
    -------
    fig: Figure instance
    '''

    # Parameters for cartopy map
    theta = np.linspace(0, 2*np.pi, 100)
    center, radius = [0.5, 0.5], 0.5
    verts = np.vstack([np.sin(theta), np.cos(theta)]).T
    circle = mpath.Path(verts * radius + center)

    # Title string for central box
    box_titel_str = 'total = %.1f Sv\npos. = %.1f Sv\nneg. = %.1f Sv\non shlf = %.1f Sv\noff shlf = %.1f Sv'
    
    fig = plt.figure(figsize=(12,6))
    fig.subplots_adjust(left=0.1, right=0.88, bottom=0.1, top=0.9, hspace=0.1, wspace=0.02)

    ax1 = fig.add_subplot(1, 2, 1, projection=cartopy.crs.SouthPolarStereo(central_longitude=0.))
    ax1.set_boundary(circle, transform=ax1.transAxes)
    ax1.set_extent([-300, 60, lat, -90], cartopy.crs.PlateCarree())
    ax1.add_feature(cartopy.feature.LAND, color='lightgrey')
    ax1.coastlines(resolution='50m', linewidth=1.0, color='grey')
    datal[0].where(datal[1].wet==1).plot(ax=ax1, x='x', y='y', vmin=vrange[0], vmax=vrange[1], cmap='RdBu_r', 
                                         add_labels=False, add_colorbar=False, transform=cartopy.crs.PlateCarree())
    datal[1].deptho.plot.contour(ax=ax1, x='x', y='y', levels=[1000.0], linewidths=1, colors='k', 
                                 add_labels=False, transform=cartopy.crs.PlateCarree())
    ax1.add_artist(AnchoredText(box_titel_str% \
                                (np.round((datal[0]*1e-6*datal[1]['areacello']).sum(['x','y']).values,1),
                                 np.round((datal[0].where(datal[0]>0)*1e-6*datal[1]['areacello'])\
                                          .sum(['x','y']).values,1),
                                 np.round((datal[0].where(datal[0]<0)*1e-6*datal[1]['areacello'])\
                                          .sum(['x','y']).values,1),
                                 np.round((datal[0].where(datal[1].deptho<1000)*1e-6*datal[1]['areacello'])\
                                          .sum(['x','y']).values,1),
                                 np.round((datal[0].where(datal[1].deptho>=1000)*1e-6*datal[1]['areacello'])\
                                          .sum(['x','y']).values,1)),
                                loc='center', prop={'size': 11, 'weight': 'bold'}, frameon=False,
                                bbox_to_anchor=(0.58, 0.52), bbox_transform=ax1.transAxes))
    ax1.set_title(titles[0], fontsize=14, fontweight='bold')
    
    ax2 = fig.add_subplot(1, 2, 2, projection=cartopy.crs.SouthPolarStereo(central_longitude=0.))
    ax2.set_boundary(circle, transform=ax2.transAxes)
    ax2.set_extent([-300, 60, lat, -90], cartopy.crs.PlateCarree())
    ax2.add_feature(cartopy.feature.LAND, color='lightgrey')
    ax2.coastlines(resolution='50m', linewidth=1.0, color='grey')
    p = datar[0].where(datar[1].wet==1).plot(ax=ax2, x='x', y='y', vmin=vrange[0], vmax=vrange[1], cmap='RdBu_r', 
                                             add_labels=False, add_colorbar=False, 
                                             transform=cartopy.crs.PlateCarree())
    datar[1].deptho.plot.contour(ax=ax2, x='x', y='y', levels=[1000.0], linewidths=1, colors='k', 
                                 add_labels=False, transform=cartopy.crs.PlateCarree())
    ax2.add_artist(AnchoredText(box_titel_str% \
                                (np.round((datar[0]*1e-6*datar[1]['areacello']).sum(['x','y']).values,1),
                                 np.round((datar[0].where(datar[0]>0)*1e-6*datar[1]['areacello'])\
                                          .sum(['x','y']).values,1),
                                 np.round((datar[0].where(datar[0]<0)*1e-6*datar[1]['areacello'])\
                                          .sum(['x','y']).values,1),
                                 np.round((datar[0].where(datar[1].deptho<1000)*1e-6*datar[1]['areacello'])\
                                          .sum(['x','y']).values,1),
                                 np.round((datar[0].where(datar[1].deptho>=1000)*1e-6*datar[1]['areacello'])\
                                          .sum(['x','y']).values,1)),
                                loc='center', prop={'size': 11, 'weight': 'bold'}, frameon=False,
                                bbox_to_anchor=(0.58, 0.52), bbox_transform=ax2.transAxes))
    ax2.set_title(titles[1], fontsize=14, fontweight='bold')
    
    cax = fig.add_axes([0.9, 0.2 , 0.02, 0.6])
    cb = fig.colorbar(p, cax=cax, orientation='vertical', shrink=0.5, pad=0.2)
    cb.set_label(cb_label, fontsize=13)
    cb.ax.tick_params(labelsize=12)
    
    return fig


def map_comparison_so_3by1(datal, datac, datar, titles, cb_label, lat = -50, vrange = [-1e-5, 1e-5],
                           add_integral=True):

    '''
    Generates a 3-by-1 figure using SouthPolarStereo cartopy projection.
    
    Parameters
    ----------
    datal : list [xarray.DataArray, xarray.Dataset]
        Input data for left map
    datac : list [xarray.DataArray, xarray.Dataset]
        Input data for center map
    datar : list [xarray.DataArray, xarray.Dataset]
        Input data for right map
    titles : list [str, str]
        Title for each map
    cb_label : str
        Label for colorbar
    lat : int (optional)
        Latitude of the northern edge of the map
    vrange : list (optional)
        Minimum and maximum value of colorscale
    add_integral: bool (optional)
        Option to display areal integral (net, positive, negative)

        
    Returns
    -------
    fig: Figure instance
    '''

    # Title string for central box
    box_titel_str = 'net = %.1f Sv\npos. = %.1f Sv\nneg. = %.1f Sv'

    # Parameters for cartopy map
    circle = get_so_map_boundary()

    fig = plt.figure(figsize=(18,6))
    fig.subplots_adjust(left=0.1, right=0.89, bottom=0.1, top=0.9, hspace=0.1, wspace=0.02)

    ax1 = fig.add_subplot(1, 3, 1, projection=cartopy.crs.SouthPolarStereo(central_longitude=0.),
                          facecolor='lightgrey')
    ax1.set_boundary(circle, transform=ax1.transAxes)
    ax1.set_extent([-300, 60, lat, -90], cartopy.crs.PlateCarree())
    p = datal[0].where(datal[1].wet==1).plot(ax=ax1, x='x', y='y',vmin=vrange[0], vmax=vrange[1],cmap='RdBu_r',
                                             transform=cartopy.crs.PlateCarree(),add_labels=False,add_colorbar=False)
    ax1.set_title(titles[0],fontsize=12)
    
    ax2 = fig.add_subplot(1, 3, 2, projection=cartopy.crs.SouthPolarStereo(central_longitude=0.),
                          facecolor='lightgrey')
    ax2.set_boundary(circle, transform=ax2.transAxes)
    ax2.set_extent([-300, 60, lat, -90], cartopy.crs.PlateCarree())
    datac[0].where(datac[1].wet==1).plot(ax=ax2, x='x', y='y', vmin=vrange[0], vmax=vrange[1], cmap='RdBu_r',
                                         transform=cartopy.crs.PlateCarree(),add_labels=False,add_colorbar=False)
    ax2.set_title(titles[1],fontsize=12)
    
    ax3 = fig.add_subplot(1, 3, 3, projection=cartopy.crs.SouthPolarStereo(central_longitude=0.),
                          facecolor='lightgrey')
    ax3.set_boundary(circle, transform=ax3.transAxes)
    ax3.set_extent([-300, 60, lat, -90], cartopy.crs.PlateCarree())
    datar[0].where(datar[1].wet==1).plot(ax=ax3, x='x', y='y', vmin=vrange[0], vmax=vrange[1], cmap='RdBu_r',
                                         transform=cartopy.crs.PlateCarree(),add_labels=False,add_colorbar=False)
    ax3.set_title(titles[2],fontsize=12)
    
    if add_integral:
        for (ax,data) in zip([ax1,ax2,ax3],[datal,datac,datar]):
            ax.add_artist(AnchoredText(box_titel_str% \
                                        (np.round((data[0]*1e-6*data[1]['areacello']).sum(['x','y']).values,1),
                                         np.round((data[0].where(data[0]>0)*1e-6*data[1]['areacello'])\
                                                  .sum(['x','y']).values,1),
                                         np.round((data[0].where(data[0]<0)*1e-6*data[1]['areacello'])\
                                                  .sum(['x','y']).values,1)),
                                        loc='center', prop={'size': 10, 'weight': 'bold'}, frameon=False,
                                        bbox_to_anchor=(0.54, 0.51), bbox_transform=ax.transAxes))
        
    cax = fig.add_axes([0.9, 0.1 , 0.02, 0.8])
    cb = fig.colorbar(p, cax=cax, orientation='vertical', shrink=0.5, pad=0.1)
    cb.set_label(cb_label, fontsize=14)
    cb.ax.tick_params(labelsize=12)

    return fig, ax1, ax2, ax3



def map_comparison_so_ecco(datal, datar, titles, cb_label, lat = -50, vrange = [-1e-5, 1e-5]):

    '''
    Generates a 2-by-1 figure using SouthPolarStereo cartopy projection.
    
    Parameters
    ----------
    datal : list [xarray.DataArray, xarray.Dataset]
        Input data for left map
    datar : list [xarray.DataArray, xarray.Dataset]
        Input data for right map, which is set to be ECCOv4. Note: Coordinates lat and lon need to be assigned.
    titles : list [str, str]
        Title for each map
    cb_label : str
        Label for colorbar
    lat : int
        Latitude of the northern edge of the map
        
    Returns
    -------
    fig: Figure instance
    '''

    # Parameters for cartopy map
    theta = np.linspace(0, 2*np.pi, 100)
    center, radius = [0.5, 0.5], 0.5
    verts = np.vstack([np.sin(theta), np.cos(theta)]).T
    circle = mpath.Path(verts * radius + center)

    # Title string for central box
    box_titel_str = 'total = %.1f Sv\npos. = %.1f Sv\nneg. = %.1f Sv\non shlf = %.1f Sv\noff shlf = %.1f Sv'

    fig = plt.figure(figsize=(12,6))
    fig.subplots_adjust(left=0.1, right=0.89, bottom=0.1, top=0.9, hspace=0.1, wspace=0.02)
    
    ax1 = fig.add_subplot(1, 2, 1, projection=cartopy.crs.SouthPolarStereo(central_longitude=0.),
                          facecolor='lightgrey')
    ax1.set_boundary(circle, transform=ax1.transAxes)
    ax1.set_extent([-300, 60, lat, -90], cartopy.crs.PlateCarree())
    p = datal[0].where(datal[1].wet==1).plot(ax=ax1, x='x', y='y',vmin=vrange[0], vmax=vrange[1],cmap='RdBu_r',
                                             transform=cartopy.crs.PlateCarree(),add_labels=False,add_colorbar=False)
    ax1.add_artist(AnchoredText(box_titel_str% \
                                (np.round((datal[0]*1e-6*datal[1]['areacello']).sum(['x','y']).values,1),
                                 np.round((datal[0].where(datal[0]>0)*1e-6*datal[1]['areacello'])\
                                          .sum(['x','y']).values,1),
                                 np.round((datal[0].where(datal[0]<0)*1e-6*datal[1]['areacello'])\
                                          .sum(['x','y']).values,1),
                                 np.round((datal[0].where(datal[1].deptho<1000)*1e-6*datal[1]['areacello'])\
                                          .sum(['x','y']).values,1),
                                 np.round((datal[0].where(datal[1].deptho>=1000)*1e-6*datal[1]['areacello'])\
                                          .sum(['x','y']).values,1)),
                                loc='center', prop={'size': 11, 'weight': 'bold'}, frameon=False,
                                bbox_to_anchor=(0.58, 0.52), bbox_transform=ax1.transAxes))

    ax1.set_title(titles[0],fontsize=12)
    
    ax2 = fig.add_subplot(1, 2, 2, projection=cartopy.crs.SouthPolarStereo(central_longitude=0.),
                          facecolor='lightgrey')
    ax2.set_boundary(circle, transform=ax2.transAxes)
    ax2.set_extent([-300, 60, lat, -90], cartopy.crs.PlateCarree())
    
    datar[0].where(datar[1].wet==1).sel(x=slice(None,29))\
    .plot(ax=ax2, x='lon', y='lat', vmin=vrange[0], vmax=vrange[1], cmap='RdBu_r', add_labels=False, add_colorbar=False,
          transform=cartopy.crs.PlateCarree())
    datar[0].where(datar[1].wet==1).sel(x=slice(30,179),y=slice(60,None))\
    .plot(ax=ax2, x='lon', y='lat', vmin=vrange[0], vmax=vrange[1], cmap='RdBu_r', add_labels=False, add_colorbar=False,
          transform=cartopy.crs.PlateCarree())
    datar[0].where(datar[1].wet==1).sel(x=slice(180,212))\
    .plot(ax=ax2, x='lon', y='lat', vmin=vrange[0], vmax=vrange[1], cmap='RdBu_r', add_labels=False, add_colorbar=False,
          transform=cartopy.crs.PlateCarree())
    datar[0].where(datar[1].wet==1).sel(x=slice(213,217),y=slice(49,None))\
    .plot(ax=ax2, x='lon', y='lat', vmin=vrange[0], vmax=vrange[1], cmap='RdBu_r', add_labels=False, add_colorbar=False,
          transform=cartopy.crs.PlateCarree())
    datar[0].where(datar[1].wet==1).sel(x=slice(218,None))\
    .plot(ax=ax2, x='lon', y='lat', vmin=vrange[0], vmax=vrange[1], cmap='RdBu_r', add_labels=False, add_colorbar=False,
          transform=cartopy.crs.PlateCarree())
    ax2.add_artist(AnchoredText(box_titel_str% \
                                (np.round((datar[0]*1e-6*datar[1]['areacello']).sum(['x','y']).values,1),
                                 np.round((datar[0].where(datar[0]>0)*1e-6*datar[1]['areacello'])\
                                          .sum(['x','y']).values,1),
                                 np.round((datar[0].where(datar[0]<0)*1e-6*datar[1]['areacello'])\
                                          .sum(['x','y']).values,1),
                                 np.round((datar[0].where(datar[1].deptho<1000)*1e-6*datar[1]['areacello'])\
                                          .sum(['x','y']).values,1),
                                 np.round((datar[0].where(datar[1].deptho>=1000)*1e-6*datar[1]['areacello'])\
                                          .sum(['x','y']).values,1)),
                                loc='center', prop={'size': 11, 'weight': 'bold'}, frameon=False,
                                bbox_to_anchor=(0.58, 0.52), bbox_transform=ax2.transAxes))

    ax2.set_title(titles[1],fontsize=12)
    
    cax = fig.add_axes([0.9, 0.1 , 0.02, 0.8])
    cb = fig.colorbar(p, cax=cax, orientation='vertical', shrink=0.5, pad=0.1)
    cb.set_label(cb_label, fontsize=12)
    cb.ax.tick_params(labelsize=10)
    
    return fig


def map_comparison_so_3by1_ecco(datal, datac, datar, titles, cb_label, lat = -50, 
                                vrange = [None, None], fsize = [18,6]):
    '''
    Generates a 3-by-1 figure using SouthPolarStereo cartopy projection.
    
    Parameters
    ----------
    datal : list [xarray.DataArray, xarray.Dataset]
        Input data for left map
    datac : list [xarray.DataArray, xarray.Dataset]
        Input data for center map
    datar : list [xarray.DataArray, xarray.Dataset]
        Input data for right map, which is set to be ECCOv4. Note: Coordinates lat and lon need to be assigned.
    titles : list [str, str]
        Title for each map
    cb_label : str
        Label for colorbar
    lat : int
        Latitude of the northern edge of the map
        
    Returns
    -------
    fig: Figure instance
    '''

    circle = get_so_map_boundary()

    fig = plt.figure(figsize=fsize)
    fig.subplots_adjust(left=0.1, right=0.89, bottom=0.1, top=0.9, hspace=0.1, wspace=0.02)
    
    ax1 = fig.add_subplot(1, 3, 1, projection=cartopy.crs.SouthPolarStereo(central_longitude=0.),facecolor='grey')
    ax1.set_boundary(circle, transform=ax1.transAxes)
    ax1.set_extent([-300, 60, lat, -90], cartopy.crs.PlateCarree())
    p = datal[0].where(datal[1].wet==1).plot(ax=ax1, x='x', y='y',vmin=vrange[0], vmax=vrange[1],cmap='RdBu_r',
                                             transform=cartopy.crs.PlateCarree(),add_labels=False,add_colorbar=False)
    ax1.set_title(titles[0],fontsize=12)
    
    ax2 = fig.add_subplot(1, 3, 2, projection=cartopy.crs.SouthPolarStereo(central_longitude=0.),facecolor='grey')
    ax2.set_boundary(circle, transform=ax2.transAxes)
    ax2.set_extent([-300, 60, lat, -90], cartopy.crs.PlateCarree())
    datac[0].where(datac[1].wet==1).plot(ax=ax2, x='x', y='y', vmin=vrange[0], vmax=vrange[1], cmap='RdBu_r',
                                         transform=cartopy.crs.PlateCarree(),add_labels=False,add_colorbar=False)
    ax2.set_title(titles[1],fontsize=12)
    
    ax3 = fig.add_subplot(1, 3, 3, projection=cartopy.crs.SouthPolarStereo(central_longitude=0.),facecolor='grey')
    ax3.set_boundary(circle, transform=ax3.transAxes)
    ax3.set_extent([-300, 60, lat, -90], cartopy.crs.PlateCarree())
    datar[0].where(datar[1].wet==1).sel(x=slice(None,29))\
    .plot(ax=ax3, x='lon', y='lat', vmin=vrange[0], vmax=vrange[1], cmap='RdBu_r', add_labels=False, 
          add_colorbar=False, transform=cartopy.crs.PlateCarree())
    datar[0].where(datar[1].wet==1).sel(x=slice(30,179),y=slice(60,None))\
    .plot(ax=ax3, x='lon', y='lat', vmin=vrange[0], vmax=vrange[1], cmap='RdBu_r', add_labels=False, 
          add_colorbar=False, transform=cartopy.crs.PlateCarree())
    datar[0].where(datar[1].wet==1).sel(x=slice(180,212))\
    .plot(ax=ax3, x='lon', y='lat', vmin=vrange[0], vmax=vrange[1], cmap='RdBu_r', add_labels=False, 
          add_colorbar=False, transform=cartopy.crs.PlateCarree())
    datar[0].where(datar[1].wet==1).sel(x=slice(213,217),y=slice(49,None))\
    .plot(ax=ax3, x='lon', y='lat', vmin=vrange[0], vmax=vrange[1], cmap='RdBu_r', add_labels=False, 
          add_colorbar=False, transform=cartopy.crs.PlateCarree())
    datar[0].where(datar[1].wet==1).sel(x=slice(218,None))\
    .plot(ax=ax3, x='lon', y='lat', vmin=vrange[0], vmax=vrange[1], cmap='RdBu_r', add_labels=False, 
          add_colorbar=False, transform=cartopy.crs.PlateCarree())
    ax3.set_title(titles[2],fontsize=12)
    
    cax = fig.add_axes([0.9, 0.1 , 0.02, 0.8])
    cb = fig.colorbar(p, cax=cax, orientation='vertical', shrink=0.5, pad=0.1)
    cb.set_label(cb_label, fontsize=12)
    cb.ax.tick_params(labelsize=10)
    
    return fig

