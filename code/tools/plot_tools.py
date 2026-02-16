import matplotlib.pyplot as plt
import numpy as np
import os
import xarray as xr
import cartopy.crs as ccrs
import matplotlib.ticker as mticker
from matplotlib.colors import LogNorm
plt.rcParams["axes.edgecolor"] = "black"
plt.rcParams["axes.linewidth"] = 1.5
import cartopy.feature as cfeature

def plot_map_zoom(var,lon,lat,title=None,vmax=None,vmin=None,colorbar_label=None,axs=None,fig=None,zoom_extent=[-20, 0, 40, 60],cmap='RdBu_r',zoom_ax = [0.25, 0.10, 0.5, 0.5],lat_mask=5,lognorm=None):

    if not axs:
        fig, axs = plt.subplots(nrows=1,ncols=1,
                        subplot_kw={'projection': ccrs.PlateCarree()},
                        figsize=(11*1,7.5*1))
        #p0 = plt.pcolormesh(lon2D, lat2D, norm_uv, cmap='jet',vmax=1)
    #vmax_glob=np.nanmax(var)
    #vmin_glob=np.nanmin(var)

        
    if lat_mask is not None :
        axs.hlines(lat_mask,-180,180,color='k', alpha=1, linestyle='--')
        axs.hlines(-lat_mask,-180,180,color='k', alpha=1, linestyle='--')

    if lognorm:
        p0 = plt.pcolormesh(lon, lat, var, cmap=cmap,norm=LogNorm(vmin=vmin, vmax=vmax))
    else:  
        p0 = plt.pcolormesh(lon, lat, var, cmap=cmap,vmax=vmax,vmin=vmin)

    if title:
        axs.set_title(title)

    axs.coastlines(resolution='10m', lw=0.5)
    #axs.add_feature(cfeature.LAND.with_scale('50m'), facecolor='#EEEEEE', edgecolor='face',alpha=1)

    # optional add grid lines
    p0.axes.gridlines(color='black', alpha=0., linestyle='--')

    # draw parallels/meridiens and write labels
    gl = p0.axes.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                            linewidth=0.1, color='black', alpha=0.5, linestyle='--')
    # adjust labels to taste
    gl.top_labels = True
    gl.right_labels = False
    gl.bottom_labels = False
    gl.left_labels = True
    #gl.xlabels_bottom = False
    gl.ylocator = mticker.FixedLocator([-90, -60, -30, 0, 30, 60, 90])
    gl.xlocator = mticker.FixedLocator([-180, -60, 0,  60, 180])
    #gl.xformatter = LONGITUDE_FORMATTER
    #gl.yformatter = LATITUDE_FORMATTER
    gl.xlabel_style = {'size': 10, 'color': 'black'}
    gl.ylabel_style = {'size': 10, 'color': 'black'}

    #get size and extent of axes:
    axpos = axs.get_position()
    pos_x = axpos.x0+axpos.width + 0.01# + 0.25*axpos.width
    pos_y = axpos.y0
    cax_width = 0.02
    cax_height = axpos.height
    #create new axes where the colorbar should go.
    #it should be next to the original axes and have the same height!
    pos_cax = fig.add_axes([pos_x,pos_y,cax_width,cax_height])
    cbar=plt.colorbar(p0, cax=pos_cax)
    if colorbar_label:
        cbar.set_label(colorbar_label)

    #zoom_extent = [-20, 0, 40, 60]  # Exemple de zoom sur l'Europe
   
    lon_mg,lat_mg = np.meshgrid(lon,lat)
    var_f = np.where(lat_mg > zoom_extent[2] , var, np.nan)
    var_f = np.where(lat_mg < zoom_extent[3] , var_f, np.nan)
    var_f = np.where(lon_mg < zoom_extent[1] , var_f, np.nan)
    var_f = np.where(lon_mg > zoom_extent[0] , var_f, np.nan)

    if not(vmax):
        vmax=np.nanmax(var_f)
        vmin=np.nanmin(var_f)

    # Ajouter un zoom sur une région spécifique
    ax_zoom = fig.add_axes(zoom_ax, projection=ccrs.PlateCarree())
    ax_zoom.set_extent(zoom_extent, crs=ccrs.PlateCarree())  # Exemple de zoom sur l'Europe
    ax_zoom.coastlines()

    if lognorm:
        contour_zoom = ax_zoom.pcolormesh(lon, lat, var, cmap=cmap,transform=ccrs.PlateCarree(),norm=LogNorm(vmin=vmin, vmax=vmax))
    else: 
        contour_zoom = ax_zoom.pcolormesh(lon, lat, var, cmap=cmap,transform=ccrs.PlateCarree(),vmax=vmax,vmin=vmin)

    #cax = fig.add_axes([ax_zoom.get_position().x1+0.01,ax_zoom.get_position().y0,0.02,ax_zoom.get_position().height])
    cax = fig.add_axes([ax_zoom.get_position().x0,ax_zoom.get_position().y0-0.03,ax_zoom.get_position().width,0.02])
    cbar = plt.colorbar(contour_zoom,cax=cax,orientation="horizontal")


    # Ajouter les contours de la zone zoomée sur la carte principale
    zoom_rect = plt.Rectangle((zoom_extent[0], zoom_extent[2]),
                            zoom_extent[1] - zoom_extent[0],
                            zoom_extent[3] - zoom_extent[2],
                            linewidth=2, edgecolor='k', facecolor='none',
                            transform=ccrs.PlateCarree())
    axs.add_patch(zoom_rect)


def plot_uv_map(uv,lon2D,lat2D,cmap=plt.cm.RdBu_r,vmax=1,vmin=-1,title=None,colorbar_title="norm(U) [m/s]",axs=None,fig=None,colorbar=True,lat_mask=None):

    if not axs:

        fig, axs = plt.subplots(nrows=1,ncols=1,
                            subplot_kw={'projection': ccrs.PlateCarree()},
                            figsize=(11*1,5*1))

    axs.add_feature(
    cfeature.LAND,
    facecolor='lightgray',  # Couleur de remplissage
    edgecolor='black',      # Couleur des bordures (optionnel)
    zorder=2                # Ordre de dessin (0 = en arrière-plan)
    )
    
    if lat_mask is not None:
        axs.hlines(lat_mask,np.min(lon2D),np.max(lon2D),color='k', alpha=1, linestyle='--')
        axs.hlines(-lat_mask,np.min(lon2D),np.max(lon2D),color='k', alpha=1, linestyle='--')

    if title:
        axs.set_title(title)

    #p0 = plt.pcolormesh(lon2D, lat2D, norm_uv, cmap='jet',vmax=1)
    p0 = axs.pcolormesh(lon2D, lat2D, uv, cmap=cmap,vmax=vmax,vmin=vmin)  

    axs.coastlines(resolution='10m', lw=0.5)
    #axs.add_feature(cfeature.LAND.with_scale('50m'), facecolor='#EEEEEE', edgecolor='face',alpha=1)

    # optional add grid lines
    p0.axes.gridlines(color='black', alpha=0., linestyle='--')

    # draw parallels/meridiens and write labels
    gl = p0.axes.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                            linewidth=0.2, color='black', alpha=0.8, linestyle='--')
    # adjust labels to taste
    gl.top_labels = False
    gl.right_labels = False
    gl.bottom_labels = True
    gl.left_labels = True
    #gl.ylocator = mticker.FixedLocator([-90, -60, -30, 0, 30, 60, 90])
    #gl.xlocator = mticker.FixedLocator([-180, -60, 0,  60, 180])
    gl.xlabel_style = {'size': 10, 'color': 'black'}
    gl.ylabel_style = {'size': 10, 'color': 'black'}

    #get size and extent of axes:
    axpos = axs.get_position()
    pos_x = axpos.x0+axpos.width + 0.01# + 0.25*axpos.width
    pos_y = axpos.y0
    cax_width = 0.02
    cax_height = axpos.height

    if colorbar:
        #create new axes where the colorbar should go.
        #it should be next to the original axes and have the same height!
        pos_cax = fig.add_axes([pos_x,pos_y,cax_width,cax_height])
        cbar=plt.colorbar(p0, cax=pos_cax)
        cbar.set_label(colorbar_title)

    #cax = fig.add_axes([0.92, 0.37, 0.02, 0.25])
    #cbar = fig.colorbar(p0, cax=axs, orientation='vertical')
    #cax.set_ylabel('Number of data', fontweight='bold')


    
def plot_map_zoom_only(var,lon,lat,title=None,vmax=None,vmin=None,colorbar_label=None,axs=None,fig=None,zoom_extent=[-20, 0, 40, 60],cmap='RdBu_r',zoom_ax = [0.25, 0.10, 0.5, 0.5],lat_mask=5):

    if not axs:
        fig, axs = plt.subplots(nrows=1,ncols=1,
                        subplot_kw={'projection': ccrs.PlateCarree()},
                        figsize=(11*1,7.5*1))
        #p0 = plt.pcolormesh(lon2D, lat2D, norm_uv, cmap='jet',vmax=1)
    #vmax_glob=np.nanmax(var)
    #vmin_glob=np.nanmin(var)

        # Ajouter un zoom sur une région spécifique
    axs.set_extent(zoom_extent, crs=ccrs.PlateCarree())  # Exemple de zoom sur l'Europe
    axs.coastlines()

    axs.hlines(lat_mask,-180,180,color='k', alpha=1, linestyle='--')
    axs.hlines(-lat_mask,-180,180,color='k', alpha=1, linestyle='--')

    lon_mg,lat_mg = np.meshgrid(lon,lat)
    var_f = np.where(lat_mg > zoom_extent[2] , var, np.nan)
    var_f = np.where(lat_mg < zoom_extent[3] , var_f, np.nan)
    var_f = np.where(lon_mg < zoom_extent[1] , var_f, np.nan)
    var_f = np.where(lon_mg > zoom_extent[0] , var_f, np.nan)

    if not(vmax):
        vmax=np.nanmax(var_f)
        vmin=np.nanmin(var_f)

    contour_zoom = axs.pcolormesh(lon, lat, var_f, cmap=cmap,transform=ccrs.PlateCarree(),vmax=vmax,vmin=vmin)

    if title:
        axs.set_title(title)

    axs.coastlines(resolution='10m', lw=0.5)
    #axs.add_feature(cfeature.LAND.with_scale('50m'), facecolor='#EEEEEE', edgecolor='face',alpha=1)

    # optional add grid lines
    contour_zoom.axes.gridlines(color='black', alpha=0., linestyle='--')

    # draw parallels/meridiens and write labels
    gl = contour_zoom.axes.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                            linewidth=0.1, color='black', alpha=0.5, linestyle='--')
    # adjust labels to taste
    gl.top_labels = True
    gl.right_labels = False
    gl.bottom_labels = False
    gl.left_labels = True
    #gl.xlabels_bottom = False
    gl.ylocator = mticker.FixedLocator([-90, -60, -30, 0, 30, 60, 90])
    gl.xlocator = mticker.FixedLocator([-180, -60, 0,  60, 180])
    #gl.xformatter = LONGITUDE_FORMATTER
    #gl.yformatter = LATITUDE_FORMATTER
    gl.xlabel_style = {'size': 10, 'color': 'black'}
    gl.ylabel_style = {'size': 10, 'color': 'black'}



    #get size and extent of axes:
    axpos = axs.get_position()
    pos_x = axpos.x0+axpos.width + 0.01# + 0.25*axpos.width
    pos_y = axpos.y0
    cax_width = 0.02
    cax_height = axpos.height
    #create new axes where the colorbar should go.
    #it should be next to the original axes and have the same height!
    pos_cax = fig.add_axes([pos_x,pos_y,cax_width,cax_height])
    cbar=plt.colorbar(contour_zoom, cax=pos_cax)
    if colorbar_label:
        cbar.set_label(colorbar_label)


def plot_map_multizoom(var,lon,lat,title=None,vmax=None,vmin=None,colorbar_label=None,axs=None,fig=None,list_zoom_extent=[[-20, 0, 40, 60]],cmap='RdBu_r',list_zoom_ax = [[0.25, 0.10, 0.5, 0.5]],lat_mask=None,lognorm=None,cbox='k'):

    if not axs:
        fig, axs = plt.subplots(nrows=1,ncols=1,
                        subplot_kw={'projection': ccrs.PlateCarree()},
                        figsize=(11*1,7.5*1))
        #p0 = plt.pcolormesh(lon2D, lat2D, norm_uv, cmap='jet',vmax=1)
    #vmax_glob=np.nanmax(var)
    #vmin_glob=np.nanmin(var)
    if lat_mask is not None :
        axs.hlines(lat_mask,-180,180,color='k', alpha=1, linestyle='--')
        axs.hlines(-lat_mask,-180,180,color='k', alpha=1, linestyle='--')

    if lognorm:
        p0 = plt.pcolormesh(lon, lat, var, cmap=cmap,norm=LogNorm(vmin=vmin, vmax=vmax))
    else:  
        p0 = plt.pcolormesh(lon, lat, var, cmap=cmap,vmax=vmax,vmin=vmin)

    if title:
        axs.set_title(title)

    axs.coastlines(resolution='10m', lw=0.5)
    #axs.add_feature(cfeature.LAND.with_scale('50m'), facecolor='#EEEEEE', edgecolor='face',alpha=1)

    # optional add grid lines
    p0.axes.gridlines(color='black', alpha=0., linestyle='--')

    # draw parallels/meridiens and write labels
    gl = p0.axes.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                            linewidth=0.1, color='black', alpha=0.5, linestyle='--')
    # adjust labels to taste
    gl.top_labels = True
    gl.right_labels = False
    gl.bottom_labels = False
    gl.left_labels = True
    #gl.xlabels_bottom = False
    gl.ylocator = mticker.FixedLocator([-90, -60, -30, 0, 30, 60, 90])
    gl.xlocator = mticker.FixedLocator([-180, -60, 0,  60, 180])
    #gl.xformatter = LONGITUDE_FORMATTER
    #gl.yformatter = LATITUDE_FORMATTER
    gl.xlabel_style = {'size': 10, 'color': 'black'}
    gl.ylabel_style = {'size': 10, 'color': 'black'}

    #get size and extent of axes:
    axpos = axs.get_position()
    pos_x = axpos.x0+axpos.width + 0.01# + 0.25*axpos.width
    pos_y = axpos.y0
    cax_width = 0.02
    cax_height = axpos.height
    #create new axes where the colorbar should go.
    #it should be next to the original axes and have the same height!
    pos_cax = fig.add_axes([pos_x,pos_y,cax_width,cax_height])
    cbar=plt.colorbar(p0, cax=pos_cax)
    if colorbar_label:
        cbar.set_label(colorbar_label)

    #zoom_extent = [-20, 0, 40, 60]  # Exemple de zoom sur l'Europe
   
    for zoom_ax,zoom_extent in zip(list_zoom_ax,list_zoom_extent):

        lon_mg,lat_mg = np.meshgrid(lon,lat)
        var_f = np.where(lat_mg > zoom_extent[2] , var, np.nan)
        var_f = np.where(lat_mg < zoom_extent[3] , var_f, np.nan)
        var_f = np.where(lon_mg < zoom_extent[1] , var_f, np.nan)
        var_f = np.where(lon_mg > zoom_extent[0] , var_f, np.nan)

        if not(vmax):
            vmax=np.nanmax(var_f)
            vmin=np.nanmin(var_f)


        # Ajouter un zoom sur une région spécifique
        ax_zoom = fig.add_axes(zoom_ax, projection=ccrs.PlateCarree())
        ax_zoom.set_extent(zoom_extent, crs=ccrs.PlateCarree())  # Exemple de zoom sur l'Europe
        ax_zoom.coastlines()

        if lognorm:
            contour_zoom = ax_zoom.pcolormesh(lon, lat, var, cmap=cmap,transform=ccrs.PlateCarree(),norm=LogNorm(vmin=vmin, vmax=vmax))
        else: 
            contour_zoom = ax_zoom.pcolormesh(lon, lat, var, cmap=cmap,transform=ccrs.PlateCarree(),vmax=vmax,vmin=vmin)

        #cax = fig.add_axes([ax_zoom.get_position().x1+0.01,ax_zoom.get_position().y0,0.02,ax_zoom.get_position().height])
        #cax = fig.add_axes([ax_zoom.get_position().x0,ax_zoom.get_position().y0-0.03,ax_zoom.get_position().width,0.02])
        #cbar = plt.colorbar(contour_zoom,cax=cax,orientation="horizontal")


        # Ajouter les contours de la zone zoomée sur la carte principale
        zoom_rect = plt.Rectangle((zoom_extent[0], zoom_extent[2]),
                                zoom_extent[1] - zoom_extent[0],
                                zoom_extent[3] - zoom_extent[2],
                                linewidth=2, edgecolor=cbox, facecolor='none',
                                transform=ccrs.PlateCarree())
        axs.add_patch(zoom_rect)
