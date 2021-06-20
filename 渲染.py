import seaborn as sns
import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap

# deal with the map element, such as coordinate reference
import cartopy.crs as ccrs
# import cartopy.feature as cfeature
# import cartopy.io.shapereader as shpreader

import rasterio as rio
from rasterio import plot
from rasterio.warp import calculate_default_transform, reproject, Resampling

import pathlib

import geopandas as gpd
import pandas as pd
from pandas.plotting import register_matplotlib_converters
import os
from rasterio.plot import plotting_extent
import earthpy as et
import earthpy.spatial as es
import earthpy.plot as ep
import pathlib
register_matplotlib_converters()


def blank_axes(ax):

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    #
    ax.yaxis.set_ticks_position('none')
    ax.xaxis.set_ticks_position('none')

    ax.tick_params(labeltop=False, labelright=False, \
                   top=False, right=False, which='both', \
                   direction='in', width=2, length=4, pad=2, \
                   labelsize=10)

    return ax


# end blank_axes()

def Plot_legend(ax, colors, className, ncol, colSpacing=2):
    handles = []
    # for each cartopy defined color, draw a patch, append handle to list, and append color name to names list
    for c in zip(colors, className):
        #    patch = mpatches.Patch(color=cfeature.COLORS[c], label=c)
        patch = mpatches.Patch(color=c[0], label=c[1])
        handles.append(patch)
    # end for
    ax.legend(handles, className, ncol=ncol, columnspacing=colSpacing)
    #    ax.set_title('Legend',loc='left')
    return ax


def Plot_spt_distr_obs(ax_map, gdf_TA_grid, norm, cmap, study_vector):
    with rio.open(gdf_TA_grid) as raster_src:
        raster_data = raster_src.read()
        # print(raster_src.crs)
        study_vector_utmz13 = study_vector.to_crs(raster_src.crs)
        # Create plotting extent from DatasetReader object
        raster_plot_extent = plotting_extent(raster_src)
    study_vector_utmz13.plot(ax=ax_map,alpha=0.85)
    ep.plot_bands(raster_data, ax=ax_map, cmap='Blues', extent=raster_plot_extent)


def plotRaster(ax, impath, cmap,study_vector):
    with rio.open(impath) as raster_src:
        raster_data = raster_src.read()
        # print(raster_src.crs)
        study_vector_utmz13 = study_vector.to_crs(raster_src.crs)
        # Create plotting extent from DatasetReader object
        raster_plot_extent = plotting_extent(raster_src)
    study_vector_utmz13.plot(ax=ax,color='#f21170')
    ep.plot_bands(raster_data, ax=ax, cmap=cmap,extent=raster_plot_extent,cbar=False)


def Plot_NorthArrow(pos, sNorthArrowImgPath):
    ax = plt.axes(pos)

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.yaxis.set_ticks_position('none')
    ax.xaxis.set_ticks_position('none')

    ax.tick_params(labelbottom=False, labeltop=False, labelleft=False, labelright=False, \
                   bottom=False, top=False, left=False, right=False)
    #     img_extent = (minY, maxY, minX, maxX)
    img = plt.imread(sNorthArrowImgPath)

    ax.imshow(img, origin='upper', cmap='gray')


if __name__ == "__main__":
    WORK_DIR = r'E:\Desktop\空间数据处理与系统开发\实验'

    data_dir = r'E:\Desktop\空间数据处理与系统开发\实验\dataplot'

    data_root = pathlib.Path(data_dir)

    # results map pathname
    resultMap = data_root.joinpath('figures', '出图.png')
    l_cropcode = [1106, 1108, 1501, 3300]
    l_color = ['#ff9292', '#ffb4b4', '#ffdcdc', '#eec4c4']
    l_cropName = ['Spring Maize', 'Maize for Seed', 'Cotton', 'Timberland']

    # =============================================================================
    '''
    2 Set and Read the data
    '''
    # =============================================================================
    # 1 study area
    studyarea_borders = data_root.joinpath('cropsample.shp')
    gdf_border = gpd.read_file(studyarea_borders)
 
    # 2 cropMap data
    cropMapPath_reprej = data_root.joinpath('pred.tif')
    # 3 study area grid data
    grid = data_root.joinpath('prob.tif')
    title= data_root.joinpath('title.png')

    '''
    4 Layout and drawing
      using plt.axes(rect=[x,y,width,heigh])
    '''
    # =============================================================================
    # fig = plt.figure(figsize=(9, 6))

    '''
    #1 Define the global layout
    '''
    filepath_northArrow=data_root.joinpath('NorthArrow.jpg')
    fig, (ax_crop,ax_grid) = plt.subplots(1,2,figsize=(10, 10))
    # # [x0, y0, width, height]
    lf_layout = {
        'crop': [0.01, 0.10, 0.45, 0.9],
        'grid': [0.47, 0.10, 0.45, 0.9],
        'legend': [0.20, 0.21, 0.45, 0.06],
        'clrBar_grid': [0.49, 0.03, 0.4, 0.02],
        'NorthArrow': [0.49, 0.59, 0.05, 0.08],
        'title': [0.32, 0.68, 0.4, 0.1]
    }
    # print("1 done")
    # # create border and grid subplot
    # ax_crop = plt.axes(lf_layout['crop'], projection=ccrs.PlateCarree())
    ax_legend = plt.axes(lf_layout['legend'])
    Plot_NorthArrow(lf_layout['NorthArrow'], filepath_northArrow)
    Plot_NorthArrow(lf_layout['title'], title)
    # ax_grid = plt.axes(lf_layout['grid'], projection=ccrs.PlateCarree())
    # ax_clrbar_grid = plt.axes(lf_layout['clrBar_grid'])

    # define some visible parameters
    ticks_prms = {'x_num': 3,
                  'y_num': 5,
                  'decimals': 2
                  }
    visible_area = [0, 0.28, 0, 0]  # percentage of minX,minY,maxX,maxY

    '''
    #2 plot crop map 
    '''
    # Define personalized color spaces
    cmap = ListedColormap(sns.color_palette(l_color).as_hex())

    # Define the mapping relationship between color space and value space
    norm = mpl.colors.Normalize(vmin=0, vmax=3)
    print("2 done")
    # add raster data

    plotRaster(ax_crop, cropMapPath_reprej, cmap, gdf_border)
    print("3 done")
    blank_axes(ax_legend)
    ax_legend.tick_params(length=0, labelbottom=False, labelleft=False)
    Plot_legend(ax_legend, l_color, l_cropName, ncol=4, colSpacing=1.15)

    # add title
    ax_crop.annotate('pred.tif&cropsample.shp'.title(), xy=(0.0, 1.03), rotation=0, fontsize=14,
                     xycoords='axes fraction')

    '''
    #3 plot grid data
    '''
    # define a color range
    # norm = mpl.colors.Normalize(vmin=0, vmax=1)
    # cmap = mpl.cm.Reds

    # add border
    Plot_spt_distr_obs(ax_grid,grid,norm,cmap,gdf_border)

    # add title
    ax_grid.annotate('prob.tif&cropsample.shp'.title(), xy=(0.0, 1.03), rotation=0, fontsize=14,
                     xycoords='axes fraction')

    '''
    #6 save and show the result of map
    '''
    plt.savefig(resultMap, dpi=300, bbox_inches='tight')  # 保存图片
