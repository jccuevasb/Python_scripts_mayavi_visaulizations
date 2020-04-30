#!/usr/bin/env python
"""Function to plot the density field in the x-y plane, the
magnetic field stremalines, and the current field in the 
y-z plane

List of possible variables in openggcm:
    bx, by, bz : magnetic field vector (nT) 
    vx, vy, vz : velocity vector (km/s)
    rr : density (cm-3)
    pp : pressure (nPa)
    resis : resistivity 
    xjx, xjy, xjz : electric current vector (microA/m2)
"""

import numpy as np 
import vpfplot as vpf
import vis_lab as vlab
import create_cm as ccm
import mayavi
from mayavi import mlab 


def vrrxjb_field_plot(gridx,gridy,gridz,fdict,xlim):
    """
    Args:
       xgrid: 1D (480) array with the x-coordinates
       ygrid: 1D (180) array with the y-coordinates
       zgrid: 2D (180) array with the z-coordinates
       fdict: dictionary of arrays of the form
       {"field":Array(480x180x180)}
    """
    ##---Load current vectors
#    xjx=fdict["xjx"][0:xlim-1,:,:]
#    xjy=fdict["xjy"][0:xlim-1,:,:]
#    xjz=fdict["xjz"][0:xlim-1,:,:]
    ###
    xjx=fdict["xjx"]
    xjy=fdict["xjy"]
    xjz=fdict["xjz"]
    #---Threshold the data to plot just most significative values
    xjnorm=np.sqrt(xjx**2 + xjy**2 +xjz**2)
    xjmax=np.max(xjnorm)
    cthres=8e-3
    xjx[xjnorm<cthres*xjmax]=0.0    ##np.nan, 0 does not work, it still plots them 
    xjy[xjnorm<cthres*xjmax]=0.0
    xjz[xjnorm<cthres*xjmax]=0.0
    ##--------------
    xjx=xjx/xjnorm
    xjy=xjy/xjnorm
    xjz=xjz/xjnorm
    ###-----------Plot current field
    #src_vec=mlab.pipeline.vector_field(gridx,gridy,gridz,fdict["xjx"],
    #        fdict["xjy"],fdict["xjz"],scalars=vnorm,name='Current Field')
    src_vec=vpf.vecfield2point_source(xjx,xjy,xjz, 
                          gridx[0:xlim-1], gridy, gridz, 'vector')
    ###---Insert a cut for the current field normal to the y axis
    vcpcur=mlab.pipeline.vector_cut_plane(src_vec,plane_orientation='y_axes',
                       opacity=1.0,transparent=True,
                       scale_factor=4,mask_points=20,colormap='autumn',
                       view_controls=False,line_width=1.45,
                       scale_mode= 'vector',
                       mode='arrow')
    ##-Color NaN values with white color
    #vcpcur.module_manager.vector_lut_manager.lut.nan_color=(0.0,0.0,0.0,0.0)
    #vcpcur.module_manager.vector_lut_manager.data_range=[5e-5,xjmax]
    vcpcur.module_manager.vector_lut_manager.lut.scale='log10'
    vcpcur.glyph.mask_points.random_mode=True   ##random sampling mask points
#    vcpcur.glyph.scale_mode= 'data_scaling_off'
    vcpcur.implicit_plane.origin = [0.0, 0.0, 0.0]
##----Plot solar wind velocity field
     ##---Load current vectors
#    vx=fdict["vx"][0:xlim-1,:,:]
#    vy=fdict["vy"][0:xlim-1,:,:]
#    vz=fdict["vz"][0:xlim-1,:,:]
    #
    vx=fdict["vx"]
    vy=fdict["vy"]         
    vz=fdict["vz"]         
    #---Threshold the data to plot just most significative values
    vnorm=np.sqrt(vx**2 + vy**2 + vz**2)
    vmax=np.max(vnorm)
    ##---normalize vectors
    thresvel=2e-1
    vx[vnorm<thresvel*vmax]=0.0 
    vy[vnorm<thresvel*vmax]=0.0
    vz[vnorm<thresvel*vmax]=0.0
##    ##--------------
    vx=vx/vnorm
    vy=vy/vnorm
    vz=vz/vnorm

    swvf_src=vpf.vecfield2point_source(vx,vy,vz, 
                          gridx[0:xlim-1], gridy, gridz, 'vector')
   # swv3dfield=mlab.pipeline.vectors(swvf_src,opacity=0.5,transparent=True,
   #                     scale_mode= 'vector',mask_points=5e4,scale_factor=4,
   #                     mode='arrow',colormap='winter',
   #                     line_width=1.45)
    #colormap='black-white'
    #color=(1,1,1)   #white
    swv3dfield=mlab.pipeline.vector_cut_plane(swvf_src,plane_orientation='z_axes',
                       opacity=1.0,transparent=True,
                       scale_factor=4,mask_points=100,colormap='RdYlGn',
                       view_controls=False, line_width=1.45,
                       scale_mode= 'vector',
                       mode='arrow')
    ##---Settings
#    swv3dfield.module_manager.vector_lut_manager.lut.nan_color=(0.0,0.0,0.0,0.0)
#    swv3dfield.module_manager.vector_lut_manager.data_range=[2e-1*vmax,vmax]
    swv3dfield.module_manager.vector_lut_manager.lut.scale='log10'
    swv3dfield.glyph.mask_points.random_mode=True   ##random sampling mask points


#    ###-----Plot the earth   
#    vlab.plot_blue_marble(r=2, lines=False, ntheta=64, nphi=128, 
#                                      crd_system='gse')
#    vlab.plot_3dearth(radius=2.1,night_only=True,opacity=0.8,
#                                crd_system='mhd') 
    
#    ###--------Scalar cut plane
##    src_sca=mlab.pipeline.scalar_field(gridx,gridy,gridz,fdict["rr"],
##                                      name='Densitiy field')
    src_sca= vpf.field2point_source(fdict["rr"][0:xlim-1,:,:], 
            gridx[0:xlim-1], gridy, gridz, 'scalar')

    #lines = mlab.pipeline.triangle_filter(src_sca)
    lines =  mlab.pipeline.delaunay3d(src_sca)
    print('lines= ',lines)
    print('lines attributes= ',dir(lines))

    cut_plane = mlab.pipeline.scalar_cut_plane(src_sca,
                plane_orientation='z_axes', opacity=0.7,transparent=True,
                view_controls=False)
#    ##---Define scalar cut plane settings
    mvi_lut = cut_plane.module_manager.scalar_lut_manager.lut
    #cut_plane.module_manager.scalar_lut_manager.data_range=[1,15]
##---Create custom color map
##https://github.com/enthought/mayavi/issues/622
    plasma_cm=ccm.create_color_map()
   # print("lut properties= ",mvi_lut.table)
   # print("lut attributes= ",dir(mvi_lut.table))
    mvi_lut.table=plasma_cm*255
    mvi_lut.scale = 'log10'
#    scabar=mlab.scalarbar(cut_plane,title='Density',orientation='horizontal')
##    cut_plane.actor.actor.scale=(3.0,1.0,1.0)
    cut_plane.actor.actor.force_opaque=True
    #cut_plane.module_manager.lut_data_mode="point data"  #i.e. scalar
#    cut_plane.enable_contours= True         #--Enable contours
    cut_plane.contour.auto_contours= True   #--Isovalues auto well defined
#   # cut_plane.enable_warp_scalar= True      #--3d contour plot
    #cut_plane.warp_scale= 0.7               #--3d contour plot
#    #cut_plane.warp_scalar.warp_scale=0.7
    cut_plane.contour.number_of_contours = 250
#    mlab.draw()

    ##-----Plot isosurface
    isosurf_density = mlab.pipeline.iso_surface(src_sca,opacity=0.2,
                                colormap='RdYlGn',contours=[3,3.001],
                                transparent=True)
    ##---Define isosurface settings
    #isosurf_density.actor.mapper.scalar_mode='use_cell_data'
    #isosurf_density.actor.property.color=(240/255,120/255,15/255)
#    isosurf_density = mlab.pipeline.iso_surface(src_sca,opacity=0.2,
#                                colormap='jet',
#                                transparent=True)

#    scabar=mlab.scalarbar(cut_plane,title='Density',orientation='horizontal')
#    
#    #------Plot magnetic field
#   # src_mfvec=mlab.pipeline.vector_field(gridx,gridy,gridz,fdict["bx"],
#   #                       fdict["by"],fdict["bz"],name='Magnetic field')
#    src_mfvec=vpf.vecfield2point_source(fdict["bx"][0:xlim-1,:,:], 
#           fdict["by"][0:xlim-1,:,:],fdict["bz"][0:xlim-1,:,:], 
    src_mfvec=vpf.vecfield2point_source(fdict["bx"], fdict["by"],
            fdict["bz"], gridx[0:xlim-1], gridy, gridz, 'vector')
    norm_mfvec=mlab.pipeline.extract_vector_norm(src_mfvec)
#
#    ###-----Insert streamlines to track magnetic field
    mf_strlines = mlab.pipeline.streamline(norm_mfvec, seedtype='sphere',
            integration_direction='both', line_width=1,
                                         colormap='gray')
    mfsl_lut=mf_strlines.module_manager.scalar_lut_manager.lut
    mfsl_lut.scale = 'log10'
    mf_strlines.module_manager.lut_data_mode="cell data"
    mf_strlines.module_manager.scalar_lut_manager.data_range=[10, 40]
    mf_strlines.stream_tracer.maximum_propagation = 125
    mf_strlines.stream_tracer.integrator_type='runge_kutta45'
    mf_strlines.stream_tracer.initial_integration_step=0.001
#    #---Align with the y axis
    mf_strlines.seed.widget.enabled = False
    xcoord=0
    ycoord=14
    zcoord=12
    mf_strlines.seed.widget.center = [xcoord, 0, 0]
    #mf_strlines.seed.widget.origin = [xcoord, -ycoord, -zcoord]
    mf_strlines.seed.widget.phi_resolution = 10        ##--sphere
    mf_strlines.seed.widget.theta_resolution = 8      ##--sphere
    #mf_strlines.seed.widget.normal = [1, 0, 0]          ##--plane
    #mf_strlines.seed.widget.resolution = 8              ##--plane
#    mf_strlines.seed.widget.radius = 3
#    mf_strlines.streamline_type = "ribbon" # tube, ribbon or line
    mf_strlines.streamline_type = "tube" # tube, ribbon or line
    mf_strlines.tube_filter.number_of_sides = 4
    mf_strlines.tube_filter.radius = 0.2
    mf_strlines.ribbon_filter.width = 0.2
    #mf_strlines.seed.widget.point1 = [xcoord, ycoord,-zcoord]        ##--plane
    #mf_strlines.seed.widget.point2 = [xcoord, -ycoord,zcoord]        ##--plane

#    return




