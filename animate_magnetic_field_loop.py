import numpy as np
import readgrid as rg
import vpfplot as vpf
import vis_lab as vlab 
from pathlib import Path
#import ppfd_plot as pfp        ##biplanar density plot in the 3d space
#import rr_xj_bfd_plot as rxbp   #-Current, density and magnetic field plot
import v_rr_xj_bfields_plot as vrxb  #-Current, density and magnetic field plot
#import v_rr_xj_bfields_early_field_plot as vrxb#-Current, density and magnetic field plot
#from matplotlib.ticker import ScalarFormatter as ScForm
#import h5py
#import mayavi
from mayavi import mlab
#import moviepy.editor as mpy   # to animate the data
from pathlib import Path           
##---Create subdirectories
import os
import sys
#--colormaps http://research.endlessfernweh.com/colormaps/
"""
  List of possible variables in openggcm:
  bx, by, bz : magnetic field vector (nT)
  vx, vy, vz : velocity vector (km/s)
  rr : density (cm-3)
  pp : pressure (nPa)
  resis : resistivity
  xjx, xjy, xjz : electric current vector (microA/m2)
"""
                  
field={"bx":'bx',"by":'by',"bz":'bz',
       "vx":'vx',"vy":'vy',"vz":'vz',
       "xjx":'xjx',"xjy":'xjy',"xjz":'xjz',
       "rr":'rr',"pp":'pp',"resis":'resis'}

HOME=str(Path.home())
maindir=HOME+'/Dropbox/jc_documents/plasma_physics/'
#maindir='/home/juanes/Dropbox/jc_documents/plasma_physics/'
#disk_storage='/media/juan/b6722300-fa6f-4180-9259-ae0c6661babc/'
disk_storage='/media/juanes/E8E3-1743/'
gridstr= maindir+'cir_20170301_fb001_etanew_3d.grid2';
dataplots=maindir+'viscid/movies'
desfolder=dataplots+'/current'
filelist=sorted(os.listdir(maindir+'solar_storm_data_components/'))
#filelist=sorted(os.listdir(disk_storage+'solar_storm_data_components/'))
#filelist=["2003112_000002"]
#filelist=["2003112_002342"]
#filelist=["20031120_006120"]
Path(desfolder).mkdir(parents=True, exist_ok=True)
#data_dir=maindir+'results/';
#strext='_'+datestr+'.hdf5'
#filename='rr_3dfield.hdf5'
##--Inline functions
###--load vectorial quantities
def mimic_mgrid(xvec,yvec,zvec):
    ##--return the indices of a 3d array for x, y and z
    ix, iy, iz = np.indices([len(xvec), len(yvec), len(zvec)])
    return xvec[ix],yvec[iy],zvec[iz]

##------Variables to plot
fieldlist=["rr","xjx","xjy","xjz","bx","by","bz","vx","vy","vz"];
##------Read the 3D grid
gx,gy,gz,nx,ny,nz=rg.read_grid2(gridstr)
#gridx,gridy,gridz=mimic_mgrid(gx,gy,gz)
###---Find z=0 plane
min_index=np.argmin(abs(gz));
minval=gz[min_index];
indarr=np.where(gx<=60)
xlim=len(indarr[0])
##---Create a dictionary with the 3D field data
#fielddict={field: [] for field in fieldlist}
fielddict={field: np.ones((nx,ny,nz))*np.nan for field in fieldlist}
#----Print variables from bash script
#start=int(sys.argv[1])
#stop=int(sys.argv[2])
###--Define figure settings
width = 854
height = 526 #480
mlab.options.offscreen = True          ##--Don't show render screen
#--bgcolor=(255,255,255)/255
##fig = vlab.figure(size=(width,height), bgcolor=(0,0,0), fgcolor=(0.,0.,0.))
#fig = vlab.figure(size=(width,height), bgcolor=(0,0,0))
#fig = vlab.figure(size=(width,height), bgcolor=(0.54,0.51,0.51))
#fig.scene.anti_aliasing_frames=15
#fig.scene.jpeg_quality=100
###-----Start looping
for filestr in filelist:
#for i in range(1):
#for i in range(start, stop+1):
    #datestr='20031120_000002'
    #datestr='20031120_002342'
    #datestr='20031120_000162'
    #datestr=filelist[i]
    #print('datestr= ',datestr)
    datestr=filestr
    data_dir=maindir+'solar_storm_data_components/'+datestr+'/';
    #data_dir=disk_storage+'solar_storm_data_components/'+datestr+'/';
    
    ##---Create a dictionary with the 3D field data
    #fielddict={}
    #fielddict["rr"]=vpf.read3dfield(data_dir,"rr",filename,np.zeros((nx,ny,nz)))
    #print('filestr= ',filestr)
    for field in fieldlist:
        filename=field+'_'+datestr+'.hdf5'
        fielddict[field]=vpf.read3dfield(data_dir,field,
                               filename,np.zeros((nx,ny,nz)))
        ##Reduce grid in the data
        fielddict[field]=fielddict[field][0:xlim-1,:,:]

    print('File '+datestr+' opened succesfully')
    #--Invoke a function to plo the customized field
    #pfp.pressure_field_plot(gridx,gridy,gridz,fielddict)
    #rxbp.rrxjb_field_plot(gx,gy,gz,fielddict,xlim)
    print("creating figure")
    #fig = vlab.figure(size=(width,height), bgcolor=(0.54,0.51,0.51))
    fig = vlab.figure(size=(width,height), bgcolor=(0.10,0.10,0.10))
    vrxb.vrrxjb_field_plot(gx,gy,gz,fielddict,xlim)
    ###---Adjust scene view
    ##view(phi, theta, radius, focalpoint=None,
    ##             roll=None, reset_roll=True, figure=None)
    dist=160                         #--distance
    elev=80                          #--Elevation (0-180)
    #focal_point=(145.5, 138, 66.5)
    focal_point=(10, 0, 0)
    mlab.view(85, elev, dist, focal_point)     ##-windows view
    #mlab.roll(-175)                                  ##-rotate view
    #duration=15                     #--duration of the animation in seconds
    #fps=20                          #--frames per second
    #frames= (duration*fps)
    #frame = 0 
    ###---Time and animation captions
    xcoord=0.02        #[0-1]
    ycoord=0.90        #[0-1]
    txt1=mlab.text(xcoord,ycoord,datestr[0:8],figure=fig, width=0.1)     ##shadow
    txt2=mlab.text(0.85,ycoord,'Time= '+datestr[9:],figure=fig, width=0.12)
    txt1.property.bold=True 
    txt2.property.bold=True 
    txt1.property.shadow=True 
    txt2.property.shadow=True 
    #http://zulko.github.io/blog/2014/11/29/data-animations-with-python-and-moviepy
    ###--Animate the 3d visualization with moviepy
    #def make_frame(t):
    #    
    #    global frame
    #
    #    fig.scene.disable_render = True
    #    max_dt=300
    #    rescale_factor=max_dt//duration           #Exact division
    #    dt=t*rescale_factor                       #--t=0:1/fps:duration
    #    ##----Control rotation of the camera
    #    frame=frame+1
    #    phi=2*360.0/frames 
    #    print('phi= ', phi)
    #    fig.scene.camera.azimuth(phi)
    ##    print('dt= ', dt)
    #    mf_strlines.stream_tracer.maximum_propagation = dt 
    #    ##Accelerating Mayavi scripting use False to avoid rendering of the whole scene
    #    fig.scene.disable_render = False
    #    fig.scene._lift()        ##--This fix Segmentation fault problem screenshot
    #    return mlab.screenshot(antialiased=True)
    #
    #clip = mpy.VideoClip(make_frame, duration=duration)
    #clip.write_gif('magnetic_field_w_pp.gif', fps=fps)
    ##plt.show()
    #mlab.savefig(desfolder+'/'+datestr+'.png')
    #mlab.savefig(desfolder+'/'+datestr+'.tiff')
    #print('Saving figure')
    mlab.savefig(desfolder+'/'+datestr+'.x3d')
   # mlab.show()
    mlab.clf()
    mlab.close(all=True)
