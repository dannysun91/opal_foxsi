"""
# FOXSI IV - Optical Alignment:
#            Reads PNG Files of Laser Photos for Alignment and 
#            Compares to Simulation
# ----------------------------
# 09/18/2023
# Orlando Romeo
"""
############################################################
#  XX                    /\\                   ,'|
#XX  XX ------------ o--'O `.                /  /
#  XX                 `--.   `-----------._,' ,'
#                         \               ,---'
#                          ) )    _,--(   |
#                         /,^.---'     )/\\
#                        ((   \\     ((  \\
#                         \)   \)     \) (/   
#############################################################
# Import Third-party libraries
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import opalfox
from matplotlib.widgets import Button
#%%############################################################################
# Section 0: Initialize Parameters
###############################################################################
########### File Parameters for Specifying Image
pos            = 5       # Optic Position
run            = 1       # Testing Run Number
basedir        = os.path.dirname(os.getcwd()) # Change Base Directory for images
team           = "Heritage"        # Team Name
date           = '07-23-24'        # Date of Test
filetype       = '.JPG'            # Image File Type
########### Box surrounding optical laser image (usually a set of 4 x 4 squares)
bsf            = 1.0 # Box Scaling Factor for cases that image is not entire 4x4 squares (usually set to 1)
boxlen         = np.array([2.54*4.0])/bsf # Box Length (cm) - Can be 1 or 2 element long
# In the case of using simulated images
#boxlen         = np.array([16.35,28.16])
#boxlen         = np.array([32.7,32.2])
########### Simulation Parameters
# Parameters for Simulations ; Use X0 for Heritage
spos           = 0
xmod           = '0'
# List of different off-axis angles for each simulated images
sim_angles     = np.array([0,0.2,0.4])#np.concatenate([np.array([0,1.2,1.3,1.4]),np.arange(2,4.0,.1)])#np.arange(0,2,.2)#np.arange(0,1.2,.2)#
# Method for determining off axis angle
########### Optical Alignment Parameters
# Method type for calculating the off axis angle for each image (maxminratio, maxratio, centerratio)
offaxis_mthd   = 'centerratio'
# Parameters for analysis
proj_flg       = 1          # If set to 1, will perform projection check on original image (can skip after 1st use - set to 0)
ellp_flg       = 1          # If set to 1, will fit ellipse to points (can skip after 1st use - set to 0)
plt_flg        = 1          # If set to 1, will create plots      
# Account for Nagoya optic P4 having opposite offset (180)
ang_off        = 1
if team == 'Nagoya':    
    ang_off        = 180 
########### Data File Directory
file           = "POS"+str(pos)+"_RUN"+str(run)           # Original file name
fdir           = os.path.join(basedir,'images',team,date) # file directory
imdir          = os.path.join(fdir,file+filetype)         # Image Directory
pfile          = 'FOXSI_'+file+'_projected'               # Projected Image File Name
pdir           = os.path.join(fdir,'Image_Processing',pfile+filetype) # Projected Image file directory
efile          = os.path.join('Image_Processing',"FOXSI4_"+team+"_ellipse_"+file) # Ellipse Image File
#%%############################################################################
# Section 1.0: Analyze Data Image - Projection
###############################################################################
print('DATA: '+file)
nfile_flg = os.path.exists(pdir)         # Check if projected image file exists
# Fix Image Projection if set or if file does not exist
if proj_flg or not nfile_flg:
    # Read Image File
    image_original = opalfox.readimage(imdir)
    image_proj = opalfox.fixperspective(image_original, savedir=pdir,brighten=10)
else:
    image_proj = opalfox.readimage(pdir)
# Find Scaling Factor (Large square is 4 inches in length)
sz = np.shape(image_proj)
sf = np.array([boxlen[0]/sz[0], boxlen[-1]/sz[1]])
#%%############################################################################
# Section 1.1: Analyze Data Image - Image Brightness
###############################################################################
# Increase Image Brightness (Feel free to change any case based on images)
if team == 'Nagoya':
    image_proj = opalfox.alterimage(image_proj,1,5)
if 0: # FALSE
    image_proj = opalfox.alterimage(image_proj,1,1)
# Crop grey image with a min brightness set
gimage = opalfox.fixbrightness(image_proj,minbrightness=1,plot=0,crop=0)
# Find median brightness in each bin by r and theta (polar coordinates)
rbin = np.linspace(0, np.sqrt(2), num=501) # Radial bins
tbin = np.linspace(0, 2*np.pi, num=17)     # Theta bins
image_med = opalfox.binimage(gimage,rbin=rbin,tbin=tbin,plot=0)
#%%############################################################################
# Section 1.2: Analyze Data Image - Detection of Ring Edges
###############################################################################
# Set parameters
rings       = 2  # Number of rings to detect (one ring includes the inner and outer edge)
data_matrix = image_proj
# Initialize other parameters (user do not change)
centers  = 0
axes     = 0
angles   = 0
# Fit rings
edir = os.path.join(fdir,efile)
if ellp_flg or not os.path.exists(edir+".npz"):
    # Find median brightness in each bin by r and theta (polar coordinates)
    centers,axes,angles = opalfox.fitrings(data_matrix,image_bin=image_med,
                          rings=rings,auto=0,rbin=rbin,tbin=tbin,plot=0,
                          minbright=[0,0])
    # Save ellipse files
    np.savez(edir,centers=centers,axes=axes,angles=angles)
else:    
    ellipse = np.load(edir+".npz")
    centers = ellipse['centers']
    axes    = ellipse['axes']
    angles  = ellipse['angles'] 
#%%############################################################################
# Section 1.3: Analyze Data Image - Plot Fitted Rings
###############################################################################    
# Check to plot fitted rings images
if plt_flg:
    # Create image with superimposed rings
    fig, ax = plt.subplots(figsize=(24,16),num='Ring Fit')
    fig.canvas.manager.full_screen_toggle()
    
    # Plot Image
    plt.imshow(data_matrix,extent=[-boxlen[0]/2.0,boxlen[0]/2.0,-boxlen[-1]/2.0,boxlen[-1]/2.0])
    ax.set_title("FOXSI4 "+team+" Rings: P"+str(pos))

    ax.axis("on")
    plt.xlabel('X (cm)')
    plt.ylabel('Y (cm)')
    # Iterate each ring
    for rii in range(0,rings*2,2):
        # Find ellipse equation
        xi,yi = opalfox.ellipse(centers[rii,:]*sf-(boxlen)/2.0,axes[rii,:]*sf,  np.radians(angles[rii]))
        xo,yo = opalfox.ellipse(centers[rii+1,:]*sf-(boxlen)/2.0,axes[rii+1,:]*sf,np.radians(angles[rii+1]))
        # Plot ellipse
        plt.plot(xi,-yi, color='red', label='Fitted Ellipse')
        plt.plot(xo,-yo, color='red', label='Fitted Ellipse')
        # Plot inner/outer ring centers
        if rii == 0:
            plt.plot(centers[rii,0]*sf[0]-boxlen[0]/2.0,(boxlen[-1]/2.0-centers[rii,1]*sf[1]), 'r.',markersize=8)
        else:
            plt.plot(centers[rii+1,0]*sf[0]-boxlen[0]/2.0,(boxlen[-1]/2.0-centers[rii+1,1]*sf[1]), 'b.',markersize=8)
        # plt.show()
    plt.savefig(edir+"_rings.png")

    axnext = fig.add_axes([0.81, 0.5, 0.1, 0.075])
    bnext = Button(axnext, 'Next')
    # Check user clicks
    callback = opalfox.Pause(bnext)
    # create button and events
    callback.cid = plt.connect('button_press_event', callback)

    while not callback.goNext:
        plt.pause(0.1)
#%%############################################################################
# Section 1.4: Analyze Data Image - Find Ring Asymmetry
###############################################################################
# Offset Centers
new_centers = centers - centers[0,:]
# Fit ellipse
ix,iy = opalfox.ellipse(new_centers[0,:],axes[0,:],np.radians(angles[0]),theta=np.linspace(0, 2*np.pi, 10000))
ox,oy = opalfox.ellipse(new_centers[-1,:],axes[-1,:],np.radians(angles[-1]),theta=np.linspace(0, 2*np.pi, 10000))
# Find angles
itheta = np.arctan2(-iy,ix) % (2*np.pi)
otheta = np.arctan2(-oy,ox) % (2*np.pi)
# Find radial distances
ri = np.sqrt(ix**2+iy**2)
ro = np.sqrt(ox**2+oy**2)
# Find difference in radial distance
diffr = np.abs( ri[np.argsort(itheta)] - ro[np.argsort(otheta)] )
# Sort Angles
theta_sort = np.sort(itheta)
otheta_sort = np.sort(otheta)
# METHODS TO DETERMINE OFF AXIS VALUE
match offaxis_mthd:
    # METHOD 1: Find ratio of max/min radial distances from inner to outer circle
    case 'maxminratio':
        # Find ratio based on max radial distance
        maxminratio = np.max(diffr)/np.min(diffr)
        ang_offaxis = theta_sort[np.argmax(diffr)]*180/np.pi
        print('MAX/MIN RADIAL DISTANCE RATIO: ' + str(maxminratio))
        print('OFF ANGLE ASYMMETRY: ' + str(ang_offaxis)+'°')
        data_offaxis = maxminratio
        lab_offaxis  = 'Max/Min Ratio'
    # METHOD 2: Find ratio of max radial distance from inner to outer circle &
    #           radial distance 180 deg from max distance
    case 'maxratio':
        # Find rad distance 180 deg from max rad distance
        deg180 = (theta_sort[np.argmax(diffr)]+np.pi) % (2*np.pi)
        maxratio = np.max(diffr)/diffr[np.argmin(np.abs(deg180 - theta_sort))]
        ang_offaxis = theta_sort[np.argmax(diffr)]*180/np.pi
        print('MAX/MIN RADIAL DISTANCE RATIO: ' + str(maxratio))
        print('OFF ANGLE ASYMMETRY: ' + str(ang_offaxis)+'°')
        data_offaxis = maxratio
        lab_offaxis  = 'Max Ratio'
    # METHOD 3: Find ratio of inner/outer center distances with radial distance
    #           of outer circle at same angle
    case 'centerratio':
        # Find ratio of center difference and mean radial distance
        cntrratio = np.sqrt(np.sum(new_centers[3,:]**2))/np.mean(ro)
        ang_offaxis = np.arctan2(-new_centers[3,1],new_centers[3,0]) % (2*np.pi)*180/np.pi
        print('CENTER DISTANCE DIFFERENCE: ' + str(cntrratio) + '\'')
        print('CENTER ANGLE ASYMMETRY: ' + str(ang_offaxis)+'°')
        data_offaxis = cntrratio
        lab_offaxis  = 'Center Ratio'
    # METHOD 4: Find ratio of inner/outer center distances with radial distance
    #           of outer circle at same angle
    #case 'centerratio':
    case _  :
        raise ValueError("Error: Unexpected method")
#########################################################################
# Create plot
new_centers = centers - centers[0,:]
rot_centers = np.copy(new_centers)
ang_rotation = -1*(np.arctan2(new_centers[-1,1],new_centers[-1,0]) % (2*np.pi))
# New centers
rot_centers[:,0] = new_centers[:,0] * np.cos(ang_rotation) - new_centers[:,1] * np.sin(ang_rotation)
rot_centers[:,1] = new_centers[:,0] * np.sin(ang_rotation) + new_centers[:,1] * np.cos(ang_rotation)
# Create a color map
color_map = plt.colormaps.get_cmap('gist_rainbow')
if plt_flg:
    # Create rings plot
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))

    axnext = fig.add_axes([0.45, 0, 0.1, 0.075])
    bnext = Button(axnext, 'Finish')
    # Check user clicks
    callback = opalfox.Pause(bnext)
    # create button and events
    callback.cid = plt.connect('button_press_event', callback)

    ax[0].set_title('Original Rings')
    ax[0].set_xlabel("X (cm)")
    ax[0].set_ylabel("Y (cm)")
    ax[0].set_aspect('equal')
    ax[0].invert_yaxis()
    ax[0].grid(color='k',alpha=0.75)
    ax[1].set_title('Rotated Rings')
    ax[1].set_xlabel("X (cm)")
    ax[1].set_ylabel("Y (cm)")
    ax[1].set_aspect('equal')
    ax[1].invert_yaxis()
    ax[1].grid(color='k',alpha=0.75)
    for ri in range(rings):
        # Get inner and outer indexs
        i = int(ri*2)
        o = int(ri*2+1)
        # Calculate the color index based on the iteration number
        cindexi = i / (rings*2 - 1)  # Normalize to range [0, 1]
        cindexo = o / (rings*2 - 1)  # Normalize to range [0, 1]
        cindex  = ((cindexi+cindexo)/2.0)  # Normalize to range [0, 1]
        # Compute original ellipse rings
        xi1,yi1 = opalfox.ellipse(new_centers[i,:],axes[i,:],np.radians(angles[i]))
        xo1,yo1 = opalfox.ellipse(new_centers[o,:],axes[o,:],np.radians(angles[o]))
        x1_boundary = np.concatenate([xi1, xo1[::-1]])
        y1_boundary = np.concatenate([yi1, yo1[::-1]])
        ax[0].fill(x1_boundary*sf[0], y1_boundary*sf[1], alpha=.5, linewidth=0.0,color=color_map(cindex))
        ax[0].plot(new_centers[i,0]*sf[0],new_centers[i,1]*sf[1],'.',markersize=20,color=color_map(cindexi))
        ax[0].plot(new_centers[o,0]*sf[0],new_centers[o,1]*sf[1],'.',markersize=20,color=color_map(cindexo))
        ax[0].plot(xi1*sf[0],yi1*sf[1],'-',linewidth=4,color=color_map(cindexi))
        ax[0].plot(xo1*sf[0],yo1*sf[1],'-',linewidth=4,color=color_map(cindexo))
        # Compute rotated ellipse rings
        xi2,yi2 = opalfox.ellipse(rot_centers[i,:],axes[i,:],np.radians(angles[i])+ang_rotation)
        xo2,yo2 = opalfox.ellipse(rot_centers[o,:],axes[o,:],np.radians(angles[o])+ang_rotation)
        x2_boundary = np.concatenate([xi2, xo2[::-1]])
        y2_boundary = np.concatenate([yi2, yo2[::-1]])
        ax[1].fill(x2_boundary*sf[0], y2_boundary*sf[1], alpha=.5, linewidth=0.0,color=color_map(cindex)) 
        ax[1].plot(rot_centers[i,0]*sf[0],rot_centers[i,1]*sf[1],'.',markersize=20,color=color_map(cindexi))
        ax[1].plot(rot_centers[o,0]*sf[0],rot_centers[o,1]*sf[1],'.',markersize=20,color=color_map(cindexo))
        ax[1].plot(xi2*sf[0],yi2*sf[1],'-',linewidth=4,color=color_map(cindexi))
        ax[1].plot(xo2*sf[0],yo2*sf[1],'-',linewidth=4,color=color_map(cindexo))
        plt.savefig(edir+".png")

    while not callback.goNext:
        plt.pause(0.1)

    plt.close('all')
###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
#%%############################################################################
# Section 2.0: Analyze Simulation Image
###############################################################################
# Create array to house off axis value (center diff, ratio, etc.)
if False:
    sim_offaxis = np.empty(len(sim_angles))
    # Iterate over simulation angles
    for ai in range(0,len(sim_angles)):
        angle = "{:.1f}".format(sim_angles[ai])
        print('----------------------------------------------')
        print('SIM ANGLE: '+angle)
        # Simulation files
        sfile           = "angle_"+str(angle)                  # Original file name
        sdirf           = os.path.join(basedir,'images',team,'Sims',"P"+str(spos)+"_X"+str(xmod))
        sdirfile        = os.path.join(sdirf,sfile+filetype)   # Sim File Directory 
        sefile          = os.path.join("Image_Processing","FOXSI4_"+team+"_ellipse_P"+str(spos)+"_X"+str(xmod)+"_A"+str(angle))# Ellipse File
        snewfile        = 'FOXSI_'+sfile+'_projected'            # New File Name
        sdirnfile       = os.path.join(sdirf,"Image_Processing",snewfile+filetype) # New File Directory
        snfile_flg      = os.path.exists(sdirnfile)              # Check if newfile exists
        # -------------------------------------------------------------------------
        # Fix Image Projection
        if proj_flg or not snfile_flg:
            # Read Image File
            simage_original = opalfox.readimage(sdirfile)
            simage_proj = opalfox.fixperspective(simage_original, savedir=sdirnfile)
        else:
            simage_proj = opalfox.readimage(sdirnfile)
        plt.close('all')
        # -------------------------------------------------------------------------
        # Fix Image Brightness
        # Find Scaling Factor (Plot is 10 cm in length)
        ssz = np.shape(simage_proj)
        sboxlen = np.array([10.0])
        ssf = np.array([sboxlen[0]/ssz[0], sboxlen[-1]/ssz[1]])
        # Crop grey image with a min brightness set
        sgimage = opalfox.fixbrightness(simage_proj,minbrightness=1,plot=0,crop=0)
        # ------------------------------------------------------------------------- 
        # Bin Image Brightness
        # Find median brightness in each bin by r and theta (polar coordinates)
        srbin = np.linspace(0, np.sqrt(2), num=501) # Radial bins
        stbin = np.linspace(0, 2*np.pi, num=17)     # Theta bins
        simage_med = opalfox.binimage(sgimage,rbin=srbin,tbin=stbin,plot=0)
        # ------------------------------------------------------------------------- 
        # Detection of Ring Edges
        # Initialize parameters
        scenters  = 0
        saxes     = 0
        sangles   = 0
        rings     = 2
        sdata_matrix = simage_proj
        sedir = os.path.join(sdirf,sefile)
        if ellp_flg or not os.path.exists(sedir+".npz"):
            # Find median brightness in each bin by r and theta (polar coordinates)
            scenters,saxes,sangles = opalfox.fitrings(sdata_matrix,image_bin=simage_med,
                                                    rings=rings,auto=0,rbin=srbin,tbin=stbin,plot=0,
                                                    minbright=[0,0])
            # Save ellipse files
            np.savez(sedir,centers=scenters,axes=saxes,angles=sangles)
        else:    
            sellipse = np.load(sedir+".npz")
            scenters = sellipse['centers']
            saxes    = sellipse['axes']
            sangles  = sellipse['angles']
        if plt_flg:
            # Create image with superimposed rings
            fig, ax = plt.subplots(figsize=(24,16))
            # Plot Image
            plt.imshow(sdata_matrix,extent=[0,10,10,0])
            ax.set_title("FOXSI4 "+team+" Rings: P"+str(spos)+", ANG"+angle)
            ax.axis("on")
            plt.xlabel('X (cm)')
            plt.ylabel('Y (cm)')
            # Iterate each ring
            for rii in range(0,rings*2,2):
                # Find ellipse equation
                sxi,syi = opalfox.ellipse(scenters[rii,:],saxes[rii,:],np.radians(sangles[rii]))
                sxo,syo = opalfox.ellipse(scenters[rii+1,:],saxes[rii+1,:],np.radians(sangles[rii+1]))
                # Plot ellipse
                plt.plot(sxi*ssf[0],syi*ssf[1], color='red', label='Fitted Ellipse')
                plt.plot(sxo*ssf[0],syo*ssf[1], color='red', label='Fitted Ellipse')
                plt.show()
            plt.savefig(sedir+"_rings.png")
        # -----------------------------------------------------------------------------
        # Find asymmetry by max radial distance and center differences
        # Offset Centers
        snew_centers = scenters - scenters[0,:]
        # Fit ellipse
        six,siy = opalfox.ellipse(snew_centers[0,:],saxes[0,:],np.radians(sangles[0]),theta=np.linspace(0, 2*np.pi, 10000))
        sox,soy = opalfox.ellipse(snew_centers[-1,:],saxes[-1,:],np.radians(sangles[-1]),theta=np.linspace(0, 2*np.pi, 10000))
        # Find angles
        sitheta = np.arctan2(-siy,six) % (2*np.pi)
        sotheta = np.arctan2(-soy,sox) % (2*np.pi)
        # Find radial distances
        sri = np.sqrt(six**2+siy**2)
        sro = np.sqrt(sox**2+soy**2)
        # Find difference in radial distance
        sdiffr = np.abs( sri[np.argsort(sitheta)] - sro[np.argsort(sotheta)] )
        # Sort Angles
        stheta_sort = np.sort(sitheta)
        sotheta_sort = np.sort(sotheta)
        # METHODS TO DETERMINE OFF AXIS VALUE
        match offaxis_mthd:
            # METHOD 1: Find ratio of max/min radial distances from inner to outer circle
            case 'maxminratio':
                # Find ratio based on max radial distance
                smaxminratio = np.max(sdiffr)/np.min(sdiffr)
                sangle_diff = stheta_sort[np.argmax(sdiffr)]*180/np.pi
                print('MAX/MIN RADIAL DISTANCE RATIO: ' + str(smaxminratio))
                print('OFF ANGLE ASYMMETRY: ' + str(sangle_diff))
                sim_offaxis[ai] = smaxminratio
            # METHOD 2: Find ratio of max radial distance from inner to outer circle &
            #           radial distance 180 deg from max distance
            case 'maxratio':
                # Find rad distance 180 deg from max rad distance
                sdeg180 = (stheta_sort[np.argmax(sdiffr)]+np.pi) % (2*np.pi)
                smaxratio = np.max(sdiffr)/sdiffr[np.argmin(np.abs(sdeg180 - stheta_sort))]
                sangle_diff = stheta_sort[np.argmax(sdiffr)]*180/np.pi
                print('MAX/MIN RADIAL DISTANCE RATIO: ' + str(smaxratio))
                print('OFF ANGLE ASYMMETRY: ' + str(sangle_diff))
                sim_offaxis[ai] = smaxratio
            # METHOD 3: Find ratio of inner/outer center distances with radial distance
            #           of outer circle at same angle
            case 'centerratio':
                # Find ratio of center difference and mean radial distance
                scntrratio = np.sqrt(np.sum(snew_centers[3,:]**2))/np.mean(sro)
                sctheta = np.arctan2(-snew_centers[3,1],snew_centers[3,0]) % (2*np.pi)*180/np.pi
                print('CENTER DISTANCE DIFFERENCE: ' + str(scntrratio))
                print('CENTER ANGLE ASYMMETRY: ' + str(sctheta))
                sim_offaxis[ai] = scntrratio
            # METHOD 4: Find ratio of inner/outer center distances with radial distance
            #           of outer circle at same angle
            #case 'centerratio':
            case _  :
                raise ValueError("Error: Unexpected method")
        # -------------------------------------------------------------------------
        if plt_flg:
            # Compare Data and Sim
            cmap='viridis'
            fig, ax = plt.subplots(figsize=(24,16))
            # Set ranges in cm
            yrange = np.array([-boxlen[0]/2.0,boxlen[-1]/2.0])
            xrange = np.array([-boxlen[0]/2.0,boxlen[-1]/2.0])
            syrange = np.array([-sboxlen[0]/2.0,sboxlen[-1]/2.0])
            sxrange = np.array([-sboxlen[0]/2.0,sboxlen[-1]/2.0])
            # Scale inner ellipse to data
            scsim = 1#np.max(saxes[3,:])/np.max(axes[3,:])
            # Find new center from data
            x_off = -(boxlen[0]/2.0)+centers[0,0]*sf[0]
            y_off =  (boxlen[-1]/2.0)-centers[0,1]*sf[1] # Reverse y direction due to reverse yrange
            # Find new center from sim
            sx_off = -(sboxlen[0]/2.0)+scenters[0,0]*ssf[0]
            sy_off =  (sboxlen[0]/2.0)-scenters[0,1]*ssf[1]
            plt.imshow(image_proj,cmap=cmap,extent=[xrange[0]-x_off,xrange[1]-x_off,
                                                    yrange[0]-y_off,yrange[1]-y_off])
            plt.plot(0,0,'r.',markersize=30)
            rng = np.array([-5,5])/bsf
            plt.xlim(rng)
            plt.ylim(rng)
            # Iterate each ring
            for rii in range(0,rings*2,2):
                # Find ellipse equation
                xi,yi = opalfox.ellipse(new_centers[rii,:]*sf,
                                        axes[rii,:]*sf ,np.radians(angles[rii]))
                xo,yo = opalfox.ellipse(new_centers[rii+1,:]*sf,
                                        axes[rii+1,:]*sf,np.radians(angles[rii+1]))
                # Plot ellipse
                plt.plot(xi,-yi, color='red', label='Fitted Ellipse')
                plt.plot(xo,-yo, color='red', label='Fitted Ellipse')
                plt.show()
            # Plot Sim
            sgimage[np.where(sgimage==255)] = np.nan
            im = ax.imshow(sgimage,
                        #origin='lower',
                        extent=[(sxrange[0]-sx_off)*scsim,(sxrange[1]-sx_off)*scsim,
                                (syrange[0]-sy_off)*scsim,(syrange[1]-sy_off)*scsim])
            
            
            # Gridlines based on minor ticks
            ax.grid(which='major', color='w', linestyle='-', linewidth=.1)
            
            # Rotate sim image
            trans_data = mtransforms.Affine2D().rotate_deg(ang_offaxis+ang_off) + ax.transData
            im.set_transform(trans_data)
            # Display intended extent of the image
            x1, x2, y1, y2 = im.get_extent()
            ax.plot([x1, x2, x2, x1, x1], [y1, y1, y2, y2, y1], "y--",
                    transform=trans_data)
            plt.xlabel('X (cm)')
            plt.ylabel('Y (cm)')
            plt.title('FOXSI 4 DATA/SIM- '+team+': POS'+str(pos)+", ANG"+angle)
            plt.show()
            plt.plot(new_centers[3,0]*sf[0],-new_centers[3,1]*sf[1],'b.',markersize=30)
            plt.savefig(edir+"_ang_"+angle+"_match.png")
            plt.close('all')
    #%%############################################################################
    # Section 3: Match Data to Simulation
    ###############################################################################
    ###############################################################################
    # Interpolate data
    offangle = np.interp(data_offaxis,sim_offaxis,sim_angles)
    # Plot
    plt.figure(figsize=(9,6))
    plt.plot(sim_angles,sim_offaxis,'r.-',markersize=15)
    plt.plot([-1,offangle],[data_offaxis,data_offaxis],'b--')
    plt.plot([offangle,offangle],[data_offaxis,sim_offaxis[0]*.98],'b--')
    plt.plot(offangle,data_offaxis,'b.',markersize=20)
    plt.xlim([sim_angles[0]*.98,sim_angles[-1]*1.02])
    plt.ylim([sim_offaxis[0]*.98,sim_offaxis[-1]*1.02])
    #plt.xlim([sim_angles[0]*.98,1.9])
    #plt.ylim([sim_offaxis[0]*.98,.105])
    #plt.xlim(10.5,13.0)
    #plt.ylim(.3,.4)
    plt.grid()
    plt.xlabel('Off Angle (arcmin)')
    plt.ylabel('Asymmetry Value')
    plt.title("Off Axis ("+lab_offaxis+" - ASYM ANG {:.0f}°".format(ang_offaxis)+"): P"+str(pos)+ " X"+str(xmod)+ " - {:.3f}\'".format(offangle))
    plt.show()
    plt.savefig(os.path.join(fdir,"FOXSI4_"+team+"_"+file+"_offaxis-"+offaxis_mthd+".png"))
