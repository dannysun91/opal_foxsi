#+
# PROCEDURE:
#   FOXSI X-RAY OPTICS Alignment Script
# PURPOSE:
#   Align Optics with LoS of FOXSI Payload
# NOTES:
#   See below for a list of inputs directly in script
#
# CREATED BY: Orlando Romeo, 02/01/2024
#-
####################################################################################################################################
####################################################################################################################################
#                                                   ,    ###########################################################################
#      XX                    /\                   ,'|    ###########################################################################
#    XX  XX ------------ o--'O `.                /  /    ###########################################################################
#      XX                 `--.   `-----------._,' ,'     ###########################################################################
#                             \              ,---'       ###########################################################################
#                              ) )    _,--(  |           ###########################################################################
#                             /,^.---'     )/\\          ###########################################################################
#                            ((   \\      ((  \\         ###########################################################################
#                             \)   \)      \) (/         ###########################################################################
####################################################################################################################################
####################################################################################################################################
# Import Third-party libraries
import os
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
import math

# def opalfox_addline(xmin,xmax,ymin,ymax,pres=300):

#     # Number of points
#     pnum = int(np.ceil(pres*np.sqrt((xmax-xmin)**2.0+(ymax-ymin)**2.0)))
#     # Create x and y arrays
#     x = np.linspace(xmin, xmax, pnum)
#     y = np.linspace(ymin, ymax, pnum)
#     return np.array([x,y])

def stars_and_bars(n, k):
    """
    Generate all compositions of n indistinguishable items into k bins (stars and bars).
    Each result is a list of k integers summing to n. Thanks ChatGPT its 334 AM
    """
    for bars in combinations(range(n + k - 1), k - 1):
        result = []
        last = -1
        for b in bars:
            result.append(b - last - 1)
            last = b
        result.append(n + k - 2 - last)
        yield np.array(result)

def opalfox_optic(rot_ang=0.0,xoff=0.0,yoff=0.0,plot_optic=None):
    # Center of OPTIC Base
    cntr = np.array([0.0,0.0])
    # Screw Positions for shimming
    sc1 = np.array([0.0,2.8])
    sc2 = np.array([np.cos(30.0*np.pi/180.0),np.sin(30.0*np.pi/180.0)])*2.8
    sc3 = np.array([np.cos(-30.0*np.pi/180.0),np.sin(-30.0*np.pi/180.0)])*2.8
    sc4 = np.array([0.0,-2.8])
    sc5 = np.array([np.cos(-150.0*np.pi/180.0),np.sin(-150.0*np.pi/180.0)])*2.8
    sc6 = np.array([np.cos(150.0*np.pi/180.0),np.sin(150.0*np.pi/180.0)])*2.8
    ########################################################################################################
    # OPTICS BASE SHAPE
    # Each Side of structure, starting from bottom
    #seg1 = [opalfox_addline(-.45,.45,-2.95,-2.95),opalfox_addline(.45,2.3333,-2.95,-1.65)]
    #seg2 = [opalfox_addline(2.3333,2.725,-1.65,-1.025),opalfox_addline(2.725,2.725,-1.025,1.025)]
    #seg3 = [opalfox_addline(2.725,2.3333,1.025,1.65),opalfox_addline(2.3333,.45,1.65,2.95)]
    #seg4 = [opalfox_addline(.45,-.45,2.95,2.95),opalfox_addline(-.45,-2.3333,2.95,1.65)]
    #seg5 = [opalfox_addline(-2.3333,-2.725,1.65,1.025),opalfox_addline(-2.725,-2.725,1.025,-1.025)]
    #seg6 = [opalfox_addline(-2.725,-2.3333,-1.025,-1.65),opalfox_addline(-2.3333,-.45,-1.65,-2.95)]
    # seg1 = [opalfox_addline(-.4488,.4488,-2.8874,-2.8874),opalfox_addline(.4488,2.27624,-2.8874,-1.83242)]
    # seg2 = [opalfox_addline(2.27624,2.725,-1.83242,-1.055077),opalfox_addline(2.725,2.725,-1.055077,1.055077)]
    # seg3 = [opalfox_addline(2.725,2.27624,1.055077,1.83242),opalfox_addline(2.27624,.4488,1.83242,2.8874)]
    # seg4 = [opalfox_addline(.4488,-.4488,2.8874,2.8874),opalfox_addline(-.4488,-2.27624,2.8874,1.83242)]
    # seg5 = [opalfox_addline(-2.27624,-2.725,1.83242,1.055077),opalfox_addline(-2.725,-2.725,1.055077,-1.055077)]
    # seg6 = [opalfox_addline(-2.725,-2.27624,-1.055077,-1.83242),opalfox_addline(-2.27624,-.4488,-1.83242,-2.8874)]
    
    # Create OPTICS Structure (subtracted from center)
    # opt_pts      = [seg1,seg2,seg3,seg4,seg5,seg6]
    # for seg in opt_pts:
    #     for i in range(len(seg)):
    #         seg[i] = seg[i] - cntr[:, np.newaxis]
    # Check for rotation for OPTICS Shape
    rang = rot_ang*np.pi/180.0
    # opt_x = opt_pts[:,0]*np.cos(rang) - opt_pts[:,1]*np.sin(rang)
    # opt_y = opt_pts[:,0]*np.sin(rang) + opt_pts[:,1]*np.cos(rang)
    # # Account for misalignment
    # opt = [[opt_x],[opt_y]]
    ########################################################################################################
    # Compute Final Position of OPTICS Screws (From center, rotation and x/y offset)
    sc1 = sc1-cntr
    sc2 = sc2-cntr
    sc3 = sc3-cntr
    sc4 = sc4-cntr
    sc5 = sc5-cntr
    sc6 = sc6-cntr
    pt1 = np.array([sc1[0]*np.cos(rang) - sc1[1]*np.sin(rang),sc1[0]*np.sin(rang) + sc1[1]*np.cos(rang)])# + [xoff,yoff]
    pt2 = np.array([sc2[0]*np.cos(rang) - sc2[1]*np.sin(rang),sc2[0]*np.sin(rang) + sc2[1]*np.cos(rang)])# + [xoff,yoff]
    pt3 = np.array([sc3[0]*np.cos(rang) - sc3[1]*np.sin(rang),sc3[0]*np.sin(rang) + sc3[1]*np.cos(rang)])# + [xoff,yoff]
    pt4 = np.array([sc4[0]*np.cos(rang) - sc4[1]*np.sin(rang),sc4[0]*np.sin(rang) + sc4[1]*np.cos(rang)])# + [xoff,yoff]
    pt5 = np.array([sc5[0]*np.cos(rang) - sc5[1]*np.sin(rang),sc5[0]*np.sin(rang) + sc5[1]*np.cos(rang)])# + [xoff,yoff]
    pt6 = np.array([sc6[0]*np.cos(rang) - sc6[1]*np.sin(rang),sc6[0]*np.sin(rang) + sc6[1]*np.cos(rang)])# + [xoff,yoff]
    ########################################################################################################
    # Plot XRAY OPTICS
    if plot_optic is not None:
        # Symbol Thickness for WIN File
        thck = 1
        # Check to save figure
        if type(plot_optic)==str:
            if 'ps' in plot_optic.lower():
                ps   = plot_optic #  PS File
                thck = 6         # Symbol Thickness for PS File
            else:
                ps = 0

        # Set x/y ranges
        # xrange = [np.min(opt[:,0])-.2,np.max(opt[:,0])+.2]
        # yrange = [np.min(opt[:,1])-.2,np.max(opt[:,1])+.2]
        # # Set plot title
        # xoff_str = 'X'+vplot_char('off',command='sub',ps=ps)+' = '+string(xoff,format='(f+0.2)')+'in'
        # yoff_str = 'Y'+vplot_char('off',command='sub',ps=ps)+' = '+string(yoff,format='(f+0.2)')+'in'
        # title = 'XRAY OPTICS'# ('+xoff_str+', '+yoff_str+')'
        # # Plot Base shape
        # vplot,opt[*,0],opt[*,1],lim={isotropic:1,xrange:xrange,yrange:yrange,$
        # xtitle:'X'+vplot_char('payload',command='sub')+' (in)',ytitle:'Y'+vplot_char('payload',command='sub')+' (in)',title:title},$
        # vps=vps,/grid,save=ps,/delaysave
        # # Plot front aperature
        # vplot_sym,'circle',fill=0
        # vplot,0,0,lim={isotropic:1},vps=vps,/overplot,sysize=5.0,syunit='data',sym=8,save=ps,/delaysave
        # vplot,0,0,lim={isotropic:1},vps=vps,/overplot,sysize=4.0,syunit='data',sym=8,save=ps,/delaysave
        # vplot,0,0,lim={isotropic:1},vps=vps,/overplot,sysize=3.95,syunit='data',sym=8,save=ps,/delaysave
        # vplot,0,0,lim={isotropic:1},vps=vps,/overplot,sysize=3.90,syunit='data',sym=8,save=ps,/delaysave
        # vplot,0,0,lim={isotropic:1},vps=vps,/overplot,sysize=3.85,syunit='data',sym=8,save=ps,/delaysave
        # vplot,0,0,lim={isotropic:1},vps=vps,/overplot,sysize=1.25,syunit='data',sym=8,save=ps,/delaysave
        # vplot_sym,'circle',thick=10
        # vplot,0,0,lim={isotropic:1},vps=vps,/overplot,sysize=2.0,syunit='data',sym=8,save=ps,/delaysave
        # # Plot Alignment Crosshairs
        # vplot_sym,'crosshairs',thick=thck
        # vplot,xoff,yoff,lim={isotropic:1},vps=vps,/overplot,sym=8,sysize=2,shade=254b,save=ps,/delaysave
        # vplot_sym,/default
        # # Screw 1
        # vplot,pt1[0],pt1[1],vps=vps,/over,syunit='data',sym=8,sysize=.15,lim={isotropic:1},shade=43b,save=ps,/delaysave
        # xyouts,pt1[0],pt1[1]-.25,'1',align=0.5
        # # Screw 2
        # vplot,pt2[0],pt2[1],vps=vps,/over,syunit='data',sym=8,sysize=.15,lim={isotropic:1},shade=43b,save=ps,/delaysave
        # xyouts,pt2[0]-.1,pt2[1]-.2,'2',align=0.5
        # # Screw 3
        # vplot,pt3[0],pt3[1],vps=vps,/over,syunit='data',sym=8,sysize=.15,lim={isotropic:1},shade=43b,save=ps,/delaysave
        # xyouts,pt3[0]-.15,pt3[1]+.1,'3',align=0.5
        # # Screw 4
        # vplot,pt4[0],pt4[1],vps=vps,/over,syunit='data',sym=8,sysize=.15,lim={isotropic:1},shade=43b,save=ps,/delaysave
        # xyouts,pt4[0],pt4[1]+.1,'4',align=0.5
        # # Screw 5
        # vplot,pt5[0],pt5[1],vps=vps,/over,syunit='data',sym=8,sysize=.15,lim={isotropic:1},shade=43b,save=ps,/delaysave
        # xyouts,pt5[0]+.15,pt5[1]+.1,'5',align=0.5
        # # Screw 6
        # vplot,pt6[0],pt6[1],vps=vps,/over,syunit='data',sym=8,sysize=.15,lim={isotropic:1},shade=43b,save=ps,/delaysave
        # xyouts,pt6[0]+.1,pt6[1]-.2,'6',align=0.5
        
        # # Save figure
        # vplot,!values.f_nan,!values.f_nan,vps=vps,/over,save=ps
    
    return np.array([pt1,pt2,pt3,pt4,pt5,pt6])
#################################################################################################################


# SECTION 0: USER INPUTS
####################################################################################################################################
# OPTIC MECHANICAL Properties (inches, degrees)
opt_len   = 5.600                  # Radius of optic (from center to screws)
rot_ang   = 30.0                      # Rotation angle to orient optic wrt Rocket (degrees)
# OPTIC ALIGNMENT Properties (degrees)
# Misalignment angle Amount - Usually within 20 arcminutes
mis_ang = 1.898/60.0
# Off Axis Amount (degrees) (from start of unit circle (x=1,y=0), going around CCW (0-360)) - Shim away from shearing
off_ang = 260
# Account for 180 shift from perspective of looking at the payload from the front vs the back
off_ang = (270.0-off_ang)+90.0
# Compute horizontal and vertical offsets in cartesian frame
xoff       = (opt_len*np.sin(mis_ang*np.pi/180.0))*np.cos(off_ang*np.pi/180.0)  # Inches
yoff       = (opt_len*np.sin(mis_ang*np.pi/180.0))*np.sin(off_ang*np.pi/180.0)  # Inches
# Compute horizontal and vertical offsets (degrees)
hoff = np.arcsin(xoff/opt_len)/np.pi*180.0
voff = np.arcsin(yoff/opt_len)/np.pi*180.0
# SHIM Properties (inches)
shim_thick = 0.002                   # Shim thickness (in)
shim_rad   = 0.100                   # Shim Radius
maxsnum    = 5                      # Max Number of Shims
# SCRIPT Flags
plot_optic = None                       # Flag to plot optic
prnt_flg   = 0                       # Flag set to print all results for every shim combo
mthd_flg   = 1                       # Flag set method for computing best shim combo (see below)
####################################################################################################################################
####################################################################################################################################
####################################################################################################################################
####################################################################################################################################
# SECTION 1: SHIM POSITIONS
####################################################################################################################################
# Total Misalignment Angle
print('------------------------------------------------------------------------------------------------------------')
print(rf'0 SHIMS   -   {hoff:.3f}° H_OFF   -  {voff:.3f}° V_OFF   -  {mis_ang:.3f}° MISALIGNMENT   -    METHOD {mthd_flg}')
print('------------------------------------------------------------------------------------------------------------')
# Find screw/shim position
shim_pos = opalfox_optic(rot_ang=rot_ang,xoff=xoff,yoff=yoff,plot_optic=plot_optic)
# Number of Screw/Shim Positions
k = 6
####################################################################################################################################
####################################################################################################################################
# SECTION 2: SHIM COMBINATIONS
####################################################################################################################################
# Iterate over number of shims
for n0 in range(maxsnum):
    n=n0+1
    # Initialize best combo given misalignment error
    best_combo = np.zeros(k,dtype=int)
    best_err   = mis_ang
    best_hoff  = hoff
    best_voff  = voff
    # Output Header Info
    if (prnt_flg):
        print('-------------------------------------------')
        print(rf'|            {n} SHIM RESULTS              |')
        print('|_________________________________________|')
        print('| '+ ' '.join(f'P{i}' for i in range(1, k + 1))+'|    XOFF      YOFF   |  ERROR |')
    
    # Iterate over possible combinations
    for combo in stars_and_bars(n, k):
        shims = combo * shim_thick
        ########################################################################################################
        # METHOD 1: Compute Vector Change from 3 Surrounding Shims to form one plane
        if mthd_flg == 1:
            v = np.array([0,0,0])
            # Iterate over each main screw
            for ki in range(k):
                # Set screws
                s1 = ki
                s2 = (ki+1) % k
                s3 = (ki+2) % k
                # Find 2 Difference Vectors
                v21 = np.concatenate((shim_pos[s2], [shims[s2]])) - np.concatenate((shim_pos[s1], [shims[s1]]))
                v31 = np.concatenate((shim_pos[s3], [shims[s3]])) - np.concatenate((shim_pos[s1], [shims[s1]]))
                # Find cross product - normal to plane of 3 points
                crss = np.cross(v21,v31)
                if crss[2] < 0:
                    crss = -crss  # Ensure Vector points upward
                # Find optic length vector
                lcrss = (crss/np.sqrt(np.sum(crss**2.0)))
                v = v+lcrss
            
            uv = v/np.sqrt(np.sum(v**2))
            # Compute new offsets from Shim Configuration
            new_hoff = 180/np.pi*np.arcsin(uv[0])
            new_voff = 180/np.pi*np.arcsin(uv[1])
            # Compute Error from new offset
            error = np.sqrt( (new_hoff+hoff)**2 + (new_voff+voff)**2.0)
        
        ########################################################################################################
        # Check best combo so far based on error
        if error < best_err:
            best_err   = error
            best_combo = combo
            best_hoff  = new_hoff
            best_voff  = new_voff
        
        # Output results
        if prnt_flg:
            combo_str = '| ' + ' '.join(str(x) for x in combo) + ' |'
            offsets_str = f" {new_hoff:.3f}°   {new_voff:.3f}°   |"
            error_str = f"{error:.3f}° |"

            print(combo_str + offsets_str, error_str)
        ########################################################################################################
        
    ########################################################################################################
    if prnt_flg:
        print('-------------------------------------------')
    # Output best combo given shim amount
    if np.sum(best_combo) == 0:
        best_err = np.sqrt( (xoff)**2 + (yoff)**2)
    
    combo_str = ' '.join(f"{x:3d}" for x in best_combo) + " COMBO"
    print(rf'{n:2d} SHIMS   -   {best_hoff+hoff:0.3f}° H_OFF   -  {best_voff+voff:0.3f}° V_OFF   -  {best_err:0.3f}° MISALIGNMENT   -  {combo_str}')