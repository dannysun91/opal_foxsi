"""
OPAL FOX: OPtical ALignment for FOXSI

Library for Determining Optical Alignment for FOXSI Mission
                      Orlando Romeo
                         07/20/23
          ---------------------------------------
Functions:
    -readimage: 
        Reads image files and only considers RGB values of image
    -plotimage:
        Plots image (increases brightness using 'brighten' keyword)
    -selectpoints:
        Allows user to select n points on image (opens and closes plot for selection)
    -fixperspective:
        Fixes the perspective of the image to ensure the grid is square with the observation plane based on user input
"""
############################################################
#    XX                    /\                   ,'|
#  XX  XX ------------ o--'O `.                /  /
#    XX                 `--.   `-----------._,' ,'
#                           \              ,---'
#                            ) )    _,--(  |
#                           /,^.---'     )/\\
#                          ((   \\      ((  \\
#                           \)   \)      \) (/   
############################################################
# Import Third-party libraries
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy.stats import binned_statistic_2d
from scipy.signal import find_peaks
from scipy.ndimage import median_filter, gaussian_filter
from matplotlib.widgets import Button, CheckButtons
import time
plt.rcParams["font.family"] = "Times New Roman" 
plt.rcParams["font.size"]   = 18
############################################################
# Reads image files and only considers RGB values of image
def readimage(file):
    image = cv2.imread(file)
    if image.any()==None:
        raise ValueError("No File Found at {file}")
    image = image[:,:,:3] # Remove alpha values (Only need RGB Values)
    return image
############################################################
# Change Contrast/Brightness of Image
def alterimage(image,contrast=1,brighten=0.0):
    #mean_brightness = np.mean(image)
    #bf              = brightness/mean_brightness
    image2           = cv2.convertScaleAbs(image, contrast, brighten)
    return image2

############################################################
# Plots image (increases brightness using 'brighten' keyword)
def plotimage(image,brighten=0,title="FOXSI Optical Alignment",cmap='viridis'):
    fig, ax = plt.subplots(figsize=(24,16),num=title) # num is window title
    # Change brightness of image
    if brighten != 0:
        image = alterimage(image,contrast=1,brighten=brighten)
    # Plot Image
    plt.imshow(image,cmap=cmap)
    ax.set_title(title)
    ax.axis('off')
    return fig, ax
############################################################
# Allows user to select n points on image (opens and closes plot for selection)
def selectpoints(image, n=4, brighten=0,zflag=False,norm=0,ax=0,cmap='viridis',title="FOXSI: Select {:.0f} Points on Image",overplot_type=None):
    title=title.format(n)
    # Show Image (and set up scatter points)
    if ax == 0:
        fig, ax = plotimage(image,brighten=brighten,title=title,cmap=cmap)
        fig.canvas.manager.full_screen_toggle()
    scatter = ax.scatter([], [], marker='.', color='red')
    if overplot_type!=None:
        overplot, = ax.plot([],[], color='red',alpha=0.3)
    # Plot Lines at different angles
    sz = np.shape(image)
    lw=0.5
    plt.plot([0,sz[1]],[sz[0]/2,sz[0]/2],'w-',linewidth=lw)
    plt.plot([0,sz[1]],[sz[0],0],'w-',linewidth=lw)
    plt.plot([sz[1]/2,sz[1]/2],[0,sz[0]],'w-',linewidth=lw)
    plt.plot([0,sz[1]],[0,sz[0]],'w-',linewidth=lw)

    # create buttons
    axdone = fig.add_axes([0.81, 0.05, 0.1, 0.075])
    bdone = Button(axdone, 'Done')
    axundo = fig.add_axes([0.09, 0.05, 0.1, 0.075])
    bundo = Button(axundo, 'Undo')
    axreset = fig.add_axes([0.45, 0.05, 0.1, 0.075])
    breset = Button(axreset, 'Reset')
    # if overplot_type != None:
        # axtoggle = fig.add_axes([0.0, 0.9, 0.01, 0.075])
        # btoggle = CheckButtons(axundo, 'Toggle')
    plt.draw()


    # Set up Callback class
    class SelectPointsCallback(object):
        def __init__(self,image_data):
            self.image_data = image_data
            self.points = []    # x,y coordinates
            self.z      = []    # z data
            self.cid    = None  # CID
            self.goNext = False # continue to next plot
            self.toggle = True
        ###
        # BUTTONS
        ###
        def done(self, event): # button to move on to next process
            if len(self.points) == n: # condition met
                self.goNext = True
                self.disconnect()
            else: # not enough points selected
                plt.text(4,0,"Not enough points picked")
                plt.draw()
        def undo(self, event): # erase last added point
            if len(self.points) > 1:
                self.points.pop()
                self.z.pop()
                scatter.set_offsets(self.points)
                if overplot_type == "ellipse": # update the overplot
                    if len(self.points) < 5:
                        overplot.set_xdata([])
                        overplot.set_ydata([])
                    else:
                        self.update_overplot()
                plt.draw()
                ax.set_title(f"Select {n - len(self.points)} more points")
            else:
                self.points = []
                self.z      = []
                scatter.set_offsets(np.array([]).reshape(0, 2))
                if overplot_type == "ellipse":
                    overplot.set_xdata([])
                    overplot.set_ydata([])
                plt.draw()
                ax.set_title(f"Select {n - len(self.points)} more points")
        def reset(self, event): # delete all selected points
            self.points = []
            self.z      = []
            scatter.set_offsets(np.array([]).reshape(0, 2))
            if overplot_type == "ellipse": # update overplot
                overplot.set_xdata([])
                overplot.set_ydata([])
            plt.draw()
            ax.set_title(f"Select {n - len(self.points)} more points")
        def toggle(self, event):
            if not self.toggle:
                overplot.set_alpha(0.3)
            else:
                overplot.set_alpha(0)
            self.toggle = not self.toggle

        def __call__(self, event):
            if not bdone.ax.contains(event)[0] and not bundo.ax.contains(event)[0] and not breset.ax.contains(event)[0]: # not click button
                if plt.get_current_fig_manager().toolbar.mode == '': # Ensure not zooming in
                    if event.name == 'button_press_event' and len(self.points) < n: # click and not all points selected
                        if event.xdata is not None and event.ydata is not None: # valid point
                            xi, yi = int(event.xdata), int(event.ydata)
                            if 0 <= xi < self.image_data.shape[1] and 0 <= yi < self.image_data.shape[0]: # within image
                                self.points.append((event.xdata, event.ydata))
                                self.z.append((self.image_data[yi, xi]))  # Include z-data (image values)
                                scatter.set_offsets(self.points)
                                plt.draw()
                                ax.set_title(f"Select {n - len(self.points)} more points")

                                if overplot_type == "ellipse" and len(self.points)>=5: # requires five points
                                    self.update_overplot()
                    elif len(self.points) == n:
                        ax.set_title(f"All points selected, click Done")

        ### HELPER FUNCTIONS ###
        def update_overplot(self): # helper function for updating overplot
            np_points = np.array(self.points)
            inner_x = np_points[:,0]
            inner_y = np_points[:,1]
            inner_pts = (np.vstack((inner_x, inner_y), dtype=np.float32)).T
            centers, axes, angles = cv2.fitEllipse(inner_pts)

            xi,yi = ellipse(centers,axes,  np.radians(angles))
            # Plot ellipse
            overplot.set_xdata(xi)
            overplot.set_ydata(yi)
        # Disconnect user clicks
        def disconnect(self):
            # plt.close()
            plt.disconnect(self.cid)


    # Check user clicks
    callback = SelectPointsCallback(image)
    # create button and events
    callback.cid = plt.connect('button_press_event', callback)
    bdone.on_clicked(callback.done)
    bundo.on_clicked(callback.undo)
    breset.on_clicked(callback.reset)
    
    # Wait for user to select all n points
    while not callback.goNext:
        plt.pause(0.01)
    # Check for returning normalized coordinates
    points = np.array(callback.points)
    if norm:
        sz = np.shape(image)
        points[:,0] = (points[:,0] - (sz[0]/2.0))/(sz[0]/2.0)
        points[:,1] = (points[:,1] - (sz[1]/2.0))/(sz[1]/2.0)
    # Check to also return z values
    if zflag:
        return points,np.array(callback.z)
    else:
        return points
############################################################
# Fixes the perspective of the image to ensure the grid is square with the observation plane based on user input
#   -Can save new image matrix by specifying directory + file in 'savedir' keyword
def fixperspective(image,savedir=0, brighten=20):
    # Select Points from user clicks on image
    print("Click on image to select 4 corner points for fixing perspective.")
    pts = selectpoints(image, n=4, brighten=brighten)

    sz  = np.shape(image)
    # Convert points to a NumPy array
    points = np.array(pts)
    # Find the center of the rectangle
    center = np.mean(points, axis=0)
    # Calculate the angles of each point with respect to the center
    angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
    # Sort the points based on the angles in ascending order (topleft, topright, bottomright, bottom_left)
    sourcePoints = points[np.argsort(angles)]
    #############################################################
    # Draw square from reference point
    square_x = np.min(sourcePoints[2::1,1])-sourcePoints[0,1]
    square_y = square_x
    # Determine target points from square
    targetPoints = np.empty((4,2))
    # Set top left point
    targetPoints[0,:] = sourcePoints[0,:]
    # Set top right point
    targetPoints[1,:] = sourcePoints[0,:] + [square_x,0.0]
    # Set bottom right point
    targetPoints[2,:] = sourcePoints[0,:] + [square_x,square_y]
    # Set bottom left point
    targetPoints[3,:] = sourcePoints[0,:] + [0.0,square_y]
    #############################################################
    # Prepare array in which the image is placed
    array = np.empty((int(sz[1]*np.sqrt(2)), int(sz[0]*np.sqrt(2)), 3), dtype='uint8')
    # Calculate affine matrix and transform image
    M, mask = cv2.findHomography(np.float32(sourcePoints), np.float32(targetPoints))
    dst     = cv2.warpPerspective(image, M,array.shape[:2]) 
    # Crop Image
    bw = 0 # Border Width
    image_final = dst[int(targetPoints[0,1]-bw):int(targetPoints[2,1]+bw),
                      int(targetPoints[0,0]-bw):int(targetPoints[1,0]+bw)]
    
    fig, ax = plotimage(image_final,brighten=1,title="FOXSI: Projected Grid")
    fig.canvas.manager.full_screen_toggle()

    axnext = fig.add_axes([0.45, 0, 0.1, 0.075])
    bnext = Button(axnext, 'Next')
    # Check user clicks
    callback = Pause(bnext)
    # create button and events
    callback.cid = plt.connect('button_press_event', callback)

    while not callback.goNext:
        plt.pause(0.1)

    image_array = np.array(image_final, dtype=np.uint8)
    # Save the image as a PNG file using cv2.imwrite()
    if isinstance(savedir, str):
        # Create directory if not exists
        if not os.path.exists(os.path.dirname(savedir)):
            # If it doesn't exist, create it
            os.makedirs(os.path.dirname(savedir))
        cv2.imwrite(savedir, image_array)
        print('SAVING: '+savedir)
    return image_final
############################################################
# Fixes the brightness of the image to highlight the laser pattern based on user input (limits min brightness and crops image)
def fixbrightness(image,savedir=0,minbrightness=-1,plot=0,crop=1):
    # Bright image(image_bright = np.mean(image_proj, axis=-1))
    image_bright = np.mean(image, axis=-1)
    # Check for min brightness from user
    if minbrightness == -1:
        # Select Points (and z value) from user clicks on image
        print("Click on 3 image pixels to select minimum brightness!")
        xy,z = selectpoints(image_bright,zflag=True,n=3,cmap='gray') 
        minbrightness = np.mean(z)
    print("MIN BRIGHTNESS: "+str(minbrightness))
    # Remove smaller brightness pixels
    image_bright[np.where(image_bright < minbrightness)] = 0.0
    # Crop Image
    sz = np.shape(image_bright)
    cntr_index = [sz[0]/2.,sz[1]/2.,]
    y = np.linspace(1,-1, num=sz[0])
    x = np.linspace(-1,1, num=sz[1])
    yy, xx = np.meshgrid(y,x)
    r = np.sqrt(xx**2 + yy**2)
    # Max radial distance with data
    if crop ==1:
        print(np.where(image_bright != 0.0))
        max_r = np.max(r[np.where(image_bright != 0.0)])
        yindex1 = round(cntr_index[0] - max_r*sz[0]/2.)
        yindex2 = round(cntr_index[0] + max_r*sz[0]/2.)
        xindex1 = round(cntr_index[1] - max_r*sz[1]/2.)
        xindex2 = round(cntr_index[1] + max_r*sz[1]/2.)
        # New cropped data
        image_bright = image_bright[yindex1:yindex2,xindex1:xindex2]
    if plot != 0:
        fig, ax = plotimage(image_bright,title="FOXSI: Brightness Filter",cmap="gray")
        fig.canvas.manager.full_screen_toggle()
    return image_bright
############################################################
# Bins the image in r and theta bins by median function
def binimage(image_i,rbin=np.linspace(0, np.sqrt(2), num=501),tbin=np.linspace(0, 2*np.pi, num=17),plot=0):
    image = np.copy(image_i)
    # Find r and theta values
    sz         = np.shape(image)
    cntr_index = [sz[0]/2.,sz[1]/2.,]
    y          = np.linspace(1,-1, num=sz[1])
    x          = np.linspace(-1,1, num=sz[0])
    yy, xx     = np.meshgrid(y,x)
    r          = np.sqrt(xx**2 + yy**2)
    theta      = np.arctan2(yy,xx) % (2*np.pi) # Convert theta to range 0 to 2*pi for histogram
    # Create bins
    rmid = (rbin[:-1] + rbin[1:]) / 2.0
    tmid = (tbin[:-1] + tbin[1:]) / 2.0
    # Bin values by median
    image[np.where(np.isnan(image))] = 0.0
    image_med, _, _,_ = binned_statistic_2d(r.ravel(),theta.ravel(),values=image.ravel(),statistic=np.median, bins=[rbin,tbin])
    image_med[np.where(np.isnan(image_med))] = 0.0
    if plot != 0:
        # Create a grid of r, theta coordinates (Add pi to theta due to how image is plotted)
        tgrid,rgrid = np.meshgrid(tmid+np.pi,rmid)
        # Plot the values in polar coordinates with a polar projection
        fig, ax = plt.subplots()
        ax = plt.subplot(111, projection='polar')
        plt.pcolormesh(tgrid, rgrid, image_med, shading='auto', cmap='plasma')
        # Remove degrees labels
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        # Set equal aspect ratio and adjust plot limits
        ax.set_aspect('equal')
        #plt.colorbar(label='Image Brightness')  # Add a colorbar
        plt.title('FOXSI Binned Image')
        plt.gca().set_theta_zero_location('N')  # Set theta zero at the top (North)
        plt.gca().set_theta_direction(-1)  # Set theta increasing clockwise
        plt.tight_layout()
        plt.show()
    return image_med
############################################################
# Equation of Ellipse
def ellipse(center,axes,angle,theta=np.linspace(0, 2*np.pi, 100)):
    x    = center[0] + axes[0]/2 * np.cos(theta) * np.cos(angle) \
                     - axes[1]/2 * np.sin(theta) * np.sin(angle)
    y    = center[1] + axes[0]/2 * np.cos(theta) * np.sin(angle) \
                     + axes[1]/2 * np.sin(theta) * np.cos(angle)
    return x,y
############################################################
# Bins the image in r and theta bins by median function
def fitrings(image,image_bin=0,rings=2,auto=1,rbin=np.linspace(0, np.sqrt(2), num=501),tbin=np.linspace(0, 2*np.pi, num=17),plot=0,minbright=[0,0]):
    image2 = np.copy(image)
    image_med = np.copy(image_bin)
    # Find shape
    sz         = np.shape(image2)
    cntr_index = [sz[0]/2.,sz[1]/2.,]
    # Check to run automatic detection of ring edges
    if auto == 1:
        print("CODE NOT FINISHED - NEEDS WORK")
        # Create bins
        rmid = (rbin[:-1] + rbin[1:]) / 2.0
        tmid = (tbin[:-1] + tbin[1:]) / 2.0
        # Select inner and outer edge for each ring
        print("Select inner and outer edges for "+str(rings)+" rings in any direction!")
        print("Start with smallest ring and move radially outward in selection!")
        ring_edges = selectpoints(image2,n=int(rings*2),norm=1,cmap="gray")
        # Find radial distances of rings
        r_rings = np.sqrt(ring_edges[:,0]**2 + ring_edges[:,1]**2)
        # Find increment in radial bins
        rinc = rmid[1] - rmid[0]
        # Create inner and outer distance arrays
        inner_r = np.empty((len(tmid),rings))
        outer_r = np.empty((len(tmid),rings))        
        # Iterate for each ring
        for ri in range(rings):
            print(ri)
            rdiff = (r_rings[int(ri*2+1)] - r_rings[int(ri*2)])
            rlen = np.floor(rdiff/rinc).astype(int)
            # Iterate for each theta bin
            for thi in range(len(tmid)):
               # Apply gaussian filter to smooth signal, but retain edges
                result        = gaussian_filter(image_med[:,thi], sigma=10)
                
                # Find all local maxima
                localmaxi     = (find_peaks(result,width=int(rlen/2),distance=int(rlen/2)))[0][0]
                # Find value of image at maximum within theta range
                maxval        = np.max(image_med[localmaxi-int(rlen*.2):localmaxi+int(rlen*.2),thi])
                # Find normalized resulting signal based on max value
                result_thi    = result/np.max(result[localmaxi])*maxval
                # Apply gaussian filter to further smooth signal, but retain values
                result_smth = median_filter(result_thi, size=rlen)
                # Find positive and negative slopes around maxima
                pos_slope = np.gradient(result_smth)
                neg_slope = -1.0*pos_slope
                # Remove small slopes and small values
                pos_slope[np.where((pos_slope < 2) | (result_thi < maxval*0.5))] = 0
                neg_slope[np.where((neg_slope < 2) | (result_thi < maxval*0.5))] = 0
                # Find closest and largest slope to maxima
                pos_pk = find_peaks(pos_slope)[0] - localmaxi
                neg_pk = find_peaks(neg_slope)[0] - localmaxi
                # Assign inner and outer radial distances
                inner_r[thi,ri] = rmid[localmaxi +(pos_pk[np.where(pos_pk < 0)])[-1]]
                outer_r[thi,ri] = rmid[localmaxi +(neg_pk[np.where(neg_pk > 0)])[0]]
                
                # Remove all data points inside the maximum outer_r for the next ring
                image_med[:localmaxi +int(rlen),thi] = 0.0
        # Find cartesian coordinates
        inner_x = inner_r*np.cos(tmid[:,np.newaxis]+np.pi/2.0)
        inner_y = inner_r*np.sin(tmid[:,np.newaxis]+np.pi/2.0) 
        outer_x = outer_r*np.cos(tmid[:,np.newaxis]+np.pi/2.0)
        outer_y = outer_r*np.sin(tmid[:,np.newaxis]+np.pi/2.0)
    # Check if user should select edges for each ring        
    else:
        # Create inner and outer distance arrays
        tnum = 8
        inner_r = np.empty((tnum,rings))
        inner_x = np.empty((tnum,rings))
        inner_y = np.empty((tnum,rings))
        outer_r = np.empty((tnum,rings))
        outer_x = np.empty((tnum,rings))
        outer_y = np.empty((tnum,rings))
        print("Start with smallest ring and move radially outward in selection!")
        # Iterate for each ring and edge
        for ri in range(rings):
            # Change brightness
            image_newbright = np.copy(image2)
            image_newbright[np.where(image_newbright < minbright[ri])] = 0
            
            # Select inner ring
            print("Select "+str(tnum)+" points for inner edge of Ring "+str(ri+1))
            inner_ring = selectpoints(image_newbright,n=tnum,norm=1,title='Select {:.0f} Points for Inner Ring '+str(ri+1), overplot_type="ellipse")#,cmap="gray")
            inner_x[:,ri] = inner_ring[:,0]
            inner_y[:,ri] = inner_ring[:,1]
            inner_r[:,ri] = np.sqrt(inner_x[:,ri]**2 + inner_y[:,ri]**2)
            # Select outer ring
            print("Select "+str(tnum)+" points for outer edge of Ring "+str(ri+1))
            outer_ring = selectpoints(image_newbright,n=tnum,norm=1,title='Select {:.0f} Points for Outer Ring '+str(ri+1), overplot_type="ellipse")#,cmap="gray")
            outer_x[:,ri] = outer_ring[:,0]
            outer_y[:,ri] = outer_ring[:,1]
            outer_r[:,ri] = np.sqrt(outer_ring[:,0]**2 + outer_ring[:,1]**2)
    # Plot ring points
    if plot != 0:
        fig, ax = plotimage(image2,cmap='gray')
        ax.axis("on")
        scatter = ax.scatter(cntr_index[0]+outer_x*cntr_index[0],cntr_index[1]+outer_y*cntr_index[1], marker='o', color='red')
        scatter = ax.scatter(cntr_index[0]+inner_x*cntr_index[0],cntr_index[1]+inner_y*cntr_index[1], marker='o', color='red')
        plt.show()
    # Fit ellipse to each ring edge
    centers = np.empty((int(rings*2),2))
    axes    = np.empty((int(rings*2),2))
    angles  = np.empty(int(rings*2))
    # Iterate each ring
    for rii in range(rings):
        # Fit the ellipse to the points
        inner_pts = (np.vstack((inner_x[:,rii]*cntr_index[0]+cntr_index[0],
                                inner_y[:,rii]*cntr_index[1]+cntr_index[1]), dtype=np.float32)).T
        outer_pts = (np.vstack((outer_x[:,rii]*cntr_index[0]+cntr_index[0],
                                outer_y[:,rii]*cntr_index[1]+cntr_index[1]), dtype=np.float32)).T
        inner_elli = cv2.fitEllipse(inner_pts)
        outer_elli = cv2.fitEllipse(outer_pts)
        # Determine indices for inner and outer edge for given ring
        i = int(rii*2)
        o = int(rii*2+1)
        # Get the ellipse parameters
        centers[i,:], axes[i,:], angles[i] = inner_elli
        centers[o,:], axes[o,:], angles[o] = outer_elli
        # Plot Inner and Outer Rings
        if plot != 0:
            # Find ellipse equation
            xi,yi = ellipse(centers[i,:],axes[i,:],np.radians(angles[i]))
            xo,yo = ellipse(centers[o,:],axes[o,:],np.radians(angles[o]))
            # Plot ellipse
            plt.plot(xi,yi, color='blue', label='Fitted Ellipse')
            plt.plot(xo,yo, color='blue', label='Fitted Ellipse')
            plt.show()  
    return centers,axes,angles


class Pause(object):
    def __init__(self,button):
        self.cid = None  # CID
        self.goNext = False
        self.button = button

    def __call__(self, event):
        if self.button.ax.contains(event)[0]: # not click button
            self.goNext = True
            # plt.close()
            plt.disconnect(self.cid)