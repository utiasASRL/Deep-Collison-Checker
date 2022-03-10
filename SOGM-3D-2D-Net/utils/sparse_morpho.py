#
#
#      0==============================0
#      |    Deep Collision Checker    |
#      0==============================0
#
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      Script for sparse point morphology
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      Hugues THOMAS - 11/06/2018
#


# ----------------------------------------------------------------------------------------------------------------------
#
#           Imports and global variables
#       \**********************************/
#


# Basic libs
from os import EX_OSFILE
import numpy as np
from sklearn.neighbors import KDTree
from scipy import ndimage
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider


def sparse_point_opening(pts_2D, positive_mask, negative_tree=None, negative_pts=None, negative_mask=None, d=1.0, erode_d=None, dilate_d=None):

    # Like image binary opening but on sparse point positions in 3D
    # 1. negatives "eat" positives (erosion)
    # 2. Remaining positives "eat back" (dilation)

    if negative_tree is None:

        if negative_pts is not None:
            if negative_mask is not None:
                negative_pts = negative_pts[negative_mask]

        elif negative_mask is not None:
            negative_pts = pts_2D[negative_mask]

        else:
            negative_pts = pts_2D[np.logical_not(positive_mask)]

        negative_tree = KDTree(negative_pts)


    if erode_d is None:
        erode_d = d

    if dilate_d is None:
        dilate_d = d
    
    # Get the dynamic points not in range of static
    rem_pos_mask = np.copy(positive_mask)
    if (np.any(rem_pos_mask)):
        dists, inds = negative_tree.query(pts_2D[positive_mask], 1)
        rem_pos_mask = np.copy(positive_mask)
        rem_pos_mask[positive_mask] = np.squeeze(dists) > erode_d

    # Get the dynamic in range of the remaining dynamic
    opened_mask = np.copy(positive_mask)
    if (np.any(rem_pos_mask)):
        tree2 = KDTree(pts_2D[rem_pos_mask])
        dists, inds = tree2.query(pts_2D[positive_mask], 1)
        opened_mask[positive_mask] = np.squeeze(dists) < dilate_d

    # Return the opened positive mask
    return opened_mask


def sparse_point_closing(pts_2D, positive_mask, negative_mask=None, d=1.0, erode_d=None, dilate_d=None):

    if erode_d is None:
        erode_d = d

    if dilate_d is None:
        dilate_d = d

    # Like image binary opening but on sparse point positions in 3D
    # 1. positives "eat" negatives (dilation)
    # 2. Remaining negatives "eat back" (erosion)

    # Is equivalent to a opening from the negatives to the positives
    if negative_mask is None:
        negative_mask = np.logical_not(positive_mask)

    openened_negatives = sparse_point_opening(pts_2D,
                                              positive_mask=negative_mask,
                                              negative_mask=positive_mask,
                                              erode_d=dilate_d,
                                              dilate_d=erode_d)

    add_mask = np.logical_and(negative_mask, np.logical_not(openened_negatives))
    closed_mask = np.logical_or(positive_mask, add_mask)

    return closed_mask


def create_circular_mask(h, w, center=None, radius=None):

    if center is None:  # use the middle of the image
        center = (int(w/2), int(h/2))
    if radius is None:  # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w-center[0], h-center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

    mask = dist_from_center <= radius
    return mask


def subpts_to_image(pts_2D, sampleDl=0.1, mask=None):

    # If no positive mask, then all points are considered
    if mask is None:
        mask = np.ones_like(pts_2D[:, 0]).astype(bool)

    # Compute pixel indice for each point
    grid_indices = np.floor(pts_2D / sampleDl).astype(int)

    # Limits of the grid
    min_corner = np.amin(grid_indices, axis=0)
    max_corner = np.amax(grid_indices, axis=0)

    # No negative inds
    grid_indices0 = grid_indices[mask] - min_corner

    # Get image
    deltaX, deltaY = max_corner - min_corner + 1
    img_2D = np.zeros((deltaY * deltaX), bool)
    vec_inds = grid_indices0[:, 0] + deltaX * grid_indices0[:, 1]
    img_2D[vec_inds] = True

    return np.reshape(img_2D, (deltaY, deltaX)), grid_indices0


def pepper_noise_removal(pts, sampleDl=0.1, pepper_margin=0.3, positive_mask=None):

    # If no positive mask, then all points are considered
    if positive_mask is None:
        positive_mask = np.ones_like(pts[:, 0]).astype(bool)

    # Make sure we have 2D points and only consider the points in mask
    pts_2D = pts[positive_mask, :2]

    # Convert to 2D image
    img_2D, proj_inds = subpts_to_image(pts_2D, sampleDl)

    # Debug structure
    debug_struct = False
    if debug_struct:
            
        struct_n = np.floor(pepper_margin / sampleDl) + 1
        struct = create_circular_mask(2 * struct_n + 1, 2 * struct_n + 1, radius=pepper_margin / sampleDl)

        figA, axA = plt.subplots(1, 1, figsize=(10, 7))
        plt.subplots_adjust(bottom=0.25)

        # Plot first frame of seq
        plotsA = [axA.imshow(struct)]

        # Make a horizontal slider to control the frequency.
        axcolor = 'lightgoldenrodyellow'
        axtime = plt.axes([0.15, 0.1, 0.75, 0.03], facecolor=axcolor)
        time_slider = Slider(ax=axtime,
                             label='radius',
                             valmin=sampleDl,
                             valmax=2 * pepper_margin,
                             valinit=pepper_margin,
                             valstep=sampleDl/30)

        # The function to be called anytime a slider's value changes
        def update_PR(val):
            struct = create_circular_mask(2 * struct_n + 1, 2 * struct_n + 1, radius=val / sampleDl)
            plotsA[0].set_array(struct)
            return plotsA

        # register the update function with each slider
        time_slider.on_changed(update_PR)

        plt.show()
        a = 1/0

    # Get structures from pepper_margin
    close_struct_n = np.floor(1.6 * pepper_margin / sampleDl)
    close_struct = create_circular_mask(2 * close_struct_n + 1, 2 * close_struct_n + 1, radius=1.6 * pepper_margin / sampleDl)
    open_struct_n = np.floor(pepper_margin / sampleDl)
    open_struct = create_circular_mask(2 * open_struct_n + 1, 2 * open_struct_n + 1, radius=pepper_margin / sampleDl)

    # Perform image closing first to make sure large blobs dont have holes
    img_2D_closed = ndimage.binary_closing(img_2D, structure=close_struct, iterations=1)

    # Then opening to remove pepper noise
    img_2D_opened = ndimage.binary_opening(img_2D_closed, structure=open_struct, iterations=1)

    # Dilate again to get back points in range of valid ones
    img_2D_dilated = ndimage.binary_dilation(img_2D_opened, structure=close_struct, iterations=1)

    # Get the openning mask
    opened_mask = img_2D_dilated[proj_inds[:, 1], proj_inds[:, 0]]
    

    # Get the mask of the valid points
    clean_mask = np.copy(positive_mask)
    clean_mask[positive_mask] = opened_mask

    # if np.any(clean_mask):
    #     plt.figure()
    #     plt.imshow(img_2D)
    #     plt.figure()
    #     plt.imshow(img_2D_closed)
    #     plt.figure()
    #     plt.imshow(img_2D_opened)
    #     plt.figure()
    #     plt.imshow(img_2D_dilated)
    #     plt.figure()
    #     img_2D_valid, _ = subpts_to_image(pts_2D, sampleDl, mask=opened_mask)
    #     plt.imshow(img_2D_valid)
    #     plt.show()

    return clean_mask
