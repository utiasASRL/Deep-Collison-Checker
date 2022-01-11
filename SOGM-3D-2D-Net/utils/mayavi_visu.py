#
#
#      0==============================0
#      |    Deep Collision Checker    |
#      0==============================0
#
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      Script for various visualization with mayavi
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
import torch
import os
import pickle
os.environ.update(OMP_NUM_THREADS='1',
                  OPENBLAS_NUM_THREADS='1',
                  NUMEXPR_NUM_THREADS='1',
                  MKL_NUM_THREADS='1',)
import numpy as np
from sklearn.neighbors import KDTree
from os import makedirs, remove, rename, listdir
from os.path import exists, join
import time
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle, Circle
from matplotlib.widgets import Slider
import imageio
from PIL import Image

import scipy
from scipy import ndimage
import scipy.ndimage.filters as filters

import sys

# PLY reader
from utils.ply import write_ply, read_ply
from utils.metrics import fast_confusion, IoU_from_confusions

# Configuration class
from utils.config import Config



class Box:
    """Box class"""

    def __init__(self, x1, y1, x2, y2):
        """
        Initialize parameters here.
        """
        self.x1 = min(x1, x2)
        self.y1 = min(y1, y2)
        self.x2 = max(x1, x2)
        self.y2 = max(y1, y2)

        return

    def dx(self):
        return self.x2 - self.x1

    def dy(self):
        return self.y2 - self.y1

    def plt_rect(self, edgecolor='black', facecolor='cyan', fill=False, lw=2):
        return Rectangle((self.x1, self.y1), self.dx(), self.dy(),
                         edgecolor=edgecolor,
                         facecolor=facecolor,
                         fill=fill,
                         lw=lw)

    def inside(self, x, y, margin=0):

        if margin == 0:
            return self.x1 < x < self.x2 \
                   and self.y1 < y < self.y2
        else:
            return self.x1 - margin < x < self.x2 + margin \
                   and self.y1 - margin < y < self.y2 + margin

    def min_box_repulsive_vector(self, pos):

        # Check along edges
        edges = self.get_edges()
        corners = self.get_corners()
        found = False
        min_mag = 1e9
        min_normal = np.zeros((2,), dtype=np.float32)
        for edge, corner in zip(edges, corners):

            edge_l = np.linalg.norm(edge)
            edge_dir = edge / edge_l

            # Project position on edge
            pos_vector = pos - corner
            proj = np.dot(pos_vector, edge_dir)
            proj_vec = proj * edge_dir

            if 0 < proj < edge_l:
                normal = pos_vector - proj_vec

                if np.linalg.norm(normal) < min_mag:
                    min_normal = normal
                    min_mag = np.linalg.norm(normal)
                    found = True

        # Check in diagonal from corners
        if not found:
            min_mag = 1e9
            for corner in corners:
                pos_vector = pos - corner
                dist = np.linalg.norm(pos_vector)
                if dist < min_mag:
                    min_mag = dist
                    min_normal = pos_vector

        return min_normal


    def get_corners(self):

        A = np.array([self.x1, self.y1])
        B = np.array([self.x2, self.y1])
        C = np.array([self.x2, self.y2])
        D = np.array([self.x1, self.y2])

        return [A, B, C, D]


    def get_edges(self):

        A, B, C, D = self.get_corners()

        return [B - A, C - B, D - C, A - D]



def show_ModelNet_models(all_points):
    from mayavi import mlab

    ###########################
    # Interactive visualization
    ###########################

    # Create figure for features
    fig1 = mlab.figure('Models', bgcolor=(1, 1, 1), size=(1000, 800))
    fig1.scene.parallel_projection = False

    # Indices
    global file_i
    file_i = 0

    def update_scene():

        #  clear figure
        mlab.clf(fig1)

        # Plot new data feature
        points = all_points[file_i]

        # Rescale points for visu
        points = (points * 1.5 + np.array([1.0, 1.0, 1.0])) * 50.0

        # Show point clouds colorized with activations
        activations = mlab.points3d(points[:, 0],
                                    points[:, 1],
                                    points[:, 2],
                                    points[:, 2],
                                    scale_factor=3.0,
                                    scale_mode='none',
                                    figure=fig1)

        # New title
        mlab.title(str(file_i), color=(0, 0, 0), size=0.3, height=0.01)
        text = '<--- (press g for previous)' + 50 * ' ' + '(press h for next) --->'
        mlab.text(0.01, 0.01, text, color=(0, 0, 0), width=0.98)
        mlab.orientation_axes()

        return

    def keyboard_callback(vtk_obj, event):
        global file_i

        if vtk_obj.GetKeyCode() in ['g', 'G']:

            file_i = (file_i - 1) % len(all_points)
            update_scene()

        elif vtk_obj.GetKeyCode() in ['h', 'H']:

            file_i = (file_i + 1) % len(all_points)
            update_scene()

        return

    # Draw a first plot
    update_scene()
    fig1.scene.interactor.add_observer('KeyPressEvent', keyboard_callback)
    mlab.show()


def show_ModelNet_examples(clouds, cloud_normals=None, cloud_labels=None):
    from mayavi import mlab

    ###########################
    # Interactive visualization
    ###########################

    # Create figure for features
    fig1 = mlab.figure('Models', bgcolor=(1, 1, 1), size=(1000, 800))
    fig1.scene.parallel_projection = False

    if cloud_labels is None:
        cloud_labels = [points[:, 2] for points in clouds]

    # Indices
    global file_i, show_normals
    file_i = 0
    show_normals = True

    def update_scene():

        #  clear figure
        mlab.clf(fig1)

        # Plot new data feature
        points = clouds[file_i]
        labels = cloud_labels[file_i]
        if cloud_normals is not None:
            normals = cloud_normals[file_i]
        else:
            normals = None

        # Rescale points for visu
        points = (points * 1.5 + np.array([1.0, 1.0, 1.0])) * 50.0

        # Show point clouds colorized with activations
        activations = mlab.points3d(points[:, 0],
                                    points[:, 1],
                                    points[:, 2],
                                    scale_factor=2.0,
                                    scale_mode='none',
                                    figure=fig1)
        if normals is not None and show_normals:
            # Dont show all normals or we cant see well
            # random_N = points.shape[0] // 4
            # random_inds = np.random.permutation(points.shape[0])[:random_N]
            random_inds = np.arange(points.shape[0])
            activations = mlab.quiver3d(points[random_inds, 0],
                                        points[random_inds, 1],
                                        points[random_inds, 2],
                                        normals[random_inds, 0],
                                        normals[random_inds, 1],
                                        normals[random_inds, 2],
                                        scale_factor=10.0,
                                        line_width=1.0,
                                        scale_mode='none',
                                        figure=fig1)

        # New title
        mlab.title(str(file_i), color=(0, 0, 0), size=0.3, height=0.01)
        text = '<--- (press g for previous)' + 50 * ' ' + '(press h for next) --->'
        mlab.text(0.01, 0.01, text, color=(0, 0, 0), width=0.98)
        mlab.orientation_axes()

        return

    def keyboard_callback(vtk_obj, event):
        global file_i, show_normals

        if vtk_obj.GetKeyCode() in ['g', 'G']:
            file_i = (file_i - 1) % len(clouds)
            update_scene()

        elif vtk_obj.GetKeyCode() in ['h', 'H']:
            file_i = (file_i + 1) % len(clouds)
            update_scene()

        elif vtk_obj.GetKeyCode() in ['n', 'N']:
            show_normals = not show_normals
            update_scene()

        return

    # Draw a first plot
    update_scene()
    fig1.scene.interactor.add_observer('KeyPressEvent', keyboard_callback)
    mlab.show()


def show_ModelNet_lrf(clouds, clouds_lrf):
    from mayavi import mlab

    ###########################
    # Interactive visualization
    ###########################

    # Create figure for features
    fig1 = mlab.figure('Models', bgcolor=(1, 1, 1), size=(1000, 800))
    fig1.scene.parallel_projection = False

    # Indices
    global file_i, show_normals
    file_i = 0
    show_normals = True

    def update_scene():

        #  clear figure
        mlab.clf(fig1)

        # Plot new data feature
        points = clouds[file_i]
        lrf = clouds_lrf[file_i]

        # Rescale points for visu
        points = (points * 1.5 + np.array([1.0, 1.0, 1.0])) * 50.0

        # Show point clouds colorized with activations
        activations = mlab.points3d(points[:, 0],
                                    points[:, 1],
                                    points[:, 2],
                                    scale_factor=2.0,
                                    scale_mode='none',
                                    figure=fig1)

        if show_normals:

            # Dont show all normals or we cant see well
            for i in range(3):
                color = [0.0, 0.0, 0.0]
                color[i] = 1.0
                activations = mlab.quiver3d(points[::100, 0],
                                            points[::100, 1],
                                            points[::100, 2],
                                            lrf[::100, 0, i],
                                            lrf[::100, 1, i],
                                            lrf[::100, 2, i],
                                            scale_factor=10.0,
                                            line_width=1.0,
                                            scale_mode='none',
                                            color=tuple(color),
                                            figure=fig1)

        # New title
        mlab.title(str(file_i), color=(0, 0, 0), size=0.3, height=0.01)
        text = '<--- (press g for previous)' + 50 * ' ' + '(press h for next) --->'
        mlab.text(0.01, 0.01, text, color=(0, 0, 0), width=0.98)
        mlab.orientation_axes()

        return

    def keyboard_callback(vtk_obj, event):
        global file_i, show_normals

        if vtk_obj.GetKeyCode() in ['g', 'G']:
            file_i = (file_i - 1) % len(clouds)
            update_scene()

        elif vtk_obj.GetKeyCode() in ['h', 'H']:
            file_i = (file_i + 1) % len(clouds)
            update_scene()

        elif vtk_obj.GetKeyCode() in ['n', 'N']:
            show_normals = not show_normals
            update_scene()

        return

    # Draw a first plot
    update_scene()
    fig1.scene.interactor.add_observer('KeyPressEvent', keyboard_callback)
    mlab.show()


def show_neighbors(query, supports, neighbors):
    from mayavi import mlab

    ###########################
    # Interactive visualization
    ###########################

    # Create figure for features
    fig1 = mlab.figure('Models', bgcolor=(1, 1, 1), size=(1000, 800))
    fig1.scene.parallel_projection = False

    # Indices
    global file_i
    file_i = 0

    def update_scene():

        #  clear figure
        mlab.clf(fig1)

        # Rescale points for visu
        p1 = (query * 1.5 + np.array([1.0, 1.0, 1.0])) * 50.0
        p2 = (supports * 1.5 + np.array([1.0, 1.0, 1.0])) * 50.0

        l1 = p1[:, 2]*0
        l1[file_i] = 1

        l2 = p2[:, 2]*0 + 2
        l2[neighbors[file_i]] = 3

        # Show point clouds colorized with activations
        activations = mlab.points3d(p1[:, 0],
                                    p1[:, 1],
                                    p1[:, 2],
                                    l1,
                                    scale_factor=2.0,
                                    scale_mode='none',
                                    vmin=0.0,
                                    vmax=3.0,
                                    figure=fig1)

        activations = mlab.points3d(p2[:, 0],
                                    p2[:, 1],
                                    p2[:, 2],
                                    l2,
                                    scale_factor=3.0,
                                    scale_mode='none',
                                    vmin=0.0,
                                    vmax=3.0,
                                    figure=fig1)

        # New title
        mlab.title(str(file_i), color=(0, 0, 0), size=0.3, height=0.01)
        text = '<--- (press g for previous)' + 50 * ' ' + '(press h for next) --->'
        mlab.text(0.01, 0.01, text, color=(0, 0, 0), width=0.98)
        mlab.orientation_axes()

        return

    def keyboard_callback(vtk_obj, event):
        global file_i

        if vtk_obj.GetKeyCode() in ['g', 'G']:

            file_i = (file_i - 1) % len(query)
            update_scene()

        elif vtk_obj.GetKeyCode() in ['h', 'H']:

            file_i = (file_i + 1) % len(query)
            update_scene()

        return

    # Draw a first plot
    update_scene()
    fig1.scene.interactor.add_observer('KeyPressEvent', keyboard_callback)
    mlab.show()


def show_input_batch(batch):
    from mayavi import mlab

    ###########################
    # Interactive visualization
    ###########################

    # Create figure for features
    fig1 = mlab.figure('Input', bgcolor=(1, 1, 1), size=(1000, 800))
    fig1.scene.parallel_projection = False

    # Unstack batch
    all_points = batch.unstack_points()
    all_neighbors = batch.unstack_neighbors()
    all_pools = batch.unstack_pools()

    # Indices
    global b_i, l_i, neighb_i, show_pools
    b_i = 0
    l_i = 0
    neighb_i = 0
    show_pools = False

    def update_scene():

        #  clear figure
        mlab.clf(fig1)

        # Rescale points for visu
        p = (all_points[l_i][b_i] * 1.5 + np.array([1.0, 1.0, 1.0])) * 50.0
        labels = p[:, 2] * 0

        if show_pools:
            p2 = (all_points[l_i + 1][b_i][neighb_i:neighb_i + 1] * 1.5 + np.array([1.0, 1.0, 1.0])) * 50.0
            p = np.vstack((p, p2))
            labels = np.hstack((labels, np.ones((1,), dtype=np.int32) * 3))
            pool_inds = all_pools[l_i][b_i][neighb_i]
            pool_inds = pool_inds[pool_inds >= 0]
            labels[pool_inds] = 2
        else:
            neighb_inds = all_neighbors[l_i][b_i][neighb_i]
            neighb_inds = neighb_inds[neighb_inds >= 0]
            labels[neighb_inds] = 2
            labels[neighb_i] = 3

        # Show point clouds colorized with activations
        mlab.points3d(p[:, 0],
                      p[:, 1],
                      p[:, 2],
                      labels,
                      scale_factor=2.0,
                      scale_mode='none',
                      vmin=0.0,
                      vmax=3.0,
                      figure=fig1)

        """
        mlab.points3d(p[-2:, 0],
                      p[-2:, 1],
                      p[-2:, 2],
                      labels[-2:]*0 + 3,
                      scale_factor=0.16 * 1.5 * 50,
                      scale_mode='none',
                      mode='cube',
                      vmin=0.0,
                      vmax=3.0,
                      figure=fig1)
        mlab.points3d(p[-1:, 0],
                      p[-1:, 1],
                      p[-1:, 2],
                      labels[-1:]*0 + 2,
                      scale_factor=0.16 * 2 * 2.5 * 1.5 * 50,
                      scale_mode='none',
                      mode='sphere',
                      vmin=0.0,
                      vmax=3.0,
                      figure=fig1)

        """

        # New title
        title_str = '<([) b_i={:d} (])>    <(,) l_i={:d} (.)>    <(N) n_i={:d} (M)>'.format(b_i, l_i, neighb_i)
        mlab.title(title_str, color=(0, 0, 0), size=0.3, height=0.90)
        if show_pools:
            text = 'pools (switch with G)'
        else:
            text = 'neighbors (switch with G)'
        mlab.text(0.01, 0.01, text, color=(0, 0, 0), width=0.3)
        mlab.orientation_axes()

        return

    def keyboard_callback(vtk_obj, event):
        global b_i, l_i, neighb_i, show_pools

        if vtk_obj.GetKeyCode() in ['[', '{']:
            b_i = (b_i - 1) % len(all_points[l_i])
            neighb_i = 0
            update_scene()

        elif vtk_obj.GetKeyCode() in [']', '}']:
            b_i = (b_i + 1) % len(all_points[l_i])
            neighb_i = 0
            update_scene()

        elif vtk_obj.GetKeyCode() in [',', '<']:
            if show_pools:
                l_i = (l_i - 1) % (len(all_points) - 1)
            else:
                l_i = (l_i - 1) % len(all_points)
            neighb_i = 0
            update_scene()

        elif vtk_obj.GetKeyCode() in ['.', '>']:
            if show_pools:
                l_i = (l_i + 1) % (len(all_points) - 1)
            else:
                l_i = (l_i + 1) % len(all_points)
            neighb_i = 0
            update_scene()

        elif vtk_obj.GetKeyCode() in ['n', 'N']:
            neighb_i = (neighb_i - 1) % all_points[l_i][b_i].shape[0]
            update_scene()

        elif vtk_obj.GetKeyCode() in ['m', 'M']:
            neighb_i = (neighb_i + 1) % all_points[l_i][b_i].shape[0]
            update_scene()

        elif vtk_obj.GetKeyCode() in ['g', 'G']:
            if l_i < len(all_points) - 1:
                show_pools = not show_pools
                neighb_i = 0
            update_scene()

        return

    # Draw a first plot
    update_scene()
    fig1.scene.interactor.add_observer('KeyPressEvent', keyboard_callback)
    mlab.show()


def show_input_normals(batch):
    from mayavi import mlab

    ###########################
    # Interactive visualization
    ###########################

    # Create figure for features
    fig1 = mlab.figure('Input', bgcolor=(1, 1, 1), size=(1000, 800))
    fig1.scene.parallel_projection = False

    # Unstack batch
    all_points = batch.unstack_points()
    all_neighbors = batch.unstack_neighbors()
    all_normals = batch.unstack_pools()

    # Indices
    global b_i, l_i, neighb_i, show_pools
    b_i = 0
    l_i = 0
    neighb_i = 0
    show_pools = False

    def update_scene():

        #  clear figure
        mlab.clf(fig1)

        # Rescale points for visu
        p = (all_points[l_i][b_i] * 1.5 + np.array([1.0, 1.0, 1.0])) * 50.0
        labels = p[:, 2] * 0

        if show_pools:
            p2 = (all_points[l_i + 1][b_i][neighb_i:neighb_i + 1] * 1.5 + np.array([1.0, 1.0, 1.0])) * 50.0
            p = np.vstack((p, p2))
            labels = np.hstack((labels, np.ones((1,), dtype=np.int32) * 3))
            pool_inds = all_pools[l_i][b_i][neighb_i]
            pool_inds = pool_inds[pool_inds >= 0]
            labels[pool_inds] = 2
        else:
            neighb_inds = all_neighbors[l_i][b_i][neighb_i]
            neighb_inds = neighb_inds[neighb_inds >= 0]
            labels[neighb_inds] = 2
            labels[neighb_i] = 3

        # Show point clouds colorized with activations
        mlab.points3d(p[:, 0],
                      p[:, 1],
                      p[:, 2],
                      labels,
                      scale_factor=2.0,
                      scale_mode='none',
                      vmin=0.0,
                      vmax=3.0,
                      figure=fig1)

        """
        mlab.points3d(p[-2:, 0],
                      p[-2:, 1],
                      p[-2:, 2],
                      labels[-2:]*0 + 3,
                      scale_factor=0.16 * 1.5 * 50,
                      scale_mode='none',
                      mode='cube',
                      vmin=0.0,
                      vmax=3.0,
                      figure=fig1)
        mlab.points3d(p[-1:, 0],
                      p[-1:, 1],
                      p[-1:, 2],
                      labels[-1:]*0 + 2,
                      scale_factor=0.16 * 2 * 2.5 * 1.5 * 50,
                      scale_mode='none',
                      mode='sphere',
                      vmin=0.0,
                      vmax=3.0,
                      figure=fig1)

        """

        # New title
        title_str = '<([) b_i={:d} (])>    <(,) l_i={:d} (.)>    <(N) n_i={:d} (M)>'.format(b_i, l_i, neighb_i)
        mlab.title(title_str, color=(0, 0, 0), size=0.3, height=0.90)
        if show_pools:
            text = 'pools (switch with G)'
        else:
            text = 'neighbors (switch with G)'
        mlab.text(0.01, 0.01, text, color=(0, 0, 0), width=0.3)
        mlab.orientation_axes()

        return

    def keyboard_callback(vtk_obj, event):
        global b_i, l_i, neighb_i, show_pools

        if vtk_obj.GetKeyCode() in ['[', '{']:
            b_i = (b_i - 1) % len(all_points[l_i])
            neighb_i = 0
            update_scene()

        elif vtk_obj.GetKeyCode() in [']', '}']:
            b_i = (b_i + 1) % len(all_points[l_i])
            neighb_i = 0
            update_scene()

        elif vtk_obj.GetKeyCode() in [',', '<']:
            if show_pools:
                l_i = (l_i - 1) % (len(all_points) - 1)
            else:
                l_i = (l_i - 1) % len(all_points)
            neighb_i = 0
            update_scene()

        elif vtk_obj.GetKeyCode() in ['.', '>']:
            if show_pools:
                l_i = (l_i + 1) % (len(all_points) - 1)
            else:
                l_i = (l_i + 1) % len(all_points)
            neighb_i = 0
            update_scene()

        elif vtk_obj.GetKeyCode() in ['n', 'N']:
            neighb_i = (neighb_i - 1) % all_points[l_i][b_i].shape[0]
            update_scene()

        elif vtk_obj.GetKeyCode() in ['m', 'M']:
            neighb_i = (neighb_i + 1) % all_points[l_i][b_i].shape[0]
            update_scene()

        elif vtk_obj.GetKeyCode() in ['g', 'G']:
            if l_i < len(all_points) - 1:
                show_pools = not show_pools
                neighb_i = 0
            update_scene()

        return

    # Draw a first plot
    update_scene()
    fig1.scene.interactor.add_observer('KeyPressEvent', keyboard_callback)
    mlab.show()


def show_point_cloud(points):
    from mayavi import mlab


    # Create figure for features
    #fig1 = mlab.figure('Deformations', bgcolor=(1.0, 1.0, 1.0), size=(1280, 920))
    #fig1.scene.parallel_projection = False

    mlab.points3d(points[:, 0],
                  points[:, 1],
                  points[:, 2],
                  resolution=8,
                  scale_factor=1,
                  scale_mode='none',
                  color=(0, 1, 1))
    #mlab.show()

    #TODO: mayavi interactive mode?

    input('press enter to resume script')


    a = 1/0


def show_bundle_adjustment(bundle_frames_path):
    from mayavi import mlab

    ##################################
    # Load ply file with bundle frames
    ##################################

    # Load ply
    data = read_ply(bundle_frames_path)
    points = np.vstack((data['x'], data['y'], data['z'])).T
    bundle_inds = data['b']
    steps = data['s']

    # Get steps and bundli indices
    B = int(np.max(bundle_inds)) + 1
    S = int(np.max(steps)) + 1

    # Adjust bundle inds for color
    bundle_inds[bundle_inds < 0.1] -= 2 * B

    # Reshape points
    points = points.reshape(S, B, -1, 3)
    bundle_inds = bundle_inds.reshape(S, B, -1)

    # Get original frame
    size = np.linalg.norm(points[0, 0, -1] - points[0, 0, 0])
    x = np.linspace(0, size, 50, dtype=np.float32)
    p0 = np.hstack((np.vstack((x, x * 0, x * 0)), np.vstack((x * 0, x, x * 0)), np.vstack((x * 0, x * 0, x)))).T


    ###############
    # Visualization
    ###############

    # Create figure for features
    fig1 = mlab.figure('Bundle', bgcolor=(1.0, 1.0, 1.0), size=(1280, 920))
    fig1.scene.parallel_projection = False

    # Indices
    global s, plots, p_scale
    p_scale = 0.003
    s = 0
    plots = {}

    def update_scene():
        global s, plots, p_scale

        # Get the current view
        v = mlab.view()
        roll = mlab.roll()

        #  clear figure
        for key in plots.keys():
            plots[key].remove()

        plots = {}

        # Get points we want to show
        p = points[s].reshape(-1, 3)
        b = bundle_inds[s].reshape(-1)

        plots['points'] = mlab.points3d(p[:, 0],
                                        p[:, 1],
                                        p[:, 2],
                                        b,
                                        resolution=8,
                                        scale_factor=p_scale,
                                        scale_mode='none',
                                        figure=fig1)

        # Show original frame
        plots['points0'] = mlab.points3d(p0[:, 0],
                                         p0[:, 1],
                                         p0[:, 2],
                                         resolution=8,
                                         scale_factor=p_scale * 2,
                                         scale_mode='none',
                                         color=(0, 0, 0),
                                         figure=fig1)

        # New title
        plots['title'] = mlab.title(str(s), color=(0, 0, 0), size=0.3, height=0.01)
        text = '<--- (press g for previous)' + 50 * ' ' + '(press h for next) --->'
        plots['text'] = mlab.text(0.01, 0.01, text, color=(0, 0, 0), width=0.98)
        # plots['orient'] = mlab.orientation_axes()

        # Set the saved view
        mlab.view(*v)
        mlab.roll(roll)

        return

    def keyboard_callback(vtk_obj, event):
        global s, plots, p_scale

        if vtk_obj.GetKeyCode() in ['b', 'B']:
            p_scale /= 1.5
            update_scene()

        elif vtk_obj.GetKeyCode() in ['n', 'N']:
            p_scale *= 1.5
            update_scene()

        if vtk_obj.GetKeyCode() in ['g', 'G']:
            s = (s - 1) % S
            update_scene()

        elif vtk_obj.GetKeyCode() in ['h', 'H']:
            s = (s + 1) % S
            update_scene()

        return

    # Draw a first plot
    update_scene()
    fig1.scene.interactor.add_observer('KeyPressEvent', keyboard_callback)
    mlab.show()


def compare_Shapenet_results(logs, log_names):
    from mayavi import mlab

    ######
    # Init
    ######

    # dataset parameters
    n_parts = [4, 2, 2, 4, 4, 3, 3, 2, 4, 2, 6, 2, 3, 3, 3, 3]
    label_to_names = {0: 'Airplane',
                      1: 'Bag',
                      2: 'Cap',
                      3: 'Car',
                      4: 'Chair',
                      5: 'Earphone',
                      6: 'Guitar',
                      7: 'Knife',
                      8: 'Lamp',
                      9: 'Laptop',
                      10: 'Motorbike',
                      11: 'Mug',
                      12: 'Pistol',
                      13: 'Rocket',
                      14: 'Skateboard',
                      15: 'Table'}
    name_to_label = {v: k for k, v in label_to_names.items()}

    # Get filenames that are common to all tests
    example_paths = [join(f, '_Examples') for f in logs]
    n_log = len(logs)
    file_names = np.sort([f for f in listdir(example_paths[0]) if f.endswith('.ply')])
    for log_path in example_paths:
        file_names = np.sort([f for f in listdir(log_path) if f in file_names])

    logs_clouds = []
    logs_labels = []
    logs_preds = []
    logs_IoUs = []
    logs_objs = []
    for log_path in example_paths:
        logs_clouds += [[]]
        logs_labels += [[]]
        logs_preds += [[]]
        logs_IoUs += [[]]
        logs_objs += [[]]
        for file_name in file_names:
            file_path = join(log_path, file_name)
            obj_lbl = name_to_label[file_name.split('_')[0]]
            logs_objs[-1] += [obj_lbl]

            data = read_ply(file_path)
            logs_clouds[-1] += [np.vstack((data['x'], -data['y'], data['z'])).T]

            lbls = data['gt']
            preds = data['pre']
            logs_labels[-1] += [lbls]
            logs_preds[-1] += [preds]

            # Compute IoUs
            parts = [j for j in range(n_parts[obj_lbl])]
            C = fast_confusion(lbls, preds, np.array(parts, dtype=np.int32))
            IoU = np.mean(IoU_from_confusions(C))

            logs_IoUs[-1] += [IoU]

    logs_IoUs = np.array(logs_IoUs, dtype=np.float32).T

    ###########################
    # Interactive visualization
    ###########################

    # Create figure for features
    fig1 = mlab.figure('Models', bgcolor=(1, 1, 1), size=(1300, 700))
    fig1.scene.parallel_projection = False

    # Indices
    global file_i, v, roll
    file_i = 0
    v = None
    roll = None

    def update_scene():
        global v, roll

        # Get current view
        if v is not None:
            v = mlab.view()
            roll = mlab.roll()

        #  clear figure
        mlab.clf(fig1)

        # Plot new data feature
        print(logs_IoUs[file_i])

        n_p = n_parts[logs_objs[0][file_i]]

        # Show point clouds colorized with activations
        activations = mlab.points3d(logs_clouds[0][file_i][:, 0],
                                    logs_clouds[0][file_i][:, 1],
                                    logs_clouds[0][file_i][:, 2],
                                    logs_labels[0][file_i],
                                    scale_factor=0.03,
                                    scale_mode='none',
                                    vmin=0,
                                    vmax=n_p-1,
                                    figure=fig1)

        for i in range(n_log):
            # Rescale points for visu
            pts = logs_clouds[i][file_i] + np.array([0.0, 2.5, 0.0]) * (i + 1)

            # Show point clouds colorized with activations
            activations = mlab.points3d(pts[:, 0],
                                        pts[:, 1],
                                        pts[:, 2],
                                        logs_preds[i][file_i],
                                        scale_factor=0.03,
                                        scale_mode='none',
                                        vmin=0,
                                        vmax=n_p-1,
                                        figure=fig1)

        # New title

        s = '{:s}: GT'.format(file_names[file_i])
        for IoU in logs_IoUs[file_i]:
            s += ' / {:.1f}'.format(100 * IoU)

        mlab.title(s, color=(0, 0, 0), size=0.2, height=0.01)
        mlab.orientation_axes()

        # Set/Get the current view
        if v is None:
            v = mlab.view()
            roll = mlab.roll()
        else:
            mlab.view(*v)
            mlab.roll(roll)

        return

    def keyboard_callback(vtk_obj, event):
        global file_i, points, labels, predictions

        if vtk_obj.GetKeyCode() in ['g', 'G']:
            file_i = (file_i - 1) % len(logs_labels[0])
            update_scene()

        elif vtk_obj.GetKeyCode() in ['h', 'H']:
            file_i = (file_i + 1) % len(logs_labels[0])
            update_scene()

        return

    # Draw a first plot
    update_scene()
    fig1.scene.interactor.add_observer('KeyPressEvent', keyboard_callback)
    mlab.show()


def save_future_anim(gif_name, future_imgs, close=False):

    fig, ax = plt.subplots()
    im = plt.imshow(future_imgs[0])

    def animate(i):
        im.set_array(future_imgs[i])
        return [im]

    anim = FuncAnimation(fig, animate,
                         frames=np.arange(future_imgs.shape[0]),
                         interval=50,
                         blit=True)
    anim.save(gif_name, fps=20)

    if close:
        plt.close(fig)
        return
    else:
        return fig, anim


def colorize_collisions(collision_imgs, k_background=True):

    if k_background:
        colored_img = (collision_imgs[..., [2, 0, 1]] * 255).astype(np.uint8)
        background = np.array([0, 0, 0], dtype=np.float64)
        shortT1 = np.array([1.0, 0, 0], dtype=np.float64)
        shortT2 = np.array([1.0, 1.0, 0], dtype=np.float64)
    else:
        colored_img = ((1.0 - collision_imgs[..., [1, 2, 0]]) * 255).astype(np.uint8)
        background = np.array([1, 1, 1], dtype=np.float64)
        shortT1 = np.array([1.0, 0, 0], dtype=np.float64)
        shortT2 = np.array([1.0, 1.0, 0], dtype=np.float64)

    resolution = 256
    shortT1 = np.array([1.0, 0, 0], dtype=np.float64)
    shortT2 = np.array([1.0, 1.0, 0], dtype=np.float64)
    cmap_shortT = np.vstack((np.linspace(background, shortT1, resolution), np.linspace(shortT1, shortT2, resolution)))
    cmap_shortT = (cmap_shortT * 255).astype(np.uint8)                        
    shortT = collision_imgs[..., 2]
    mask = shortT > 0.05
    inds = np.around(shortT * (cmap_shortT.shape[0] - 1)).astype(np.int32)
    pooled_cmap = cmap_shortT[inds]
    colored_img[mask] = pooled_cmap[mask]

    return colored_img


def zoom_collisions(collision_imgs, zoom=1):

    if zoom > 1:
        collision_imgs = np.repeat(collision_imgs, zoom, axis=-3)
        collision_imgs = np.repeat(collision_imgs, zoom, axis=-2)

    return collision_imgs


def fast_save_future_anim(gif_name, future_imgs, zoom=1, correction=False):

    if (future_imgs.dtype == np.uint8):
        future_imgs = future_imgs.astype(np.float32) / 255

    # Apply colorization
    if correction:
        colored_img = colorize_collisions(future_imgs)
    else:
        colored_img = future_imgs

    # Apply zoom
    colored_img = zoom_collisions(colored_img, zoom)

    # Save
    imageio.mimsave(gif_name, colored_img, fps=20)
    return


def save_zoom_img(im_name, img, zoom=1, correction=False, k_background=True):

    if (img.dtype == np.uint8):
        img = img.astype(np.float32) / 255

    if k_background:
        colored_img = (img[..., [2, 0, 1]] * 255).astype(np.uint8)
        background = np.array([0, 0, 0], dtype=np.float64)
        shortT1 = np.array([1.0, 0, 0], dtype=np.float64)
        shortT2 = np.array([1.0, 1.0, 0], dtype=np.float64)
    else:
        colored_img = ((1.0 - img[..., [1, 2, 0]]) * 255).astype(np.uint8)
        background = np.array([1, 1, 1], dtype=np.float64)
        shortT1 = np.array([1.0, 0, 0], dtype=np.float64)
        shortT2 = np.array([1.0, 1.0, 0], dtype=np.float64)

    if correction:
        resolution = 256
        shortT1 = np.array([1.0, 0, 0], dtype=np.float64)
        shortT2 = np.array([1.0, 1.0, 0], dtype=np.float64)
        cmap_shortT = np.vstack((np.linspace(background, shortT1, resolution),
                                np.linspace(shortT1, shortT2, resolution)))
        cmap_shortT = (cmap_shortT * 255).astype(np.uint8)                        
        shortT = img[..., 2]
        mask = shortT > 0.05
        inds = np.around(shortT * (cmap_shortT.shape[0] - 1)).astype(np.int32)
        pooled_cmap = cmap_shortT[inds]
        colored_img[mask] = pooled_cmap[mask]

    
    if zoom > 1:
        colored_img = np.repeat(colored_img, zoom, axis=0)
        colored_img = np.repeat(colored_img, zoom, axis=1)

    imageio.imsave(im_name, colored_img)
    return


def superpose_gt(pred_imgs, gt_imgs, ingt_imgs, ingts_fade=(50, -5)):

    # Pred shape = [..., T, H, W, 3]
    #   gt_shape = [..., T, H, W, 3]
    # ingt_shape = [..., n, H, W, 3]

    # Define color palette
    background = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    perma = np.array([1.0, 1.0, 0.0], dtype=np.float64)
    longT = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    shortT1 = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    shortT2 = np.array([1.0, 0.0, 1.0], dtype=np.float64)
    gt_shortT = np.array([0.0, 1.0, 1.0], dtype=np.float64)
    past_shortT = np.array([0.0, 1.0, 0.0], dtype=np.float64)

    # Define colormaps
    resolution = 256
    cmap_perma = (np.linspace(background, perma, resolution) * 255).astype(np.uint8)
    cmap_longT = (np.linspace(background, longT, resolution) * 255).astype(np.uint8)
    cmap_gt_shortT = (np.linspace(background, gt_shortT, resolution) * 255).astype(np.uint8)
    cmap_past_shortT = (np.linspace(background, past_shortT, resolution) * 255).astype(np.uint8)
    cmap_shortT = np.vstack((np.linspace(background, shortT1, resolution), np.linspace(shortT1, shortT2, resolution)))
    cmap_shortT = (cmap_shortT * 255).astype(np.uint8)

    # Create past and future images
    future_imgs = np.zeros_like(pred_imgs).astype(np.uint8)
    past_imgs = np.zeros_like(ingt_imgs).astype(np.uint8)

    # Color future image
    for cmap, values in zip([cmap_perma, cmap_longT, cmap_shortT, cmap_gt_shortT],
                            [pred_imgs[..., 0], pred_imgs[..., 1], pred_imgs[..., 2], gt_imgs[..., 2]]):
        mask = values > 0.05
        pooled_cmap = cmap[np.around(values * (cmap.shape[0] - 1)).astype(np.int32)]
        future_imgs[mask] = np.minimum(future_imgs[mask] + pooled_cmap[mask], 255)

    # Color past image
    for cmap, values in zip([cmap_perma, cmap_longT, cmap_past_shortT],
                            [ingt_imgs[..., 0], ingt_imgs[..., 1], ingt_imgs[..., 2]]):
        mask = values > 0.05
        pooled_cmap = cmap[np.around(values * (cmap.shape[0] - 1)).astype(np.int32)]
        past_imgs[mask] = np.minimum(past_imgs[mask] + pooled_cmap[mask], 255)

    # Concatenate past and future
    all_imgs = np.concatenate((past_imgs, future_imgs), axis=-4)
    
    # Add ghost of the input
    mask = np.sum(ingt_imgs[..., 2], axis=-3, keepdims=True) > 0.05
    g = all_imgs[..., 1]
    mask = np.tile(mask, (g.shape[-3], 1, 1))

    fade_init, fade_inc = ingts_fade

    if fade_inc < 0:
        fading_green = np.hstack([np.ones(ingt_imgs.shape[-4]) * fade_init, np.arange(fade_init, 0, fade_inc)])
        fading_green = fading_green[:g.shape[-3]]
    else:
        fading_green = np.ones(g.shape[-3]) * fade_init
    fading_green = np.pad(fading_green, (0, g.shape[-3] - fading_green.shape[0]))
    fading_green = np.reshape(fading_green, (-1, 1, 1))
    fading_green = np.zeros_like(g) + fading_green
    g[mask] = np.minimum(g[mask] + fading_green[mask], 255)

    return all_imgs


def superpose_gt_contour(pred_imgs, gt_imgs, ingt_imgs, no_in=True):

    # Pred shape = [..., T, H, W, 3]
    #   gt_shape = [..., T, H, W, 3]
    # ingt_shape = [..., n, H, W, 3]

    # Define color palette
    background = np.array([0, 0, 0], dtype=np.float64)
    perma = np.array([1.0, 1.0, 0.0], dtype=np.float64)
    longT = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    shortT1 = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    shortT2 = np.array([1.0, 0.0, 1.0], dtype=np.float64)
    gt_shortT = np.array([0.0, 1.0, 1.0], dtype=np.float64)
    past_shortT = np.array([0.0, 1.0, 0.0], dtype=np.float64)

    # Merge color function
    if np.mean(background) > 0.5:
        merge_func = np.minimum
    else:
        merge_func = np.maximum

    # Define colormaps
    resolution = 256
    cmap_perma = (np.linspace(background, perma, resolution) * 255).astype(np.uint8)
    cmap_longT = (np.linspace(background, longT, resolution) * 255).astype(np.uint8)
    cmap_gt_shortT = (np.linspace(background, gt_shortT, resolution) * 255).astype(np.uint8)
    cmap_past_shortT = (np.linspace(background, past_shortT, resolution) * 255).astype(np.uint8)
    cmap_shortT = np.vstack((np.linspace(background, shortT1, resolution), np.linspace(shortT1, shortT2, resolution)))
    cmap_shortT = (cmap_shortT * 255).astype(np.uint8)

    # Create past and future images
    future_imgs = (np.ones_like(pred_imgs) * background * 255).astype(np.uint8)
    past_imgs = np.zeros_like(ingt_imgs).astype(np.uint8)

    # Color future image
    for cmap, values in zip([cmap_perma, cmap_longT, cmap_shortT],
                            [pred_imgs[..., 0], pred_imgs[..., 1], pred_imgs[..., 2]]):   
        mask = values > 0.05
        pooled_cmap = cmap[np.around(values * (cmap.shape[0] - 1)).astype(np.int32)]
        future_imgs[mask] = merge_func(future_imgs[mask], pooled_cmap[mask])

    # Add GT contour
    mask = gt_imgs[..., 2] > 0.05
    close_struct = np.ones((1, 1, 10, 10))
    erode_struct = np.ones((1, 1, 5, 5))
    mask = ndimage.binary_closing(mask, structure=close_struct, iterations=2)
    mask = np.logical_and(mask, np.logical_not(ndimage.binary_erosion(mask, structure=erode_struct)))
    gt_color = (past_shortT * 255).astype(np.uint8)
    future_imgs[mask] = merge_func(future_imgs[mask], gt_color)

    # Color past image
    for cmap, values in zip([cmap_perma, cmap_longT, cmap_past_shortT],
                            [ingt_imgs[..., 0], ingt_imgs[..., 1], ingt_imgs[..., 2]]):
        mask = values > 0.05
        pooled_cmap = cmap[np.around(values * (cmap.shape[0] - 1)).astype(np.int32)]
        past_imgs[mask] = np.minimum(past_imgs[mask] + pooled_cmap[mask], 255)

    # Concatenate past and future
    if no_in:
        all_imgs = future_imgs
    else:
        all_imgs = np.concatenate((past_imgs, future_imgs), axis=-4)
    
    # # Add ghost of the input
    # mask = np.sum(ingt_imgs[..., 2], axis=-3, keepdims=True) > 0.05
    # bin_struct = np.ones((1, 1, 3, 3))
    # mask = np.logical_and(mask, np.logical_not(ndimage.binary_erosion(mask, structure=bin_struct)))
    # g = all_imgs[..., 1]
    # mask = np.tile(mask, (g.shape[-3], 1, 1))
    # fade_init, fade_inc = ingts_fade

    # if fade_inc < 0:
    #     fading_green = np.hstack([np.ones(ingt_imgs.shape[-4]) * fade_init, np.arange(fade_init, 0, fade_inc)])
    #     fading_green = fading_green[:g.shape[-3]]
    # else:
    #     fading_green = np.ones(g.shape[-3]) * fade_init
    # fading_green = np.pad(fading_green, (0, g.shape[-3] - fading_green.shape[0]))
    # fading_green = np.reshape(fading_green, (-1, 1, 1))
    # fading_green = np.zeros_like(g) + fading_green
    # g[mask] = np.minimum(g[mask] + fading_green[mask], 255)

    return all_imgs


def get_local_maxima(data, neighborhood_size=5, threshold=0.1, smooth=-1):

    # Optional smoothing for better max value
    if smooth > 0:
        datatype = data.dtype
        data = data.astype(np.float32)
        sigmas = np.ones(data.ndim, dtype=np.float32)
        sigmas[-2:] = smooth / 3
        sigmas[:-2] = smooth / 30
        data = filters.gaussian_filter(data, sigmas, truncate=smooth)
        data = data.astype(datatype)
    
    # Get maxima positions as a mask
    dim_size = np.ones(data.ndim, dtype=np.int32)
    dim_size[-2:] = neighborhood_size
    data_max = filters.maximum_filter(data, dim_size)
    max_mask = (data == data_max)

    # Remove maxima if their peak is not higher than threshold in the neighborhood
    data_min = filters.minimum_filter(data, dim_size)
    diff = ((data_max - data_min) > threshold)
    max_mask[diff == 0] = 0

    return max_mask



def superpose_and_merge(pred_imgs, gt_imgs, ingt_imgs, traj=True, contour=False):

    # Pred shape = [..., T, H, W, 3]
    #   gt_shape = [..., T, H, W, 3]
    # ingt_shape = [..., n, H, W, 3]

    # Define color palette
    background = np.array([0, 0, 0], dtype=np.float64)
    perma = np.array([1.0, 1.0, 0.0], dtype=np.float64)
    longT = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    shortT1 = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    shortT2 = np.array([1.0, 0.0, 0.2], dtype=np.float64)
    gt_shortT = np.array([0.0, 1.0, 1.0], dtype=np.float64)
    past_shortT = np.array([0.0, 1.0, 0.0], dtype=np.float64)

    # Merge color function
    if np.mean(background) > 0.5:
        merge_func = np.min
        add_func = np.minimum
        
    else:
        merge_func = np.max
        add_func = np.maximum

    # Define colormaps
    resolution = 256
    cmap_perma = (np.linspace(background, perma, resolution) * 255).astype(np.uint8)
    cmap_longT = (np.linspace(background, longT, resolution) * 255).astype(np.uint8)
    cmap_gt_shortT = (np.linspace(background, gt_shortT, resolution) * 255).astype(np.uint8)
    cmap_past_shortT = (np.linspace(background, past_shortT, resolution) * 255).astype(np.uint8)
    cmap_shortT = np.vstack((np.linspace(background, shortT1, resolution), np.linspace(shortT1, shortT2, resolution)))
    cmap_shortT = (cmap_shortT * 255).astype(np.uint8)

    # Create past and future images
    future_imgs = (np.ones_like(pred_imgs) * background * 255).astype(np.uint8)

    # Color future image
    for cmap, values in zip([cmap_perma, cmap_longT, cmap_shortT],
                            [pred_imgs[..., 0], pred_imgs[..., 1], pred_imgs[..., 2]]):   
        mask = values > 0.05
        pooled_cmap = cmap[np.around(values * (cmap.shape[0] - 1)).astype(np.int32)]
        future_imgs[mask] = add_func(future_imgs[mask], pooled_cmap[mask])


    future_imgs = merge_func(future_imgs, axis=-4)
    

    if traj:

        # Add GT traj
        max_mask = get_local_maxima(gt_imgs[..., 2], neighborhood_size=15, threshold=0.3, smooth=9)
        
        shape = np.ones(max_mask.ndim, dtype=np.int32)
        shape[-3:] = 3
        close_struct = np.ones(shape)
        max_mask = ndimage.binary_dilation(max_mask, structure=close_struct, iterations=4)
        max_mask = ndimage.binary_erosion(max_mask, structure=close_struct, iterations=2)

        cmap_tmp = (np.linspace(gt_shortT, past_shortT, max_mask.shape[-3]) * 255).astype(np.uint8)
        shape = np.ones(max_mask.ndim, dtype=np.int32)
        shape[-3] = max_mask.shape[-3]
        shape = np.hstack((shape, [3]))
        cmap_tmp = np.reshape(cmap_tmp, shape)

        collored_traj = np.expand_dims(max_mask, -1) * cmap_tmp
        
        max_mask = np.max(max_mask, axis=-3)
        collored_traj = np.max(collored_traj, axis=-4)
        
        if np.sum(max_mask.astype(np.int32)) > 0:
            future_imgs[max_mask] = collored_traj[max_mask]


    if contour:

        # Add GT contour
        gt_imgs = np.max(gt_imgs, axis=-4)
        mask = gt_imgs[..., 2] > 0.05

        shape = np.ones(mask.ndim, dtype=np.int32)
        shape[-2:] = 10
        close_struct = np.ones(shape)
        shape[-2:] = 5
        erode_struct = np.ones(shape)
        mask = ndimage.binary_closing(mask, structure=close_struct, iterations=2)
        mask = np.logical_and(mask, np.logical_not(ndimage.binary_erosion(mask, structure=erode_struct)))
        gt_color = (past_shortT * 255).astype(np.uint8)
        future_imgs[mask] = add_func(future_imgs[mask], gt_color)

    all_imgs = future_imgs


    return all_imgs


def show_local_maxima(collision_risk, neighborhood_size, threshold, show=True):

    # Find local maxima
    # *****************

    all_x = []
    all_y = []
    for data in collision_risk:

        # Get maxima positions as a mask
        data_max = filters.maximum_filter(data, neighborhood_size)
        maxima = (data == data_max)

        # Remove maxima if their peak is not higher than threshold in the neighborhood
        data_min = filters.minimum_filter(data, neighborhood_size)
        diff = ((data_max - data_min) > threshold)
        maxima[diff == 0] = 0

        labeled, num_objects = ndimage.label(maxima)
        slices = ndimage.find_objects(labeled)
        x, y = [], []
        for dy,dx in slices:
            x_center = (dx.start + dx.stop - 1)/2
            x.append(x_center)
            y_center = (dy.start + dy.stop - 1)/2    
            y.append(y_center)

        all_x.append(x)
        all_y.append(y)


    fig, ax = plt.subplots(1, 1)
    images = []
    images.append(ax.imshow(collision_risk[0]))
    plt.autoscale(False)
    images += ax.plot(all_x[0], all_y[0], 'ro')

    def animate(i):
        images[0].set_array(collision_risk[i])
        images[1].set_xdata(all_x[i])
        images[1].set_ydata(all_y[i])
        return images

    anim = FuncAnimation(fig, animate,
                         frames=np.arange(collision_risk.shape[0]),
                         interval=50,
                         blit=True)

    if show:
        plt.show()

    

    return fig, anim


def show_risk_diffusion(collision_risk, dl=0.12, diff_range=1.5, p=5, show=True):


    # Diffuse the collision risk
    def diffuse_risk(im, dist_kernel):
        
        def kernel_func(x):
            return np.max(x * dist_kernel.ravel())

        return ndimage.generic_filter(im, kernel_func, size=dist_kernel.shape[0])

    device = torch.device("cuda:1")

    # Definition of the kernel
    k_range = int(np.ceil(diff_range / dl))
    k = 2 * k_range + 1
    dist_kernel = np.zeros((k, k))
    for i, vv in enumerate(dist_kernel):
        for j, v in  enumerate(vv):
            dist_kernel[i, j] = np.sqrt((i - k_range) ** 2 + (j - k_range) ** 2)
    dist_kernel = (np.clip(1.0 - dist_kernel * dl / diff_range, 0, 1)) ** p

    # Save kernel
    cm = plt.get_cmap('viridis')
    print(dist_kernel.shape)
    dist_kernel_visu = (zoom_collisions(cm(dist_kernel), 5) * 255).astype(np.uint8)
    print(dist_kernel_visu.shape)
    imageio.imwrite('results/show_kernel.png', dist_kernel_visu)

    one_kernel = (dist_kernel > 1e-6).astype(dist_kernel.dtype)
    fixed_conv = torch.nn.Conv2d(1, 2, k, stride=1, padding=k_range, bias=False)
    fixed_conv.weight.requires_grad = False
    fixed_conv.weight *= 0
    fixed_conv.weight += torch.from_numpy(np.expand_dims(np.stack((dist_kernel, one_kernel)), 1))
    fixed_conv.to(device)

    def diffuse_risk_gpu(im, conv, device):

        gpu_im = torch.from_numpy(im)
        gpu_im = torch.unsqueeze(gpu_im, 1)
        gpu_im = gpu_im.to(device)

        dilated_im = fixed_conv(gpu_im)

        return dilated_im.detach().cpu().numpy()



    # # Remove residual preds (hard hysteresis)
    # collision_risk *= (collision_risk > 0.05).astype(collision_risk.dtype)

    # Remove residual preds (soft hysteresis)
    lim1 = 0.05
    lim2 = 0.08
    dlim = lim2 - lim1
    mask0 = collision_risk <= lim1
    mask1 = np.logical_and(collision_risk < lim2, collision_risk > lim1)
    collision_risk[mask0] *= 0
    collision_risk[mask1] *= (1 - ((collision_risk[mask1] - lim2) / dlim) ** 2) ** 2

    print('Start gpu diffusion')
    t0 = time.time()

    # Diffuse Static risk

    #static_risk = (np.max(collision_risk[..., :2], axis=-1)[:1] > 0.3).astype(collision_risk.dtype)
    static_risk = np.max(collision_risk[..., :2], axis=-1)[:1]

    static_risk = np.tile(static_risk, (collision_risk.shape[0], 1, 1))
    diffused_0 = diffuse_risk_gpu(static_risk, fixed_conv, device)
    diffused_0 = diffused_0[:, 0, :, :]

    static_normalized = static_risk / (diffused_0 + 1e-6)
    diffused_0 = diffuse_risk_gpu(static_normalized, fixed_conv, device)
    diffused_0 = diffused_0[:, 0, :, :]

    # Diffuse Moving risk
    diffused_1 = diffuse_risk_gpu(collision_risk[..., 2], fixed_conv, device)
    diffused_1 = diffused_1[:, 0, :, :]

    # moving_normalized = collision_risk[..., 2] / (diffused_1 + 1e-6)
    # diffused_1 = diffuse_risk_gpu(moving_normalized, fixed_conv, device)
    # diffused_1 = diffused_1[:, 0, :, :]

    # Squared root
    diffused_0 = np.power(np.maximum(0, diffused_0), 1/p)
    diffused_1 = np.power(np.maximum(0, diffused_1), 1/p)


    # Rescale both to [0, 1]
    diffused_0 *= 1 / np.max(diffused_0)
    diffused_1 *= 1 / np.max(diffused_1)

    diffused_risk_gpu = np.maximum(diffused_0, diffused_1)
    

    t1 = time.time()

    print('computed gpu diffusion in {:.1f} ms'.format(1000 * (t1 - t0)))



    print('Start diffusion')
    t0 = time.time()

    # diffused_risk = []
    # for im, im2 in zip(collision_risk, static_risk):

    #     diffused_0 = diffuse_risk(im2, dist_kernel)
    #     diffused_1 = diffuse_risk(im[..., 2], dist_kernel)

    #     #diffused_risk = np.clip(diffused_0 + diffused_1, 0, 1)
    #     diffused_risk.append(np.maximum(diffused_0, diffused_1))
    
    # diffused_risk = np.stack(diffused_risk)
    # diffused_risk_gpu = diffused_risk

    t1 = time.time()

    print('computed diffusion in {:.1f} ms'.format(1000 * (t1 - t0)))
    # TODO IF WE USE SUM, PARALLELISE CONVOLUTION ON GPU????


    vmin0 = np.min(collision_risk)
    vmax0 = np.max(collision_risk)
    vmin1 = np.min(diffused_risk_gpu)
    vmax1 = np.max(diffused_risk_gpu)

    #show_preds = np.maximum(static_risk, collision_risk[..., 2])
    show_preds = np.max(collision_risk, axis=-1)
    fig, axes = plt.subplots(1, 2)
    images = []
    images.append(axes[0].imshow(show_preds[0], vmin=vmin0, vmax=vmax0))
    images.append(axes[1].imshow(diffused_risk_gpu[0], vmin=vmin1, vmax=vmax1))

    def animate(i):
        images[0].set_array(show_preds[i])
        images[1].set_array(diffused_risk_gpu[i])
        return images

    anim = FuncAnimation(fig, animate,
                         frames=np.arange(show_preds.shape[0]),
                         interval=50,
                         blit=True)

    if show:
        plt.show()

    # Get the color map by name:
    cm = plt.get_cmap('viridis')
    diffused_risk_gpu2 = diffused_risk_gpu / vmax1
    static_normalized_visu = static_normalized / np.max(static_normalized)
    imageio.mimsave('results/show_diffuse_in.gif', (zoom_collisions(cm(show_preds), 5) * 255).astype(np.uint8), fps=20)
    imageio.mimsave('results/show_diffuse_out.gif', (zoom_collisions(cm(diffused_risk_gpu2), 5) * 255).astype(np.uint8), fps=20)
    imageio.mimsave('results/norm_risk.gif', (zoom_collisions(cm(static_normalized_visu), 5) * 255).astype(np.uint8), fps=20)

    return fig, anim







