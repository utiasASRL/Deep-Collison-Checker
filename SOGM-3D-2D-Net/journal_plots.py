#
#
#      0==============================0
#      |    Deep Collision Checker    |
#      0==============================0
#
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      Callable script to test any model on any dataset
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

# Common libs
from gettext import find
import os
import shutil
import torch
os.environ.update(OMP_NUM_THREADS='1',
                  OPENBLAS_NUM_THREADS='1',
                  NUMEXPR_NUM_THREADS='1',
                  MKL_NUM_THREADS='1',)
import numpy as np
from utils.config import bcolors
import matplotlib.pyplot as plt
from os.path import isfile, join, exists
from os import listdir, remove, getcwd, makedirs
from sklearn.metrics import confusion_matrix
import time
import pickle
from torch.utils.data import DataLoader
from matplotlib.animation import FuncAnimation
import matplotlib.patches as patches
from matplotlib.widgets import Slider, Button, RadioButtons
import imageio

# My libs
from utils.config import Config
from utils.metrics import IoU_from_confusions, smooth_metrics, fast_confusion, fast_threshold_stats
from utils.ply import read_ply, write_ply
from models.architectures import FakeColliderLoss, KPCollider
from utils.tester import ModelTester
from utils.mayavi_visu import fast_save_future_anim, save_zoom_img, colorize_collisions, zoom_collisions, superpose_gt, \
    show_local_maxima, show_risk_diffusion, superpose_gt_contour, superpose_and_merge, SRM_colors

from gt_annotation_video import loading_session, motion_rectified, open_3d_vid

# Datasets
from datasets.MyhalCollision import MyhalCollisionDataset, MyhalCollisionSampler, MyhalCollisionCollate
from train_MyhalCollision import MyhalCollisionConfig
from datasets.MultiCollision import MultiCollisionDataset, MultiCollisionSampler, MultiCollisionCollate, MultiCollisionSamplerTest
from MyhalCollision_sessions import UTI3D_H_sessions, UTI3D_A_sessions, old_A_sessions, UTI3D_A_sessions_v2

from scipy import ndimage
import scipy.ndimage.filters as filters

# ----------------------------------------------------------------------------------------------------------------------
#
#           Utility functions
#       \***********************/
#


def get_local_maxima(data, neighborhood_size=5, threshold=0.1):
    
    # Get maxima positions as a mask
    data_max = filters.maximum_filter(data, neighborhood_size)
    max_mask = (data == data_max)

    # Remove maxima if their peak is not higher than threshold in the neighborhood
    data_min = filters.minimum_filter(data, neighborhood_size)
    diff = ((data_max - data_min) > threshold)
    max_mask[diff == 0] = 0

    return max_mask


def mask_to_pix(mask):
    
    # Get positions in world coordinates
    labeled, num_objects = ndimage.label(mask)
    slices = ndimage.find_objects(labeled)
    x, y = [], []

    mask_pos = []
    for dy, dx in slices:

        x_center = (dx.start + dx.stop - 1) / 2
        y_center = (dy.start + dy.stop - 1) / 2
        mask_pos.append(np.array([x_center, y_center], dtype=np.float32))

    return mask_pos


def diffusing_convolution(obstacle_range, dl, norm_p, dim1D=False):
    
    k_range = int(np.ceil(obstacle_range / dl))
    k = 2 * k_range + 1

    if dim1D:
        dist_kernel = np.zeros((k, 1, 1))
        for i in range(k):
            dist_kernel[i, 0, 0] = abs(i - k_range)

    else:
        dist_kernel = np.zeros((k, k))
        for i, vv in enumerate(dist_kernel):
            for j, v in enumerate(vv):
                dist_kernel[i, j] = np.sqrt((i - k_range) ** 2 + (j - k_range) ** 2)


    dist_kernel = np.clip(1.0 - dist_kernel * (dl / obstacle_range), 0, 1) ** norm_p

    if dim1D:
        fixed_conv = torch.nn.Conv3d(1, 1, (k, 1, 1),
                                     stride=1,
                                     padding=(k_range, 0, 0),
                                     padding_mode='replicate',
                                     bias=False)

    else:
        fixed_conv = torch.nn.Conv2d(1, 1, k, stride=1, padding=k_range, bias=False)

    fixed_conv.weight.requires_grad = False
    fixed_conv.weight *= 0

    fixed_conv.weight += torch.from_numpy(dist_kernel)

    return fixed_conv


def get_diffused_risk(config, collision_preds, dynamic_t_range=1.0, norm_p=3, normalization=True):

    # Get the GPU for PyTorch
    device = torch.device("cuda:0")
    collision_preds = collision_preds.to(device)
    
    static_range = 1.2
    dynamic_range = 1.2
    # dynamic_t_range = 1.0
    # norm_p = 3
    norm_invp = 1 / norm_p
    maxima_layers = [15]


    # Convolution for Collision risk diffusion
    static_conv = diffusing_convolution(static_range, config.dl_2D, norm_p)
    static_conv.to(device)
    dynamic_conv = diffusing_convolution(dynamic_range, config.dl_2D, norm_p)
    dynamic_conv.to(device)

    # Convolution for time diffusion
    dt = config.T_2D / config.n_2D_layers
    time_conv = diffusing_convolution(dynamic_t_range, dt, norm_p, dim1D=True)
    time_conv.to(device)

                                
    # # Remove residual preds (hard hysteresis)
    # collision_risk *= (collision_risk > 0.06).type(collision_risk.dtype)
                
    # Remove residual preds (soft hysteresis)
    # lim1 = 0.06
    # lim2 = 0.09
    lim1 = 0.15
    lim2 = 0.2
    dlim = lim2 - lim1
    mask0 = collision_preds <= lim1
    mask1 = torch.logical_and(collision_preds < lim2, collision_preds > lim1)
    collision_preds[mask0] *= 0
    collision_preds[mask1] *= (1 - ((collision_preds[mask1] - lim2) / dlim) ** 2) ** 2

    # Static risk
    # ***********

    # Get risk from static objects, [1, 1, W, H]
    static_preds = torch.unsqueeze(torch.max(collision_preds[:1, :, :, :2], dim=-1)[0], 1)

    if normalization:
        # Normalize risk values between 0 and 1 depending on density
        static_risk = static_preds / (static_conv(static_preds) + 1e-6)

    else:
        # No Normalization
        static_risk = static_preds / (static_conv(static_preds) * 0.0 + 1.0)


    # Diffuse the risk from normalized static objects
    diffused_0 = static_conv(static_risk).cpu().detach().numpy()

    # Do not repeat we only keep it for the first layer: [1, 1, W, H] -> [W, H]
    diffused_0 = np.squeeze(diffused_0)
    
    # Inverse power for p-norm
    diffused_0 = np.power(np.maximum(0, diffused_0), norm_invp)

    # Dynamic risk
    # ************

    # Get dynamic risk [T, W, H]
    dynamic_risk = collision_preds[..., 2]

    # Get high risk area
    high_risk_threshold = 0.4
    high_risk_mask = dynamic_risk > high_risk_threshold
    high_risk = torch.zeros_like(dynamic_risk)
    # high_risk[high_risk_mask] = dynamic_risk[high_risk_mask]
    high_risk[high_risk_mask] = 1

    # On the whole dynamic_risk, convolution
    # Higher value for larger area of risk even if low risk
    dynamic_risk = torch.unsqueeze(dynamic_risk, 1)
    diffused_1 = torch.squeeze(dynamic_conv(dynamic_risk))

    # Inverse power for p-norm
    diffused_1 = torch.pow(torch.clamp(diffused_1, min=0), norm_invp)

    # Rescale this low_risk at smaller value
    low_risk_value = 0.4
    diffused_1 = low_risk_value * diffused_1 / (torch.max(diffused_1) + 1e-6)

    # On the high risk, we normalize to have similar value of highest risk (around 1.0)
    high_risk_norm = torch.squeeze(dynamic_conv(torch.unsqueeze(high_risk, 1)))
    high_risk_norm = torch.unsqueeze(torch.unsqueeze(high_risk_norm, 0), 0)
    high_risk_norm = torch.squeeze(time_conv(high_risk_norm))
    high_risk_normalized = high_risk / (high_risk_norm + 1e-6)

    # We only diffuse time for high risk (as this is useful for the beginning of predictions)
    diffused_2 = torch.squeeze(dynamic_conv(torch.unsqueeze(high_risk_normalized, 1)))
    diffused_2 = torch.unsqueeze(torch.unsqueeze(diffused_2, 0), 0)
    diffused_2 = torch.squeeze(time_conv(diffused_2))

    # Inverse power for p-norm
    diffused_2 = torch.pow(torch.clamp(diffused_2, min=0), norm_invp)

    # Rescale and combine risk
    # ************************
    
    # Combine dynamic risks
    diffused_1 = torch.maximum(diffused_1, diffused_2).detach().cpu().numpy()

    # Rescale risk with a fixed value, because thx to normalization, the mx should be close to one, 
    # There are peak at border so we divide by 1.1 to take it into consideration
    diffused_1 *= 1.0 / 1.1
    diffused_0 *= 1.0 / 1.1

    if not normalization:
        diffused_0 *= 1.0 / np.max(diffused_0)


    # merge the static risk as the first layer of the vox grid (with the delay this layer is useless for dynamic)
    diffused_1[0, :, :] = diffused_0

    # Convert to uint8 for message 0-254 = prob, 255 = fixed obstacle
    diffused_risk = np.minimum(diffused_1 * 255, 255).astype(np.uint8)
    
    # # Save walls for debug
    # debug_walls = np.minimum(diffused_risk[10] * 255, 255).astype(np.uint8)
    # cm = plt.get_cmap('viridis')
    # print(batch.t0)
    # print(type(batch.t0))
    # im_name = join(ENV_HOME, 'catkin_ws/src/collision_trainer/results/debug_walls_{:.3f}.png'.format(batch.t0))
    # imageio.imwrite(im_name, zoom_collisions(cm(debug_walls), 5))

    # Get local maxima in moving obstacles
    obst_mask = None
    for layer_i in maxima_layers:
        if obst_mask is None:
            obst_mask = get_local_maxima(diffused_1[layer_i])
        else:
            obst_mask = np.logical_or(obst_mask, get_local_maxima(diffused_1[layer_i]))

    # Use max pool to get obstacles in one cell over two [H, W] => [H//2, W//2]
    stride = 2
    pool = torch.nn.MaxPool2d(stride, stride=stride, return_indices=True)
    unpool = torch.nn.MaxUnpool2d(stride, stride=stride)
    output, indices = pool(static_preds.detach())
    static_preds_2 = unpool(output, indices, output_size=static_preds.shape)

    # Merge obstacles
    obst_mask[np.squeeze(static_preds_2.cpu().numpy()) > 0.3] = 1

    # Convert to pixel positions
    obst_pos = mask_to_pix(obst_mask)

    # Get mask of static obstacles for visu
    static_mask = np.squeeze(static_preds.detach().cpu().numpy()) > 0.3
    
    return diffused_risk, obst_pos, static_mask


def print_sorted_val_table(logs_names, log_val_days, sorted_val_days, data_folders):

    # Get width of first and next columns
    n_fmt0 = np.max([len(log_name) for log_name in logs_names]) + 2
    n_fmt1 = 7

    # First print the data path
    all_val_days = [val_day for folder_val_days in sorted_val_days for val_day in folder_val_days]
    
    line0 = '{:^{width}s} '.format(' ', width=n_fmt0)
    for i, data_folder in enumerate(data_folders):
        n_fmt2 = len(sorted_val_days[i]) * (n_fmt1 + 1) - 1
        line0 += '{:^{width}s}|'.format(data_folder, width=n_fmt2)


    # Create a table with the presence of validation days for each log
    lines = ['{:^{width}s}|'.format('     \\  Val', width=n_fmt0)]
    lines += ['{:^{width}s}|'.format('Logs  \\    ', width=n_fmt0)]
    lines += ['{:-^{width}s}|'.format('', width=n_fmt0)]
    for log_i, log in enumerate(logs_names):
        lines += ['{:^{width}s}|'.format(logs_names[log_i], width=n_fmt0)]

    for d_i, val_d in enumerate(all_val_days):

        # Fill the first 3 lines of the table
        year = val_d[:4]
        month_day = val_d[5:10]
        lines[0] += '{:^{width}s}|'.format(year, width=n_fmt1)
        lines[1] += '{:^{width}s}|'.format(month_day, width=n_fmt1)
        lines[2] += '{:-^{width}s}|'.format('', width=n_fmt1)

        # Fill the other lines
        for log_i, log in enumerate(logs_names):
            if val_d in log_val_days[log_i]:
                lines[log_i+3] += '{:}{:^{width}s}{:}|'.format(bcolors.OKBLUE, u'\u2713', bcolors.ENDC, width=n_fmt1)
            else:
                lines[log_i+3] += '{:}{:^{width}s}{:}|'.format(bcolors.FAIL, u'\u2718', bcolors.ENDC, width=n_fmt1)

    print(line0)
    for line_str in lines:
        print(line_str)
    
    return


def wanted_gifs(chosen_log, sorted_val_days, dataset_paths, adding_extra=False, all_wanted_s=[], all_wanted_f=[]):

    # Get training and validation days
    config = Config()
    config.load(chosen_log)
    # val_path = join(chosen_log, 'val_preds')
    # val_days = np.unique([f[:19] for f in listdir(val_path) if f.endswith('pots.ply')])

    # Dataset
    test_dataset = MultiCollisionDataset(config,
                                         sorted_val_days,
                                         chosen_set='validation',
                                         dataset_paths=dataset_paths,
                                         simulated=['Simulation' in dataset_path for dataset_path in dataset_paths],
                                         balance_classes=False,)
    seq_inds = test_dataset.all_inds[:, 0]
    frame_inds = test_dataset.all_inds[:, 1]

    im_lim = config.radius_2D / np.sqrt(2)

    if len(all_wanted_f) < 1:

        # convertion from labels to colors
        colormap = np.array([[209, 209, 209],
                            [122, 122, 122],
                            [255, 255, 0],
                            [0, 98, 255],
                            [255, 0, 0]], dtype=np.float32) / 255

        all_pts = [[] for frames in test_dataset.frames]
        all_colors = [[] for frames in test_dataset.frames]
        all_labels = [[] for frames in test_dataset.frames]
        for s_ind, s_frames in enumerate(test_dataset.frames):

            # Advanced display
            N = len(s_frames)
            progress_n = 30
            fmt_str = '[{:<' + str(progress_n) + '}] {:5.1f}%'
            print('\nGetting gt for ' + test_dataset.sequences[s_ind])

            for f_ind, frame in enumerate(s_frames):

                # Get groundtruth in 2D points format
                gt_file = join(test_dataset.colli_path[s_ind], frame + '_2D.ply')

                # Read points
                data = read_ply(gt_file)
                pts_2D = np.vstack((data['x'], data['y'])).T
                labels_2D = data['classif']
                
                # Special treatment to old simulation annotations
                if test_dataset.sim_sequences[s_ind]:
                    times_2D = data['t']
                    time_mask = np.logical_and(times_2D > -0.001, times_2D < 0.001)
                    pts_2D = pts_2D[time_mask]
                    labels_2D = labels_2D[time_mask]

                # Recenter
                p0 = test_dataset.poses[s_ind][f_ind][:2, 3]
                centered_2D = (pts_2D - p0).astype(np.float32)

                # Remove outside boundaries of images
                img_mask = np.logical_and(centered_2D < im_lim, centered_2D > -im_lim)
                img_mask = np.logical_and(img_mask[:, 0], img_mask[:, 1])
                centered_2D = centered_2D[img_mask]
                labels_2D = labels_2D[img_mask]

                # Get the number of points per label (only present in image)
                label_v, label_n = np.unique(labels_2D, return_counts=True)
                label_count = np.zeros((colormap.shape[0],), dtype=np.int32)
                label_count[label_v] = label_n
                
                all_pts[s_ind].append(centered_2D)
                all_colors[s_ind].append(colormap[labels_2D])
                all_labels[s_ind].append(label_count)

                print('', end='\r')
                print(fmt_str.format('#' * (((f_ind + 1) * progress_n) // N), 100 * (f_ind + 1) / N), end='', flush=True)

            # Show a nice 100% progress bar
            print('', end='\r')
            print(fmt_str.format('#' * progress_n, 100), flush=True)
            print('\n')

        # Figure
        global f_i
        f_i = 0
        for s_ind, seq in enumerate(test_dataset.sequences):

            figA, axA = plt.subplots(1, 1, figsize=(10, 7))
            plt.subplots_adjust(bottom=0.25)

            # Plot first frame of seq
            plotsA = [axA.scatter(all_pts[s_ind][0][:, 0],
                                  all_pts[s_ind][0][:, 1],
                                  s=2.0,
                                  c=all_colors[s_ind][0])]

            # Show a circle of the loop closure area
            axA.add_patch(patches.Circle((0, 0), radius=0.2,
                                         edgecolor=[0.2, 0.2, 0.2],
                                         facecolor=[1.0, 0.79, 0],
                                         fill=True,
                                         lw=1))

            plt.subplots_adjust(left=0.1, bottom=0.15)

            # # Customize the graph
            # axA.grid(linestyle='-.', which='both')
            axA.set_xlim(-im_lim, im_lim)
            axA.set_ylim(-im_lim, im_lim)
            axA.set_aspect('equal', adjustable='box')
            
            # Make a horizontal slider to control the frequency.
            axcolor = 'lightgoldenrodyellow'
            axtime = plt.axes([0.1, 0.04, 0.8, 0.02], facecolor=axcolor)
            time_slider = Slider(ax=axtime,
                                 label='ind',
                                 valmin=0,
                                 valmax=len(all_pts[s_ind]) - 1,
                                 valinit=0,
                                 valstep=1)

            # The function to be called anytime a slider's value changes
            def update_PR(val):
                global f_i
                f_i = (int)(val)
                for plot_i, plot_obj in enumerate(plotsA):
                    plot_obj.set_offsets(all_pts[s_ind][f_i])
                    plot_obj.set_color(all_colors[s_ind][f_i])

            # register the update function with each slider
            time_slider.on_changed(update_PR)
            
            dyn_img = np.vstack(all_labels[s_ind]).T
            dyn_img = dyn_img[-1:]
            dyn_img[dyn_img > 10] = 10
            dyn_img[dyn_img > 0] += 10
            axdyn = plt.axes([0.1, 0.02, 0.8, 0.015])
            axdyn.imshow(dyn_img, cmap='OrRd', aspect='auto')
            axdyn.set_axis_off()
            
            wanted_f = []

            # Register event
            def onkey(event):
                if event.key == 'enter':
                    wanted_f.append(f_i)
                    print('Added current frame to the wanted indices. Now containing:', wanted_f)

                elif event.key == 'backspace':
                    if wanted_f:
                        wanted_f.pop()
                    print('removed last frame from the wanted indices. Now containing:', wanted_f)

                elif event.key == 'x':
                    if wanted_f:
                        remove_i = np.argmin([abs(i - f_i) for i in wanted_f])
                        wanted_f.pop(remove_i)
                    print('removed closest frame from the wanted indices. Now containing:', wanted_f)

            cid = figA.canvas.mpl_connect('key_press_event', onkey)
            print('\n---------------------------------------\n')
            print('Instructions:\n')
            print('> Press Enter to add current frame to the wanted indices.')
            print('> Press Backspace to remove last frame added to the wanted indices.')
            print('> Press x to to remove the closest frame to current one from the wanted indices.')
            print('\n---------------------------------------\n')

            plt.show()

            all_wanted_f += wanted_f
            all_wanted_s += [seq for _ in wanted_f]

    s_str = 'all_wanted_s = ['
    n_indent = len(s_str)
    for w_s in all_wanted_s:
        s_str += "'{:s}',\n".format(w_s) + ' ' * n_indent
    s_str = s_str[:-2 - n_indent] + ']'
    print(s_str)

    f_str = 'all_wanted_f = ['
    n_indent = len(f_str)
    for w_f in all_wanted_f:
        f_str += "{:d},\n".format(w_f) + ' ' * n_indent
    f_str = f_str[:-2 - n_indent] + ']'
    print(f_str)

    wanted_inds = []
    for seq, f_i in zip(all_wanted_s, all_wanted_f):

        if (f_i + 4 >= frame_inds.shape[0]):
            raise ValueError('Error: Asking frame number {:d} for sequence {:s}, with only {:d} frames'.format(f_i, seq, frame_inds.shape[0]))

        s_i = np.argwhere(test_dataset.sequences == seq)[0][0]
        mask = np.logical_and(seq_inds == s_i, frame_inds == f_i)
        w_i = np.argwhere(mask)[0][0]
        if (adding_extra):
            wanted_inds += [w_i - 4, w_i, w_i + 4]
        else:
            wanted_inds += [w_i]

    return wanted_inds


def comparison_metrics_multival(list_of_paths, list_of_names, sorted_val_days, dataset_paths, plt_chkp=-1):

    ############
    # Parameters
    ############

    if list_of_names is None:
        list_of_names = [str(i) for i in range(len(list_of_paths))]

    dataset_folders = [f.split('/')[-1] for f in dataset_paths]

    # Set which gpu is going to be used (auto for automatic choice)
    GPU_ID = 'auto'

    comparison_TP_FP_FN = []
    comparison_MSE = []
    all_chkps = []
    horizons = []
    n_2D_layers = []

    for i_chosen_log, chosen_log in enumerate(list_of_paths):

        ########
        # Config
        ########

        # Load parameters
        config = Config()
        config.load(chosen_log)
        n_2D_layers.append(config.n_2D_layers)
        horizons.append(config.T_2D)

        # Change parameters for the test here. For example, you can stop augmenting the input data.
        config.augment_noise = 0
        config.augment_scale_min = 1.0
        config.augment_scale_max = 1.0
        config.augment_symmetries = [False, False, False]
        config.augment_rotation = 'none'
        config.validation_size = 1000

        ############
        # Parameters
        ############

        # Find all checkpoints in the chosen training folder
        chkp_path = join(chosen_log, 'checkpoints')
        chkps = np.sort([join(chkp_path, f) for f in listdir(chkp_path) if f[:4] == 'chkp'])

        # Find which chkp we want to use
        if plt_chkp < -1:
            chosen_chkp_i = [i for i in range(len(chkps))]
        elif plt_chkp < 0:
            chosen_chkp_i = -1
        else:
            chkp_inds = np.array([int(f[:-4].split('_')[-1]) for f in chkps])
            chosen_chkp_i = np.argmin(np.abs(chkp_inds - plt_chkp))

        # Reduce checkpoint list to the wanted one
        chkps = chkps[np.array([chosen_chkp_i])]

        # Save checkpoints
        all_chkps.append(chkps)

        # Get the chkp_inds
        chkp_inds = [int(f[:-4].split('_')[-1]) for f in chkps]

        # # Get training and validation days
        # val_path = join(chosen_log, 'val_preds')
        # val_days = np.unique([f[:19] for f in listdir(val_path) if f.endswith('pots.ply')])

        # Util ops
        sigmoid_2D = torch.nn.Sigmoid()
        fake_loss = FakeColliderLoss(config)

        # Result folder
        visu_path = join(config.saving_path, 'test_metrics')
        if not exists(visu_path):
            makedirs(visu_path)

        ####################################
        # Preload to avoid long computations
        ####################################

        # Dataset
        test_dataset = MultiCollisionDataset(config,
                                             sorted_val_days,
                                             chosen_set='validation',
                                             dataset_paths=dataset_paths,
                                             simulated=['Simulation' in dataset_path for dataset_path in dataset_paths],
                                             balance_classes=False,)

        # List all precomputed preds (each ckp that has all val folder computed)
        saved_res = np.sort([f for f in listdir(visu_path) if f.startswith('metrics_val') and f.endswith('.pkl')])
        saved_res_chkps = [int(f[:-4].split('_')[-1]) for f in saved_res]
        for data_folder in dataset_folders:
            saved_res_chkps = np.intersect1d(saved_res_chkps, [int(f[:-4].split('_')[-1]) for f in saved_res if data_folder in f])

        # # REDO
        # if i_chosen_log >= len(list_of_paths) - 20:
        #     saved_res_inds = []

        # List of the chkp to do
        todo_inds = [ind for ind in chkp_inds if ind not in saved_res_chkps]
        to_load_inds = [ind for ind in chkp_inds if ind in saved_res_chkps]

        # Results
        all_TP_FP_FN = [[None for ind in chkp_inds] for _ in dataset_folders]
        all_MSE = [[None for ind in chkp_inds] for _ in dataset_folders]

        # Load if available
        if len(to_load_inds) > 0:

            print('\nFound previous predictions, loading them')

            for chkp_i, chkp in enumerate(chkps):

                if chkp_inds[chkp_i] not in to_load_inds:
                    continue

                for data_i, d_folder in enumerate(dataset_folders):

                    # Load preds for this chkp
                    chkp_stat_file = join(visu_path, 'metrics_val_{:s}_chkp_{:04d}.pkl'.format(d_folder, chkp_inds[chkp_i]))
                    with open(chkp_stat_file, 'rb') as rfile:
                        chkp_TP_FP_FN, chkp_MSE = pickle.load(rfile)

                    # Store all predictions
                    all_TP_FP_FN[data_i][chkp_i] = chkp_TP_FP_FN
                    all_MSE[data_i][chkp_i] = chkp_MSE

        ########
        # Or ...
        ########

        if len(todo_inds) > 0:

            ############
            # Choose GPU
            ############
            
            torch.cuda.empty_cache()

            # Automatic choice (need pynvml to be installed)
            if GPU_ID == 'auto':
                print('\nSearching a free GPU:')
                for i in range(torch.cuda.device_count()):
                    a = torch.cuda.list_gpu_processes(i)
                    print(torch.cuda.list_gpu_processes(i))
                    a = a.split()
                    if a[1] == 'no':
                        GPU_ID = a[0][-1:]

            # Safe check no free GPU
            if GPU_ID == 'auto':
                print('\nNo free GPU found!\n')
                a = 1 / 0

            else:
                print('\nUsing GPU:', GPU_ID, '\n')

            # Set GPU visible device
            chosen_gpu = int(GPU_ID)

            ###########################
            # Initialize model and data
            ###########################

            # Specific sampler with pred inds
            test_sampler = MultiCollisionSamplerTest(test_dataset, test_ratio=0.1)
            test_loader = DataLoader(test_dataset,
                                     batch_size=1,
                                     sampler=test_sampler,
                                     collate_fn=MultiCollisionCollate,
                                     num_workers=config.input_threads,
                                     pin_memory=True)

            # Calibrate samplers
            if config.max_val_points < 0:
                config.max_val_points = 1e9
                test_loader.dataset.max_in_p = 1e9
                test_sampler.calib_max_in(config, test_loader, untouched_ratio=0.95, verbose=True)
            test_sampler.calibration(test_loader, verbose=True)

            # Init model
            net = KPCollider(config, test_dataset.label_values, test_dataset.ignored_labels)

            # Choose to train on CPU or GPU
            if torch.cuda.is_available():
                device = torch.device("cuda:{:d}".format(chosen_gpu))
                net.to(device)
            else:
                device = torch.device("cpu")
            
            torch.cuda.synchronize(device)

            ######################################
            # Start predictions with ckpts weights
            ######################################


            for chkp_i, chkp in enumerate(chkps):
                
                print('Starting Deep predictions')
                t0 = time.time()

                if chkp_inds[chkp_i] not in todo_inds:
                    continue

                # Load new checkpoint weights
                if torch.cuda.is_available():
                    checkpoint = torch.load(chkp, map_location=device)
                else:
                    checkpoint = torch.load(chkp, map_location=torch.device('cpu'))
                net.load_state_dict(checkpoint['model_state_dict'])
                epoch_i = checkpoint['epoch'] + 1
                net.eval()
                print("\nModel and training state restored from " + chkp)

                # Results storage container
                PR_resolution = 100
                chkp_TP_FP_FN = []
                chkp_MSE = []
                chkp_done = []
                for s_ind, seq_frames in enumerate(test_dataset.frames):
                    chkp_TP_FP_FN.append(np.zeros((len(seq_frames), config.n_2D_layers, PR_resolution, 3), dtype=np.int32))
                    chkp_MSE.append(np.zeros((len(seq_frames), config.n_2D_layers), dtype=np.float32))
                    chkp_done.append(np.zeros((len(seq_frames)), dtype=bool))

                        
                # No gradient computation here
                with torch.no_grad():

                    # Predict wanted inds with this chkp
                    last_count = -1
                    for i, batch in enumerate(test_loader):

                        if 'cuda' in device.type:
                            batch.to(device)

                        # Forward pass
                        outputs, preds_init_2D, preds_2D = net(batch, config)

                        # Get probs and labels
                        f_inds = batch.frame_inds.cpu().numpy()
                        lengths = batch.lengths[0].cpu().numpy()
                        stck_init_preds = sigmoid_2D(preds_init_2D).cpu().detach().numpy()
                        stck_future_logits = preds_2D.cpu().detach().numpy()
                        stck_future_preds = sigmoid_2D(preds_2D).cpu().detach().numpy()
                        stck_future_gts = batch.future_2D.cpu().detach().numpy()
                        torch.cuda.synchronize(device)

                        # Loop on batch
                        i0 = 0
                        for b_i, length in enumerate(lengths):

                            # Get the 2D predictions and gt (init_2D)
                            img0 = stck_init_preds[b_i, 0, :, :, :]
                            gt_im0 = np.copy(stck_future_gts[b_i, config.n_frames - 1, :, :, :])
                            gt_im1 = np.copy(stck_future_gts[b_i, config.n_frames - 1, :, :, :])
                            gt_im1[:, :, 2] = np.max(stck_future_gts[b_i, :, :, :, 2], axis=0)
                            img1 = stck_init_preds[b_i, 1, :, :, :]
                            s_ind = f_inds[b_i, 0]
                            f_ind = f_inds[b_i, 1]

                            # Get the 2D predictions and gt (prop_2D)
                            img = stck_future_preds[b_i, :, :, :, :]
                            gt_im = stck_future_gts[b_i, config.n_frames:, :, :, :]
                            
                            # Get the input frames gt
                            ingt_im = stck_future_gts[b_i, :config.n_frames, :, :, :]

                            # # Future errors defined the same as the loss
                            future_errors_bce = fake_loss.apply(gt_im, stck_future_logits[b_i, :, :, :, :], error='bce')
                            # future_errors = fake_loss.apply(gt_im, stck_future_logits[b_i, :, :, :, :], error='linear')
                            # future_errors = np.concatenate((future_errors_bce, future_errors), axis=0)

                            # # Save prediction too in gif format
                            # s_ind = f_inds[b_i, 0]
                            # f_ind = f_inds[b_i, 1]
                            # filename = '{:s}_{:07d}_e{:04d}.npy'.format(test_dataset.sequences[s_ind], f_ind, epoch_i)
                            # gifpath = join(config.saving_path, 'test_visu', filename)
                            # fast_save_future_anim(gifpath[:-4] + '_f_gt.gif', gt_im, zoom=5, correction=True)
                            # fast_save_future_anim(gifpath[:-4] + '_f_pre.gif', img, zoom=5, correction=True)
                            
                            # Get this image of dynamic obstacles [T, H, W]
                            gx = stck_future_gts[b_i, config.n_frames:, :, :, 2] > 0.01
                            x = stck_future_preds[b_i, :, :, :, 2]

                            gt_flat = np.reshape(gx, (config.n_2D_layers, -1))
                            p_flat = np.reshape(x, (config.n_2D_layers, -1))

                            # Get the result metrics [T, n_thresh, 3]
                            t00 = time.time()
                            res_TP_FP_FN = fast_threshold_stats(gt_flat, p_flat, n_thresh=PR_resolution)
                            t11 = time.time()
                            if (t11 - t00 > 1.0):
                                print('WARNING: SLOW TEST', gt_flat.shape, p_flat.shape, PR_resolution, t11 - t00)

                            # Get the mse result [T]
                            res_MSE = np.mean(np.square(x - gx.astype(np.float32)), axis=(1, 2))

                            # Store result in container [seqs][frames, T, n_thresh, 3]
                            chkp_TP_FP_FN[s_ind][f_ind, :, :, :] = res_TP_FP_FN
                            chkp_MSE[s_ind][f_ind, :] = res_MSE
                            chkp_done[s_ind][f_ind] = True

                        # print([np.sum(c_done.astype(np.int32)) / np.prod(c_done.shape) for c_done in chkp_done])

                        count = np.sum([np.sum(c_done.astype(np.int32)) for c_done in chkp_done])
                        if count == last_count:
                            break
                        else:
                            last_count = count

                # Store all predictions (per validation)
                for data_i, d_folder in enumerate(dataset_folders):

                    val_TP_FP_FN = [tpfpfn for s_i, tpfpfn in enumerate(chkp_TP_FP_FN) if d_folder in test_dataset.seq_path[s_i]]
                    val_MSE = [mse for s_i, mse in enumerate(chkp_MSE) if d_folder in test_dataset.seq_path[s_i]]
                    val_TP_FP_FN = np.concatenate(val_TP_FP_FN, axis=0)
                    val_MSE = np.concatenate(val_MSE, axis=0)

                    # Do not save useless data
                    is_done = np.sum(np.abs(val_MSE), axis=1) > 1e-9
                    all_TP_FP_FN[data_i][chkp_i] = val_TP_FP_FN[is_done]
                    all_MSE[data_i][chkp_i] = val_MSE[is_done]


                t1 = time.time()
                print('Done in {:.1f}s\n'.format(t1 - t0))

                print('Saving to files')
                t0 = time.time()

                # Save preds for this chkp
                for data_i, d_folder in enumerate(dataset_folders):
                    chkp_stat_file = join(visu_path, 'metrics_val_{:s}_chkp_{:04d}.pkl'.format(d_folder, chkp_inds[chkp_i]))
                    with open(chkp_stat_file, 'wb') as wfile:
                        pickle.dump((np.copy(all_TP_FP_FN[data_i][chkp_i]), np.copy(all_MSE[data_i][chkp_i])), wfile)

                t1 = time.time()
                print('Done in {:.1f}s\n'.format(t1 - t0))

            # Free cuda memory
            torch.cuda.empty_cache()

        # All TP_FP_FN shape: [val_n][chkp_n, frames_n, T, nt, 3]
        all_TP_FP_FN = [np.stack(val_TP_FP_FN, axis=0) for val_TP_FP_FN in all_TP_FP_FN]
        all_MSE = [np.stack(val_MSE, axis=0) for val_MSE in all_MSE]
        comparison_TP_FP_FN.append(all_TP_FP_FN)
        comparison_MSE.append(all_MSE)

    ################
    # Visualizations
    ################
    
    wanted_s_inds = [s_f_inds[0] for s_f_inds in test_dataset.all_inds]
    wanted_f_inds = [s_f_inds[1] for s_f_inds in test_dataset.all_inds]
    is_sim = np.array([test_dataset.sim_sequences[s_ind] for s_ind in wanted_s_inds])


    # Stack: comparison_preds shape: [log_n, frames_n, T_disp, H, W, 3]
    dts = [t / n for t, n in zip(horizons, n_2D_layers)]
    log_times = []
    for log_i, mse in enumerate(comparison_MSE):
        t = horizons[log_i]
        dt = dts[log_i]
        log_times.append(np.arange(dt, t + 0.1 * dt, dt))


    if True:
        
        print('\n')
        print('Table of results:')
        print('*****************\n')

            
        #######
        # Table
        #######

        metric_names = ['AP_1', 'AP_2', 'AP_3', 'AP_tot']
        metric_names = ['mAP_1', 'mAP_2', 'mAP_3', 'mAP_tot']
        # metric_names = ['MSE_1', 'MSE_2', 'MSE_3', 'mMSE']

        # Get width of first and next columns
        n_fmt0 = np.max([len(log_name) for log_name in list_of_names]) + 2
        n_fmt1 = 9
        n_fmt2 = n_fmt1 * len(metric_names)

        s = '{:^{width}s} '.format(' ', width=n_fmt0)
        for i, data_folder in enumerate(dataset_folders):
            s += '{:^{width}s}  '.format(data_folder, width=n_fmt2)
        print(s)

        s = '{:^{width}s} '.format(' ', width=n_fmt0)
        for i, _ in enumerate(dataset_folders):
            for m_name in metric_names:
                s += '{:^{width}s}'.format(m_name, width=n_fmt1)
            s += '  '
        print(s)

        # Plot last PR curve for each log
        for i, name in enumerate(list_of_names):

            # TP_FP_FN per frames [val_n][frames_n, T, nt, 3]
            all_TP_FP_FN = [comp_TP_FP_FN[-1] for comp_TP_FP_FN in comparison_TP_FP_FN[i]]

            # Init x-axis values
            times = log_times[i]

            # MSE [T]
            all_MSE = [comp_MSE[-1] for comp_MSE in comparison_MSE[i]]
            all_MSE = np.stack([np.mean(mse, axis=0) for mse in all_MSE], axis=0)
            
            s = ''
            for ax_i, (chosen_TP_FP_FN, MSE) in enumerate(zip(all_TP_FP_FN, all_MSE)):

                
                if 'mAP_1' in metric_names:

                    # [nt, frames, T]
                    chosen_TP_FP_FN = np.transpose(chosen_TP_FP_FN, (2, 0, 1, 3))
                    tps = chosen_TP_FP_FN[..., 0]
                    fps = chosen_TP_FP_FN[..., 1]
                    fns = chosen_TP_FP_FN[..., 2]
                    pre = tps / (fns + tps + 1e-6)
                    rec = tps / (fps + tps + 1e-6)
                    f1s = 2 * tps / (2 * tps + fps + fns + 1e-6)
                    
                    # Reverse for AP
                    rec = rec[::-1]
                    pre = pre[::-1]

                    # Correct last recs that become zero
                    end_rec = rec[-10:]
                    end_rec[end_rec < 0.01] = 1.0
                    pre = np.vstack((np.ones_like(pre[:1]), pre))
                    rec = np.vstack((np.zeros_like(rec[:1]), rec))

                    # Average precision as computed by scikit [frames, T]
                    AP = np.sum((rec[1:] - rec[:-1]) * pre[1:], axis=0)

                    # Mean to get to [T]
                    AP = np.mean(AP, axis=0)

                    # AP on full SOGMs [frames]
                    full_TP_FP_FN = np.sum(chosen_TP_FP_FN, axis=2)
                    tps = full_TP_FP_FN[..., 0]
                    fps = full_TP_FP_FN[..., 1]
                    fns = full_TP_FP_FN[..., 2]
                    pre = tps / (fns + tps + 1e-6)
                    rec = tps / (fps + tps + 1e-6)
                    f1s = 2 * tps / (2 * tps + fps + fns + 1e-6)
                    rec = rec[::-1]
                    pre = pre[::-1]
                    AP_tot = np.sum((rec[1:] - rec[:-1]) * pre[1:], axis=0)
                    AP_tot = np.mean(AP_tot)

                    
                else:
                    
                    # All stats from real and sim sequences combined [nt, T, 3]
                    combined_TP_FP_FN = np.sum(chosen_TP_FP_FN, axis=0)
                    combined_TP_FP_FN = np.transpose(combined_TP_FP_FN, (1, 0, 2))
                    
                    # [nt, T]
                    tps = combined_TP_FP_FN[..., 0]
                    fps = combined_TP_FP_FN[..., 1]
                    fns = combined_TP_FP_FN[..., 2]

                    pre = tps / (fns + tps + 1e-6)
                    rec = tps / (fps + tps + 1e-6)
                    f1s = 2 * tps / (2 * tps + fps + fns + 1e-6)

                    best_mean = np.argmax(np.mean(f1s, axis=1))
                    best_last = np.argmax(np.mean(f1s[:, -10:], axis=1))

                    rec = rec[::-1]
                    pre = pre[::-1]
                    
                    # Correct last recs that become zero
                    end_rec = rec[-10:, :]
                    end_rec[end_rec < 0.01] = 1.0
                    pre = np.vstack((np.ones_like(pre[:1]), pre))
                    rec = np.vstack((np.zeros_like(rec[:1]), rec))

                    # Average precision as computed by scikit [T]
                    AP = np.sum((rec[1:] - rec[:-1]) * pre[1:], axis=0)
                    
                    # AP on full SOGMs [1]
                    full_TP_FP_FN = np.sum(combined_TP_FP_FN, axis=1)
                    tps = full_TP_FP_FN[..., 0]
                    fps = full_TP_FP_FN[..., 1]
                    fns = full_TP_FP_FN[..., 2]
                    pre = tps / (fns + tps + 1e-6)
                    rec = tps / (fps + tps + 1e-6)
                    f1s = 2 * tps / (2 * tps + fps + fns + 1e-6)
                    rec = rec[..., ::-1]
                    pre = pre[..., ::-1]

                    # Correct last recs that become zero
                    end_rec = rec[-10:]
                    end_rec[end_rec < 0.01] = 1.0
                    pre = np.concatenate((np.ones_like(pre[:1]), pre), axis=0)
                    rec = np.concatenate((np.zeros_like(rec[:1]), rec), axis=0)

                    AP_tot = np.sum((rec[..., 1:] - rec[..., :-1]) * pre[..., 1:], axis=-1)



                if ax_i == 0:
                    s += '{:^{width}s} '.format(name, width=n_fmt0)
                else:
                    s += '  '

                ind1 = np.argmin(np.abs(times - 0.0))
                ind2 = np.argmin(np.abs(times - 2.0))
                ind3 = np.argmin(np.abs(times - 3.0))
                ind4 = np.argmin(np.abs(times - np.min(horizons)))

                if 'AP_1' in metric_names or 'mAP_1' in metric_names:
                    s += '{:^{width}.1f}'.format(100*AP[ind1], width=n_fmt1)
                    s += '{:^{width}.1f}'.format(100*AP[ind2], width=n_fmt1)
                    s += '{:^{width}.1f}'.format(100*AP[ind3], width=n_fmt1)
                    s += '{:^{width}.1f}'.format(100*np.mean(AP[:ind4]), width=n_fmt1)
                    # s += '{:^{width}.1f}'.format(100*AP_tot, width=n_fmt1)

                else:
                    s += '{:^{width}.2f}'.format(1000*MSE[ind1], width=n_fmt1)
                    s += '{:^{width}.2f}'.format(1000*MSE[ind2], width=n_fmt1)
                    s += '{:^{width}.2f}'.format(1000*MSE[ind3], width=n_fmt1)
                    s += '{:^{width}.2f}'.format(1000*np.mean(MSE[:ind4]), width=n_fmt1)

            print(s)

        # #######
        # # Plot
        # #######

        # # Figure
        # n_val = len(dataset_folders)
        # figC, axesC = plt.subplots(1, n_val, figsize=(9, 4))

        # # Plot last PR curve for each log
        # for i, name in enumerate(list_of_names):

        #     # [val_n][frames_n, T, nt, 3]
        #     all_TP_FP_FN = [comp_TP_FP_FN[-1] for comp_TP_FP_FN in comparison_TP_FP_FN[i]]

        #     # Init x-axis values
        #     times = log_times[i]
            
        #     # All stats from real and sim sequences [val_n, T, nt, 3]
        #     all_TP_FP_FN = np.stack([np.sum(tpfpfn, axis=0) for tpfpfn in all_TP_FP_FN], axis=0)

        #     # Chosen timestamps [val_n, nt, T, 3]
        #     all_TP_FP_FN = np.transpose(all_TP_FP_FN, (0, 2, 1, 3))

        #     # MSE [T]
        #     all_MSE = [comp_MSE[-1] for comp_MSE in comparison_MSE[i]]
        #     all_MSE = np.stack([np.mean(mse, axis=0) for mse in all_MSE], axis=0)
            
        #     s = ''
        #     for ax_i, (chosen_TP_FP_FN, MSE, ax) in enumerate(zip(all_TP_FP_FN, all_MSE, axesC)):
        #         tps = chosen_TP_FP_FN[..., 0]
        #         fps = chosen_TP_FP_FN[..., 1]
        #         fns = chosen_TP_FP_FN[..., 2]


        #         pre = tps / (fns + tps + 1e-6)
        #         rec = tps / (fps + tps + 1e-6)
        #         f1s = 2 * tps / (2 * tps + fps + fns + 1e-6)

        #         rec = rec[::-1]
        #         pre = pre[::-1]

        #         # Correct last recs that become zero
        #         end_rec = rec[-10:, :]
        #         end_rec[end_rec < 0.01] = 1.0
        #         pre = np.vstack((np.ones_like(pre[:1]), pre))
        #         rec = np.vstack((np.zeros_like(rec[:1]), rec))

        #         # Average precision as computed by scikit
        #         AP = np.sum((rec[1:] - rec[:-1]) * pre[1:], axis=0)

        #         ax.plot(times[:-1], AP[:-1], linewidth=1, label=name)
        #         #ax.plot(times, f1s[best_mean], linewidth=1, label=name)
                
        # for ax in axesC:
        #     ax.grid(linestyle='-.', which='both')
        #     # axA.set_ylim(0, 1)
        #     ax.set_xlabel('Time Layer in SOGM (sec)')
        #     ax.set_ylabel('AP')

        # # Display legends and title
        # plt.legend()

        # fname = 'results/AP_fig.pdf'
        # plt.savefig(fname,
        #             bbox_inches='tight')


    plt.show()

    return


def comparison_gifs(list_of_paths, list_of_names, sorted_val_days, dataset_paths, wanted_inds=[], wanted_chkp=[], redo=False):

    ############
    # Parameters
    ############

    # Set which gpu is going to be used (auto for automatic choice)
    GPU_ID = 'auto'

    if not wanted_inds:
        # For flowfollowers 1200 1400
        wanted_inds = [1200, 1400, 1500, 700, 800, 900]   # Bouncers
        # wanted_inds = [1300, 700, 1400, 1500, 100, 150, 800, 900]  # Wanderers
        #wanted_inds = [1200, 1400, 1500, 100, 150, 700, 800, 900]   # FlowFollowers
        #wanted_inds = [i for i in range(2850, 2900, 10)]   # FlowFollowersbis


    comparison_gts = []
    comparison_ingts = []
    comparison_preds = []
    visu_paths = []
    horizons = []
    n_2D_layers = []
    
    with torch.no_grad():
        for chosen_log, log_name in zip(list_of_paths, list_of_names):

            ############
            # Parameters
            ############

            # Load parameters
            config = Config()
            config.load(chosen_log)
            n_2D_layers.append(config.n_2D_layers)
            horizons.append(config.T_2D)
            

            ##################################
            # Change model parameters for test
            ##################################

            # Change parameters for the test here. For example, you can stop augmenting the input data.
            config.augment_noise = 0
            config.augment_scale_min = 1.0
            config.augment_scale_max = 1.0
            config.augment_symmetries = [False, False, False]
            config.augment_rotation = 'none'
            config.validation_size = 5000

            # Find all checkpoints in the chosen training folder
            chkp_path = join(chosen_log, 'checkpoints')
            chkps = np.sort([join(chkp_path, f) for f in listdir(chkp_path) if f[:4] == 'chkp'])
            
            wanted_chkp_mod = [w_c % len(chkps) for w_c in wanted_chkp]

            # # Get training and validation days
            # val_path = join(chosen_log, 'val_preds')
            # val_days = np.unique([f[:19] for f in listdir(val_path) if f.endswith('pots.ply')])

            # Util ops
            softmax = torch.nn.Softmax(1)
            sigmoid_2D = torch.nn.Sigmoid()
            fake_loss = FakeColliderLoss(config)

            # Result folder
            visu_path = join(config.saving_path, 'test_visu')
            if not exists(visu_path):
                makedirs(visu_path)
            visu_paths.append(visu_path)

            ####################################
            # Preload to avoid long computations
            ####################################

            # Dataset
            test_dataset = MultiCollisionDataset(config,
                                                 sorted_val_days,
                                                 chosen_set='validation',
                                                 dataset_paths=dataset_paths,
                                                 simulated=['Simulation' in dataset_path for dataset_path in dataset_paths],
                                                 balance_classes=False,)

            wanted_s_inds = [test_dataset.all_inds[ind][0] for ind in wanted_inds]
            wanted_f_inds = [test_dataset.all_inds[ind][1] for ind in wanted_inds]
            sf_to_i = {tuple(test_dataset.all_inds[ind]): i for i, ind in enumerate(wanted_inds)}

            # List all files we need per checkpoint
            all_chkp_files = []
            for chkp_i, chkp in enumerate(chkps):
                if chkp_i in wanted_chkp_mod:

                    # Check if all wanted_files exists for this chkp
                    all_wanted_files = []
                    preds_chkp_folder = join(visu_path, 'preds_{:s}'.format(chkp[:-4].split('/')[-1]))
                    for ind_i, ind in enumerate(wanted_inds):
                        seq_folder = join(preds_chkp_folder, test_dataset.sequences[wanted_s_inds[ind_i]])
                        wanted_ind_file = join(seq_folder, 'f_{:06d}.pkl'.format(wanted_f_inds[ind_i]))
                        all_wanted_files.append(wanted_ind_file)

                    all_chkp_files.append(all_wanted_files)

            # GT files
            all_gt_files = []
            gt_folder = join(visu_path, 'gt_imgs')
            for ind_i, ind in enumerate(wanted_inds):
                seq_folder = join(gt_folder, test_dataset.sequences[wanted_s_inds[ind_i]])
                wanted_ind_file = join(seq_folder, 'f_{:06d}.pkl'.format(wanted_f_inds[ind_i]))
                all_gt_files.append(wanted_ind_file)

            # Find which chkp need to be redone [n_chkp, n_w_inds]
            saved_mask = np.array([[exists(wanted_ind_file) for wanted_ind_file in all_wanted_files] for all_wanted_files in all_chkp_files], dtype=bool)
            chkp_completed = np.all(saved_mask, axis=1)
            if redo:
                chkp_completed = np.zeros_like(chkp_completed)

            ####################
            print('\n')
            title = '{:s} ({:s})'.format(log_name, chosen_log)
            print(title)
            print('*' * len(title))
            print('\n')

            n_fmt0 = np.max([len(chkp[:-4].split('/')[-1]) for chkp in chkps]) + 2
            lines = ['{:^{width}s}|'.format('Chkp', width=n_fmt0)]
            lines += ['{:-^{width}s}|'.format('', width=n_fmt0)]
            for chkp_i, chkp in enumerate(chkps):
                lines += ['{:^{width}s}|'.format(chkp[:-4].split('/')[-1], width=n_fmt0)]

            n_fmt1 = 2
            lines[0] += '{:^{width}s}'.format('Wanted Gifs already Computed?', width=n_fmt1 * len(saved_mask[0]))
            for s_i in range(saved_mask.shape[1]):
                lines[1] += '{:-^{width}s}'.format('', width=n_fmt1)
                save_i = 0
                for chkp_i, chkp in enumerate(chkps):
                    if chkp_i in wanted_chkp_mod:
                        if saved_mask[save_i, s_i]:
                            lines[chkp_i+2] += '{:}{:>{width}s}{:}'.format(bcolors.OKGREEN, u'\u2713', bcolors.ENDC, width=n_fmt1)
                        else:
                            lines[chkp_i+2] += '{:}{:>{width}s}{:}'.format(bcolors.FAIL, u'\u2718', bcolors.ENDC, width=n_fmt1)
                        save_i += 1

            for line_str in lines:
                print(line_str)
            ####################

            # If everything is already done, we do not need to prepare GPU etc
            if np.all(chkp_completed):
                
                all_preds = []
                all_gts = []
                all_ingts = []

                # Load every file
                for chkp_i, chkp in enumerate(chkps):
                    if chkp_i in wanted_chkp_mod:
                        chkp_preds = []
                        for ind_i, wanted_ind_file in enumerate(all_chkp_files[len(all_preds)]):
                            with open(wanted_ind_file, 'rb') as wfile:
                                ind_preds = pickle.load(wfile)
                            chkp_preds.append(np.copy(ind_preds))
                        chkp_preds = np.stack(chkp_preds, axis=0)
                        all_preds.append(chkp_preds)

                # All predictions shape: [chkp_n, frames_n, T, H, W, 3]
                all_preds = np.stack(all_preds, axis=0)

                # All gts shape: [frames_n, T, H, W, 3]
                for ind_i, wanted_ind_file in enumerate(all_gt_files):
                    with open(wanted_ind_file, 'rb') as wfile:
                        ind_gts, ind_ingts = pickle.load(wfile)
                    all_gts.append(np.copy(ind_gts))
                    all_ingts.append(np.copy(ind_ingts))

                all_gts = np.stack(all_gts, axis=0)
                all_ingts = np.stack(all_ingts, axis=0)

            ########
            # Or ...
            ########

            else:

                ############
                # Choose GPU
                ############
                
                torch.cuda.empty_cache()

                # Automatic choice (need pynvml to be installed)
                if GPU_ID == 'auto':
                    print('\nSearching a free GPU:')
                    for i in range(torch.cuda.device_count()):
                        a = torch.cuda.list_gpu_processes(i)
                        print(torch.cuda.list_gpu_processes(i))
                        a = a.split()
                        if a[1] == 'no':
                            GPU_ID = a[0][-1:]

                # Safe check no free GPU
                if GPU_ID == 'auto':
                    print('\nNo free GPU found!\n')
                    a = 1 / 0

                else:
                    print('\nUsing GPU:', GPU_ID, '\n')

                # Set GPU visible device
                chosen_gpu = int(GPU_ID)

                ###########################
                # Initialize model and data
                ###########################

                # Specific sampler with pred inds
                test_sampler = MultiCollisionSamplerTest(test_dataset, wanted_frame_inds=wanted_inds)
                test_loader = DataLoader(test_dataset,
                                         batch_size=1,
                                         sampler=test_sampler,
                                         collate_fn=MultiCollisionCollate,
                                         num_workers=config.input_threads,
                                         pin_memory=True)

                # Calibrate samplers
                if config.max_val_points < 0:
                    config.max_val_points = 1e9
                    test_loader.dataset.max_in_p = 1e9
                    test_sampler.calib_max_in(config, test_loader, untouched_ratio=0.95, verbose=True)
                test_sampler.calibration(test_loader, verbose=True)

                # Init model
                net = KPCollider(config, test_dataset.label_values, test_dataset.ignored_labels)

                # Choose to test on CPU or GPU
                if torch.cuda.is_available():
                    device = torch.device("cuda:{:d}".format(chosen_gpu))
                    net.to(device)
                else:
                    device = torch.device("cpu")
                
                torch.cuda.synchronize(device)

                ######################################
                # Start predictions with ckpts weights
                ######################################

                all_preds = []
                all_gts = [None for _ in wanted_inds]
                all_ingts = [None for _ in wanted_inds]

                for chkp_i, chkp in enumerate(chkps):
                    if chkp_i in wanted_chkp_mod:

                        if chkp_completed[len(all_preds)]:

                            # Load every file
                            chkp_preds = []
                            for ind_i, wanted_ind_file in enumerate(all_chkp_files[len(all_preds)]):
                                with open(wanted_ind_file, 'rb') as wfile:
                                    ind_preds = pickle.load(wfile)
                                chkp_preds.append(np.copy(ind_preds))

                            print("\nPrevious gifs found for " + chkp)

                        else:

                            # Load new checkpoint weights
                            if torch.cuda.is_available():
                                checkpoint = torch.load(chkp, map_location=device)
                            else:
                                checkpoint = torch.load(chkp, map_location=torch.device('cpu'))
                            net.load_state_dict(checkpoint['model_state_dict'])
                            epoch_i = checkpoint['epoch'] + 1
                            net.eval()
                            print("\nModel and training state restored from " + chkp)

                            chkp_preds = [None for _ in wanted_inds]

                            # Predict wanted inds with this chkp
                            for i, batch in enumerate(test_loader):

                                if 'cuda' in device.type:
                                    batch.to(device)

                                # Forward pass
                                outputs, preds_init_2D, preds_2D = net(batch, config)

                                # Get probs and labels
                                f_inds = batch.frame_inds.cpu().numpy()
                                lengths = batch.lengths[0].cpu().numpy()
                                stck_init_preds = sigmoid_2D(preds_init_2D).cpu().detach().numpy()
                                stck_future_logits = preds_2D.cpu().detach().numpy()
                                stck_future_preds = sigmoid_2D(preds_2D).cpu().detach().numpy()
                                stck_future_gts = batch.future_2D.cpu().detach().numpy()
                                torch.cuda.synchronize(device)

                                # Loop on batch
                                i0 = 0
                                for b_i, length in enumerate(lengths):

                                    # Get the 2D predictions and gt (init_2D)
                                    i_frame0 = config.n_frames - 1
                                    img0 = stck_init_preds[b_i, 0, :, :, :]
                                    gt_im0 = np.copy(stck_future_gts[b_i, i_frame0, :, :, :])
                                    gt_im1 = np.copy(stck_future_gts[b_i, i_frame0, :, :, :])
                                    gt_im1[:, :, 2] = np.max(stck_future_gts[b_i, i_frame0:, :, :, 2], axis=0)
                                    img1 = stck_init_preds[b_i, 1, :, :, :]
                                    s_ind = f_inds[b_i, 0]
                                    f_ind = f_inds[b_i, 1]

                                    # Get the 2D predictions and gt (prop_2D)
                                    img = stck_future_preds[b_i, :, :, :, :]
                                    gt_im = stck_future_gts[b_i, config.n_frames:, :, :, :]
                                    
                                    # Get the input frames gt
                                    ingt_im = stck_future_gts[b_i, :config.n_frames, :, :, :]

                                    # # Future errors defined the same as the loss
                                    future_errors_bce = fake_loss.apply(gt_im, stck_future_logits[b_i, :, :, :, :], error='bce')
                                    # future_errors = fake_loss.apply(gt_im, stck_future_logits[b_i, :, :, :, :], error='linear')
                                    # future_errors = np.concatenate((future_errors_bce, future_errors), axis=0)

                                    # # Save prediction too in gif format
                                    # s_ind = f_inds[b_i, 0]
                                    # f_ind = f_inds[b_i, 1]
                                    # filename = '{:s}_{:07d}_e{:04d}.npy'.format(test_dataset.sequences[s_ind], f_ind, epoch_i)
                                    # gifpath = join(config.saving_path, 'test_visu', filename)
                                    # fast_save_future_anim(gifpath[:-4] + '_f_gt.gif', gt_im, zoom=5, correction=True)
                                    # fast_save_future_anim(gifpath[:-4] + '_f_pre.gif', img, zoom=5, correction=True)

                                    # Store all predictions
                                    chkp_preds[sf_to_i[(s_ind, f_ind)]] = img
                                    all_gts[sf_to_i[(s_ind, f_ind)]] = gt_im
                                    all_ingts[sf_to_i[(s_ind, f_ind)]] = ingt_im

                                    if np.all([chkp_pred is not None for chkp_pred in chkp_preds]):
                                        break

                                if np.all([chkp_pred is not None for chkp_pred in chkp_preds]):
                                    break

                            # Save
                            preds_chkp_folder = join(visu_path, 'preds_{:s}'.format(chkp[:-4].split('/')[-1]))
                            for ind_i, ind in enumerate(wanted_inds):
                                seq_folder = join(preds_chkp_folder, test_dataset.sequences[wanted_s_inds[ind_i]])
                                if not exists(seq_folder):
                                    makedirs(seq_folder)
                                wanted_ind_file = join(seq_folder, 'f_{:06d}.pkl'.format(wanted_f_inds[ind_i]))
                                with open(wanted_ind_file, 'wb') as wfile:
                                    pickle.dump(np.copy(chkp_preds[ind_i]), wfile)

                            # Save GT
                            gt_folder = join(visu_path, 'gt_imgs')
                            for ind_i, ind in enumerate(wanted_inds):
                                seq_folder = join(gt_folder, test_dataset.sequences[wanted_s_inds[ind_i]])
                                if not exists(seq_folder):
                                    makedirs(seq_folder)
                                wanted_ind_file = join(seq_folder, 'f_{:06d}.pkl'.format(wanted_f_inds[ind_i]))
                                if not exists(wanted_ind_file):
                                    with open(wanted_ind_file, 'wb') as wfile:
                                        pickle.dump((np.copy(all_gts[ind_i]), np.copy(all_ingts[ind_i])),
                                                    wfile)                   

                        # Stack chkp predictions [frames_n, T, H, W, 3]
                        chkp_preds = np.stack([ccc for ccc in chkp_preds if ccc.ndim > 0], axis=0)

                        # Store all predictions
                        all_preds.append(np.copy(chkp_preds))


                # All predictions shape: [chkp_n, frames_n, T, H, W, 3]
                all_preds = np.stack(all_preds, axis=0)

                # All gts shape: [frames_n, T, H, W, 3]
                all_gts = np.stack(all_gts, axis=0)
                all_ingts = np.stack(all_ingts, axis=0)

            comparison_preds.append(np.copy(all_preds))
            comparison_gts.append(np.copy(all_gts))
            comparison_ingts.append(np.copy(all_ingts))

            # Free cuda memory
            torch.cuda.empty_cache()

    return comparison_preds, comparison_gts, comparison_ingts


def show_SRM_gifs(list_of_paths, list_of_names,
                  sorted_val_days, dataset_paths, wanted_inds,
                  comparison_preds, comparison_gts, comparison_ingts):

    horizons = []
    n_2D_layers = []
    for chosen_log, log_name in zip(list_of_paths, list_of_names):
        # Load parameters
        config = Config()
        config.load(list_of_paths[0])
        n_2D_layers.append(config.n_2D_layers)
        horizons.append(config.T_2D)


    # [log_n][chkp_n, frames_n, T, H, W, 3]
    # print(len(comparison_preds))
    # print(comparison_preds[0].shape)

    # Select one SOGM
    frame_i = 1
    sogm_visu = comparison_preds[0][0, frame_i]


    ###############
    # Old functions
    ###############

    # [T, H, W, 3]
    # print(sogm_visu.shape)
    # fig1, anim1 = show_local_maxima(sogm_visu[..., 2], neighborhood_size=5, threshold=0.1, show=False)
    # fig2, anim2 = show_risk_diffusion(sogm_visu, dl=0.12, diff_range=2.5, show=False)
    # plt.show()


    #############
    # Static SRMs
    #############

    # Diffuse the risk
    diffused_normal, _, _ = get_diffused_risk(config, torch.from_numpy(sogm_visu))
    diffused_no_t, _, _ = get_diffused_risk(config, torch.from_numpy(sogm_visu), dynamic_t_range=0.0001)
    diffused_no_norm, _, _ = get_diffused_risk(config, torch.from_numpy(sogm_visu), normalization=False)
    diffused_p1, _, _ = get_diffused_risk(config, torch.from_numpy(sogm_visu), norm_p=1)

    # Show static images
    statics = [0, 0, 0, 0]
    statics[0] = diffused_normal[0]
    statics[1] = diffused_no_t[0]
    statics[2] = diffused_no_norm[0]
    statics[3] = diffused_p1[0]

    zoom = 5
    fig, axes = plt.subplots(1, 4)
    for ax_i, static in enumerate(statics):

        im = np.squeeze(zoom_collisions(np.expand_dims(static, -1), zoom))
        im = SRM_colors(im, static=True)

        axes[ax_i].imshow(im)
        statics[ax_i] = im

    # Get the color map by name:
    cm = plt.get_cmap('viridis')
    imageio.imsave('results/static_normal.png', statics[0])
    imageio.imsave('results/static_no_norm.png', statics[2])
    imageio.imsave('results/static_p1.png', statics[3])


    ##############
    # Dynamic SRMs
    ##############

    # Show dynamic images
    dynamics = [0, 0, 0, 0]
    dyn_i = 5
    dynamics[0] = diffused_normal[dyn_i]
    dynamics[1] = diffused_no_t[dyn_i]
    dynamics[2] = diffused_no_norm[dyn_i]
    dynamics[3] = diffused_p1[dyn_i]

    fig, axes2 = plt.subplots(1, 4)
    for ax_i, dynamic in enumerate(dynamics):

        im = np.squeeze(zoom_collisions(np.expand_dims(dynamic, -1), zoom))
        im = SRM_colors(im)

        axes2[ax_i].imshow(im)
        dynamics[ax_i] = im

    # Get the color map by name:
    cm = plt.get_cmap('viridis')
    imageio.imsave('results/dynamic_normal.png', dynamics[0])
    imageio.imsave('results/dynamic_no_t.png', dynamics[1])

    sogm_im = np.expand_dims(zoom_collisions(sogm_visu, zoom), 0)
    sogm_im = superpose_gt_contour(sogm_im, sogm_im * 0, sogm_im * 0, no_in=True)


    fig, axes3 = plt.subplots(1, 1)
    axes3.imshow(sogm_im[0, dyn_i])
    imageio.imsave('results/sogm.png', sogm_im[0, dyn_i])

    plt.show()





    a = 1/0

    # Repeat gt for all checkpoints and merge with preds
    showed_gts = np.expand_dims(showed_gts, 0)
    showed_gts = np.tile(showed_gts, (showed_preds.shape[0], 1, 1, 1, 1))
    showed_ingts = np.expand_dims(showed_ingts, 0)
    showed_ingts = np.tile(showed_ingts, (showed_preds.shape[0], 1, 1, 1, 1))

    # Merge colors
    # merged_imgs = superpose_gt(showed_preds, showed_gts, showed_ingts)
    merged_imgs = superpose_gt_contour(showed_preds, showed_gts, showed_ingts, no_in=True)
    
    # # To show gt images
    # showed_preds = np.copy(showed_gts)
    # showed_preds[..., 2] *= 0.6
    # merged_imgs = superpose_gt(showed_preds, showed_gts * 0, showed_ingts, ingts_fade=(100, -5))

    # Reverse image height axis so that imshow is consistent with plot
    merged_imgs = merged_imgs[:, :, ::-1, :, :]






    # Show SOGM with RGB colrs

    # Show static risk






    return


def show_SOGM_gifs(list_of_paths, list_of_names,
                   sorted_val_days, dataset_paths, wanted_inds,
                   comparison_preds, comparison_gts, comparison_ingts):


    horizons = []
    n_2D_layers = []
    visu_paths = []
    for chosen_log, log_name in zip(list_of_paths, list_of_names):
        # Load parameters
        config = Config()
        config.load(list_of_paths[0])
        n_2D_layers.append(config.n_2D_layers)
        horizons.append(config.T_2D)
    
        # Result folder
        visu_path = join(config.saving_path, 'test_visu')
        visu_paths.append(visu_path)

    #############
    # Preparation
    #############

    # Dataset
    test_dataset = MultiCollisionDataset(config,
                                         sorted_val_days,
                                         chosen_set='validation',
                                         dataset_paths=dataset_paths,
                                         simulated=['Simulation' in dataset_path for dataset_path in dataset_paths],
                                         balance_classes=False,)
    
    wanted_s_inds = [test_dataset.all_inds[ind][0] for ind in wanted_inds]
    wanted_f_inds = [test_dataset.all_inds[ind][1] for ind in wanted_inds]


    ################
    # Visualizations
    ################

    # Choose a  checkpoint for each log
    chosen_chkp = -1
    comparison_preds = [cp[chosen_chkp] for cp in comparison_preds]

    # Preds do not have the same hozizon and frequency
    dts = [t / n for t, n in zip(horizons, n_2D_layers)]
    t_max = np.max(horizons)
    dt_max = np.min(dts)
    display_times = np.arange(dt_max, t_max + 0.1 * dt_max, dt_max) - 1e-6

    # Stack: comparison_preds shape: [log_n, frames_n, T_disp, H, W, 3]
    nT_max = np.max([cp.shape[-4] for cp in comparison_preds])
    new_preds = []
    for cp_i, cp in enumerate(comparison_preds):
        t = horizons[cp_i]
        dt = dts[cp_i]
        cp_times = np.arange(dt, t + 0.1 * dt, dt)
        interp_inds = [np.argmin(np.abs(cp_times - cp_t)) for cp_t in display_times]
        new_preds.append(np.copy(cp[:, interp_inds]))
    comparison_preds = np.stack(new_preds, axis=0)

    # # GT is the same for all chkp => [frames_n, T, H, W, 3]
    # comparison_gts = comparison_gts[0]
    # comparison_ingts = comparison_ingts[0]

    # Handle groundtruth too
    new_gts = []
    new_gts_v = []
    for gt_i, gt in enumerate(comparison_gts):
        t = horizons[gt_i]
        dt = dts[gt_i]
        gt_times = np.arange(dt, t + 0.1 * dt, dt)
        interp_inds = [np.argmin(np.abs(gt_times - gt_t)) for gt_t in display_times]
        interp_v = [np.min(np.abs(gt_times - gt_t)) for gt_t in display_times]
        new_gts.append(np.copy(gt[:, interp_inds]))
        new_gts_v.append(interp_v)

    # take the best interp value (which had the closest interpolator)
    new_gts = np.stack(new_gts, axis=0)
    new_gts_v = np.stack(new_gts_v, axis=0)
    inds00 = np.argmin(new_gts_v, axis=0)
    inds01 = np.arange(inds00.shape[0])
    comparison_gts = np.transpose(new_gts[inds00, :, inds01, :, :, :], axes=(1, 0, 2, 3, 4))
    
    # Input gt is always the same
    comparison_ingts = comparison_ingts[0]

    # # Multiply prediction to see small values
    # comparison_preds *= 2.0
    # comparison_preds = np.minimum(comparison_preds, 0.99)

    all_merged_imgs = []
    all_blob_imgs = []
    all_blob_gts = []
    # Advanced display
    N = len(wanted_inds)
    progress_n = 30
    fmt_str = '[{:<' + str(progress_n) + '}] {:5.1f}%'
    print('\nPreparing gifs')
    for frame_i, w_i in enumerate(wanted_inds):
    
        # Colorize and zoom both preds and gts
        zoom = 5
        showed_preds = zoom_collisions(comparison_preds[:, frame_i], zoom)
        showed_gts = zoom_collisions(comparison_gts[frame_i], zoom)
        showed_ingts = zoom_collisions(comparison_ingts[frame_i], zoom)

        # Repeat gt for all checkpoints and merge with preds
        showed_gts = np.expand_dims(showed_gts, 0)
        showed_gts = np.tile(showed_gts, (showed_preds.shape[0], 1, 1, 1, 1))
        showed_ingts = np.expand_dims(showed_ingts, 0)
        showed_ingts = np.tile(showed_ingts, (showed_preds.shape[0], 1, 1, 1, 1))

        # Merge colors
        merged_imgs = superpose_gt_contour(showed_preds, showed_gts, showed_ingts, no_in=True)

        # Merge Times and apply specific color
        blob_preds = np.max(showed_preds, axis=1, keepdims=True)
        blob_gts = np.max(showed_gts, axis=1, keepdims=True)
        blob_ingts = np.max(showed_ingts, axis=1, keepdims=True)
        blob_imgs = superpose_gt_contour(blob_preds, blob_gts, blob_ingts, no_in=True)
        
        # # To show gt images
        # showed_preds = np.copy(showed_gts)
        # showed_preds[..., 2] *= 0.6
        # merged_imgs = superpose_gt(showed_preds, showed_gts * 0, showed_ingts, ingts_fade=(100, -5))

        # Reverse image height axis so that imshow is consistent with plot
        merged_imgs = merged_imgs[:, :, ::-1, :, :]
        blob_imgs = blob_imgs[:, 0, ::-1, :, :]

        # => [log_n, H, W, 3]
        all_blob_imgs.append(blob_imgs)

        
        blob_gtims = superpose_gt_contour(blob_gts, blob_gts, blob_ingts, no_in=True, gt_im=True)
        blob_gtims = blob_gtims[0, 0, ::-1, :, :]
        all_blob_gts.append(blob_gtims)
        all_merged_imgs.append(merged_imgs)

        print('', end='\r')
        print(fmt_str.format('#' * (((frame_i + 1) * progress_n) // N), 100 * (frame_i + 1) / N), end='', flush=True)

    # Show a nice 100% progress bar
    print('', end='\r')
    print(fmt_str.format('#' * progress_n, 100), flush=True)
    print('\n')


    # Save the collection of images
    png_folder = join(test_dataset.path, 'sogm_preds', 'pngs')
    gif_folder = join(test_dataset.path, 'sogm_preds', 'gifs')
    if not exists(png_folder):
        makedirs(png_folder)
    if not exists(gif_folder):
        makedirs(gif_folder)
    for frame_i, w_i in enumerate(wanted_inds):     
        seq_name = test_dataset.sequences[wanted_s_inds[frame_i]]
        frame_name = test_dataset.frames[wanted_s_inds[frame_i]][wanted_f_inds[frame_i]]
        im_name = join(png_folder, '{:s}_{:s}.png'.format(seq_name, frame_name))
        imageio.imsave(im_name, all_blob_imgs[frame_i][-1])

        gif_name = join(gif_folder, '{:s}_{:s}.gif'.format(seq_name, frame_name))
        imageio.mimsave(gif_name, all_merged_imgs[frame_i][-1], fps=20)


    ###########
    # Selection
    ###########

    selec = ['2022-04-01_15-06-55_1648825749.127623',
             '2022-03-09_15-58-56_1646841621.120567',
             '2022-03-09_16-03-21_1646841889.915430',
             '2022-03-09_16-03-21_1646841916.315663',
             '2022-03-09_16-03-21_1646841949.116121',
             '2022-03-22_14-12-20_1647958437.008770',
             '2022-03-22_14-12-20_1647958444.607082',
             '2022-03-22_16-08-09_1647965337.589514',
             '2022-03-28_14-53-33_1648479336.715213',
             '2022-03-28_16-56-52_1648486619.915243',
             '2022-04-01_14-00-06_1648821611.208367',
             '2022-04-01_15-06-55_1648825618.530766',
             '2022-04-01_15-06-55_1648825625.127977',
             '2022-04-01_15-06-55_1648825694.629342',
             '2022-04-01_15-06-55_1648825713.329629']

    # Save the collection of images
    selec_folder = join(test_dataset.path, 'sogm_preds', 'selection')
    if not exists(selec_folder):
        makedirs(selec_folder)
    for frame_i, w_i in enumerate(wanted_inds):
        seq_name = test_dataset.sequences[wanted_s_inds[frame_i]]
        frame_name = test_dataset.frames[wanted_s_inds[frame_i]][wanted_f_inds[frame_i]]
        selec_name = '{:s}_{:s}'.format(seq_name, frame_name)
        if selec_name in selec:
            for log_i, log_name in enumerate(list_of_names):
                im_name = join(selec_folder, selec_name + '_{:d}_{:s}.png'.format(log_i, log_name))
                imageio.imsave(im_name, all_blob_imgs[frame_i][log_i])
            im_name = join(selec_folder, selec_name + '_{:d}_{:s}.png'.format(len(list_of_names), 'GT'))
            imageio.imsave(im_name, all_blob_gts[frame_i])
            


    return


def inspect_sogm_sessions(dataset_path, map_day, train_days, train_comments):

    
    # Threshold
    high_d = 2.0
    risky_d = 1.5
    collision_d = 0.6


    print('\n')
    print('------------------------------------------------------------------------------')
    print('\n')
    print('Start session inspection')
    print('************************')
    print('\nInitial map run:', map_day)
    print('\nInspected runs:')
    for d, day in enumerate(train_days):
        print(' >', day)
    print('')

    # Reduce number of runs to inspect
    print('You can choose to inspect only the last X runs (enter nothing to inspect all runs)')
    
    # n_runs = input("Enter the number X of runs to inspect:\n")
    n_runs = '6'

    if len(n_runs) > 0:
        n_runs = int(n_runs)
    else:
        n_runs = len(train_days)
    
    print('You choose to inspect only the last X runs')
    print(n_runs)

    if n_runs < len(train_days):
        train_days = train_days[-n_runs:]
        train_comments = train_comments[-n_runs:]

    config = MyhalCollisionConfig()

    # Initialize datasets (dummy validation)
    dataset = MyhalCollisionDataset(config,
                                    train_days,
                                    chosen_set='training',
                                    dataset_path=dataset_path,
                                    balance_classes=True)


    # convertion from labels to colors
    im_lim = config.radius_2D / np.sqrt(2)
    colormap = np.array([[209, 209, 209],
                        [122, 122, 122],
                        [255, 255, 0],
                        [0, 98, 255],
                        [255, 0, 0]], dtype=np.float32) / 255

    # We care about the dynamic class
    i_l = 4

    # Loading data for quantitative metrics
    all_times = []
    all_colli_mask = []
    all_risky_mask = []
    all_dists = []

    for s_ind, seq in enumerate(train_days):
        data = loading_session(dataset, s_ind, i_l, im_lim, colormap)
        all_pts, all_colors, all_labels, class_mask_opened, class_mask_eroded, seq_mask = data
        #    all_pts [frames][N, 3]
        # all_labels [frames][N,]

        min_dists = []
        for f_ind, pts in enumerate(all_pts):


            # Get distances on points that contain dynamic obstacles
            dyn_mask = np.sum(np.abs(all_colors[f_ind] - colormap[i_l]), axis=1) < 0.1


            # Manually correct some distances that are wrong
            if seq == '2022-06-01_18-15-07' and 1200 <= f_ind <= 1250:
                dyn_mask = np.logical_and(dyn_mask, pts[:, 0] < 0)

                
            if seq == '2022-06-01_20-36-03' and 774 <= f_ind <= 777:
                dyn_mask = np.logical_and(dyn_mask, pts[:, 1] < 0.5 * pts[:, 0])
            if seq == '2022-06-01_20-36-03' and 778 <= f_ind <= 800:
                dyn_mask = np.logical_and(dyn_mask, pts[:, 0] > 1.2)

            if np.any(dyn_mask):
                min_dists.append(np.min(np.linalg.norm(pts[dyn_mask, :2], axis=1)))
            else:
                min_dists.append(high_d)

        min_dists = np.array(min_dists, dtype=np.float32)
        min_dists = np.minimum(min_dists, high_d)

        # Manually correct some distances that are wrong
        if seq == '2022-06-01_18-23-28':
            tmp_dists = min_dists[233:265]
            error_mask = tmp_dists > 0.99 * high_d
            tmp_dists[error_mask] = np.min(tmp_dists)
            min_dists[233:265] = tmp_dists
            min_dists[1180:1215] = high_d
            
        if seq == '2022-06-01_18-20-40':
            min_dists[:130] = high_d
            
        if seq == '2022-06-01_20-36-03':
            tmp_dists = min_dists[417:435]
            error_mask = tmp_dists > 0.99 * high_d
            tmp_dists[error_mask] = np.min(tmp_dists)
            min_dists[417:435] = tmp_dists
            min_dists[1100:] = high_d
            

        
        # Threshold
        colli_mask = min_dists < collision_d
        risky_mask = min_dists < risky_d
        # risky_mask = np.logical_and(min_dists > collision_d, min_dists < risky_d)

        all_colli_mask.append(colli_mask)
        all_risky_mask.append(risky_mask)
        all_dists.append(min_dists)



    for s_ind, seq in enumerate(train_days):

        colli_mask = all_colli_mask[s_ind]
        risky_mask = all_risky_mask[s_ind]


        #########################
        # Min encounters distance
        #########################

        # Get blobs for encounters
        risky_mask_int = risky_mask.astype(np.int32)
        fronts = np.where(np.abs(risky_mask_int[1:] - risky_mask_int[:-1]) > 0)[0]
        splitted_dists = np.split(all_dists[s_ind], fronts + 1)
        splitted_dists = splitted_dists[1::2]

        # Get time to finish (from first encounter to last encounter)
        finish_time = float(dataset.frames[s_ind][fronts[-1]]) - float(dataset.frames[s_ind][fronts[0]])

        # Ge risk only during encounter times
        reduced_colli_mask = colli_mask[fronts[0]:fronts[-1]]
        reduced_risky_mask = risky_mask[fronts[0]:fronts[-1]]
        colli_index = np.sum(reduced_colli_mask.astype(np.int32)) / reduced_colli_mask.shape[0]
        risky_index = np.sum(reduced_risky_mask.astype(np.int32)) / reduced_risky_mask.shape[0]

        # Get encouter stats
        min_enc_dist = [np.min(sub_dist) for sub_dist in splitted_dists]
        enc_length = [len(sub_dist)/10 for sub_dist in splitted_dists]
        
        fmt_str = '{:s} | {:7.1f}% {:7.2f}% {:5.1f}s | {:2d} encounters: {:4.2f} m {:4.2f}s'
        print(fmt_str.format(seq,
                             100 * risky_index,
                             100 * colli_index,
                             finish_time,
                             len(splitted_dists),
                             np.mean(min_enc_dist),
                             np.mean(enc_length),
                             ))


    #########################
    # Create a display window
    #########################
    
    # Init data
    s_ind = 0
    data = loading_session(dataset, s_ind, i_l, im_lim, colormap)
    all_pts, all_colors, all_labels, class_mask_opened, class_mask_eroded, seq_mask = data



    figA, axA = plt.subplots(1, 1, figsize=(10, 7))
    plt.subplots_adjust(left=0.1, bottom=0.27, top=0.99)

    # Plot first frame of seq
    plotsA = [axA.scatter(all_pts[0][:, 0],
                          all_pts[0][:, 1],
                          s=2.0,
                          c=all_colors[0])]

    # Show a circle of the loop closure area
    axA.add_patch(patches.Circle((0, 0), radius=0.2,
                                 edgecolor=[0.2, 0.2, 0.2],
                                 facecolor=[1.0, 0.79, 0],
                                 fill=True,
                                 lw=1))

    # # Customize the graph
    # axA.grid(linestyle='-.', which='both')
    axA.set_xlim(-im_lim, im_lim)
    axA.set_ylim(-im_lim, im_lim)
    axA.set_aspect('equal', adjustable='box')

    # Make a horizontal slider to control the frequency.
    axcolor = 'lightgoldenrodyellow'
    axtime = plt.axes([0.1, 0.2, 0.8, 0.015], facecolor=axcolor)
    time_slider = Slider(ax=axtime,
                         label='ind',
                         valmin=0,
                         valmax=len(all_pts) - 1,
                         valinit=0,
                         valstep=1)

    # The function to be called anytime a slider's value changes
    def update_points(val):
        global f_i
        f_i = (int)(val)
        for plot_i, plot_obj in enumerate(plotsA):
            plot_obj.set_offsets(all_pts[f_i])
            plot_obj.set_color(all_colors[f_i])

    # register the update function with each slider
    time_slider.on_changed(update_points)

    # Ax with the presence of dynamic points
    class_mask = np.zeros_like(dataset.all_inds[:, 0], dtype=bool)
    class_mask[dataset.class_frames[i_l]] = True
    seq_mask = dataset.all_inds[:, 0] == s_ind
    seq_class_frames = class_mask[seq_mask]
    seq_class_frames = np.expand_dims(seq_class_frames, 0)
    axdyn0 = plt.axes([0.1, 0.18, 0.8, 0.015])
    axdyn0.imshow(seq_class_frames, cmap='GnBu', aspect='auto')
    axdyn0.set_axis_off()

    # # Ax with the presence of dynamic points at least 10
    # dyn_img = np.vstack(all_labels).T
    # dyn_img = dyn_img[-1:]
    # dyn_img[dyn_img > 10] = 10
    # dyn_img[dyn_img > 0] += 10
    # axdyn1 = plt.axes([0.1, 0.06, 0.8, 0.015])
    # axdyn1.imshow(dyn_img, cmap='OrRd', aspect='auto')
    # axdyn1.set_axis_off()

    # # Ax with opened
    # axdyn2 = plt.axes([0.1, 0.04, 0.8, 0.015])
    # axdyn2.imshow(class_mask_opened, cmap='OrRd', aspect='auto')
    # axdyn2.set_axis_off()

    # Ax with eroded
    axdyn1 = plt.axes([0.1, 0.16, 0.8, 0.015])
    axdyn1.imshow(class_mask_eroded, cmap='OrRd', aspect='auto')
    axdyn1.set_axis_off()

    axdyn2 = plt.axes([0.1, 0.14, 0.8, 0.015])
    axdyn2.imshow(np.expand_dims(all_risky_mask[s_ind], 0), cmap='Reds', aspect='auto')
    axdyn2.set_axis_off()

    axdyn3 = plt.axes([0.1, 0.12, 0.8, 0.015])
    axdyn3.imshow(np.expand_dims(all_colli_mask[s_ind], 0), cmap='Greys', aspect='auto')
    axdyn3.set_axis_off()

    axplotdist = plt.axes([0.1, 0.02, 0.8, 0.095])
    pltodist = axplotdist.plot(all_dists[s_ind])
    axplotdist.plot(all_dists[s_ind] * 0 + collision_d, 'k', linewidth=0.1)
    axplotdist.plot(all_dists[s_ind] * 0 + risky_d, 'r', linewidth=0.1)
    axplotdist.set_xlim(0, len(all_dists[s_ind] - 1))
    axplotdist.set_ylim(0.5, 2)
    axplotdist.set_axis_off()

    ###################
    # Saving function #
    ###################

    def onkey(event):
        global f_i

        # Save current as ptcloud
        if event.key in ['p', 'P']:
            print('Saving in progress')

            seq_name = dataset.sequences[s_ind]
            frame_name = dataset.frames[s_ind][f_i]
            sogm_folder = join(dataset.original_path, 'inspect_images')
            print(sogm_folder)
            if not exists(sogm_folder):
                makedirs(sogm_folder)

            # Save pointcloud
            H0 = dataset.poses[s_ind][f_i - 1]
            H1 = dataset.poses[s_ind][f_i]
            data = read_ply(join(dataset.seq_path[s_ind], frame_name + '.ply'))
            f_points = np.vstack((data['x'], data['y'], data['z'])).T
            f_ts = data['time']
            world_points = motion_rectified(f_points, f_ts, H0, H1)

            data = read_ply(join(dataset.annot_path[s_ind], frame_name + '.ply'))
            sem_labels = data['classif']

            ply_name = join(sogm_folder, 'ply_{:s}_{:s}.ply'.format(seq_name, frame_name))
            write_ply(ply_name,
                      [world_points, sem_labels],
                      ['x', 'y', 'z', 'classif'])

            print('Done')

        # Save current as ptcloud video
        if event.key in ['g', 'G']:

            video_i0 = -30
            video_i1 = 70
            if f_i + video_i1 >= len(dataset.frames[s_ind]) or f_i + video_i0 < 0:
                print('Invalid f_i')
                return

            sogm_folder = join(dataset.original_path, 'inspect_images')
            print(sogm_folder)
            if not exists(sogm_folder):
                makedirs(sogm_folder)

            # Video path
            seq_name = dataset.sequences[s_ind]
            video_path = join(sogm_folder, 'vid_{:s}_{:s}.gif'.format(seq_name, dataset.frames[s_ind][f_i]))

            # Get the pointclouds
            vid_pts = []
            vid_labels = []
            vid_ts = []
            vid_H0 = []
            vid_H1 = []
            for vid_i in range(video_i0, video_i1):
                frame_name = dataset.frames[s_ind][f_i + vid_i]
                H0 = dataset.poses[s_ind][f_i + vid_i - 1]
                H1 = dataset.poses[s_ind][f_i + vid_i]
                data = read_ply(join(dataset.seq_path[s_ind], frame_name + '.ply'))
                f_points = np.vstack((data['x'], data['y'], data['z'])).T
                f_ts = data['time']
                data = read_ply(join(dataset.annot_path[s_ind], frame_name + '.ply'))
                sem_labels = data['classif']

                vid_pts.append(f_points)
                vid_labels.append(sem_labels)
                vid_ts.append(f_ts)
                vid_H0.append(H0)
                vid_H1.append(H1)

            map_folder = join(dataset.original_path, 'slam_offline', map_day)
            map_names = np.sort([f for f in listdir(map_folder) if f.startswith('map_update_')])
            last_map = join(map_folder, map_names[-1])

            # Create video
            open_3d_vid(video_path,
                        vid_pts,
                        vid_labels,
                        vid_ts,
                        vid_H0,
                        vid_H1,
                        map_path=last_map)

            print('Done')

        return

    cid = figA.canvas.mpl_connect('key_press_event', onkey)


    #############################
    # Create a interactive window
    #############################


    def update_display():

        # Redifine sliders
        time_slider.val = 0
        time_slider.valmin = 0
        time_slider.valmax = len(all_pts) - 1
        time_slider.ax.set_xlim(time_slider.valmin, time_slider.valmax)

        # Redraw masks
        class_mask = np.zeros_like(dataset.all_inds[:, 0], dtype=bool)
        class_mask[dataset.class_frames[i_l]] = True
        seq_mask = dataset.all_inds[:, 0] == s_ind
        seq_class_frames = class_mask[seq_mask]
        seq_class_frames = np.expand_dims(seq_class_frames, 0)
        axdyn0.imshow(seq_class_frames, cmap='GnBu', aspect='auto')
        axdyn1.imshow(class_mask_eroded, cmap='OrRd', aspect='auto')
        axdyn2.imshow(np.expand_dims(all_risky_mask[s_ind], 0), cmap='Reds', aspect='auto')
        axdyn3.imshow(np.expand_dims(all_colli_mask[s_ind], 0), cmap='Greys', aspect='auto')
        pltodist[0].set_xdata(np.arange(len(all_dists[s_ind])))
        pltodist[0].set_ydata(all_dists[s_ind])
        axplotdist.set_xlim(0, len(all_dists[s_ind] - 1))

        # Update points
        update_points(time_slider.val)
        
        plt.draw()

        return pltodist

    # One button for each session
    figB = plt.figure(figsize=(11, 5))
    rax = plt.axes([0.05, 0.05, 0.9, 0.9], facecolor='lightgrey')
    radio_texts = [s + ': ' + train_comments[i] for i, s in enumerate(train_days)]
    radio_texts_to_i = {s: i for i, s in enumerate(radio_texts)}
    radio = RadioButtons(rax, radio_texts)
    
    def radio_func(label):
        # Load current sequence data
        nonlocal all_pts, all_colors, all_labels, class_mask_opened, s_ind, class_mask_eroded, seq_mask
        s_ind = radio_texts_to_i[label]
        data = loading_session(dataset, s_ind, i_l, im_lim, colormap)
        all_pts, all_colors, all_labels, class_mask_opened, class_mask_eroded, seq_mask = data
        update_display()
        return

    radio.on_clicked(radio_func)

    plt.show()

    print('    > Done')

    print('\n')
    print('  +-----------------------------------+')
    print('  | Finished all the annotation tasks |')
    print('  +-----------------------------------+')
    print('\n')

    return


# ----------------------------------------------------------------------------------------------------------------------
#
#           Choice of log to show
#       \***************************/
#


def trained_models():
    
    # Using the dates of the logs, you can easily gather consecutive ones. All logs should be of the same dataset.
    start = 'Log_2022-06-09_09-18-53'
    end = 'Log_2022-06-22_17-21-21'

    # Path to the results logs
    res_path = 'results'

    # Gathering names
    logs = np.sort([join(res_path, log) for log in listdir(res_path) if start <= log <= end])

    # Optinally add some specific folder that is not between start and end
    logs = np.insert(logs, 0, 'results/Log_2022-06-01_08-35-48')
    logs = np.insert(logs, 1, 'results/Log_2022-05-27_16-46-35')
    logs = logs.astype('<U50')

    # Give names to the logs (for legends). These logs were all done with e500 and rot augment
    logs_names = ['Myhal-A1234',
                  'Myhal-A123',
                  'Myhal-A12',
                  'Myhal-A1',
                  'A+Sim_40',
                  'A+Sim_20',
                  'A+Sim_50',
                  'A+Sim_60',
                  'A+Sim_80',
                  'only-Sim',
                  'A+H',
                  'A+H+Sim_50',
                  'A+H+Sim_20',
                  'A+H+Sim_80',
                  'etc']

    logs_names = np.array(logs_names[:len(logs)])

    return logs, logs_names


# ----------------------------------------------------------------------------------------------------------------------
#
#           Main call
#       \***************/
#

def Exp_lifelong():

    
    ######################################
    # Step 1: Choose what you want to plot
    ######################################

    plotting = 'PR'  # Comparison of the performances with good metrics
    # plotting = 'PR-100'  # Comparison of the performances with good metrics
    
    # Function returning the names of the log folders that we want to plot
    logs, logs_names = trained_models()
    

    ############################################
    # Step 2: See what validation we want to use
    ############################################

    log_val_days = []
    for log in logs:
        config = Config()
        config.load(log)
        val_path = join(log, 'val_preds')
        if exists(val_path):
            this_val_days = np.unique([f[:19] for f in listdir(val_path) if f.endswith('pots.ply')])
        else:
            this_val_days = np.array([])
        log_val_days += [this_val_days]
    all_val_days = np.unique(np.hstack(log_val_days))


    # List all possible datasets
    # data_folders = ['UTI3D_A', 'Simulation']
    data_folders = ['UTI3D_A', 'UTI3D_H', 'Simulation']
    data_paths = [join('../Data', f) for f in data_folders]

    sorted_val_days = []
    for data_path in data_paths:
        if 'Simulation' in data_path:
            val_days = [val_day for val_day in all_val_days
                        if exists(join(data_path, 'simulated_runs', val_day))]
        else:
            val_days = [val_day for val_day in all_val_days
                        if exists(join(data_path, 'runs', val_day))]

        sorted_val_days.append(val_days)


    print('\n')
    
    print('Possible Validation')
    print('*******************\n')

    print_sorted_val_table(logs_names, log_val_days, sorted_val_days, data_folders)


    # We perform validation on each dataset independantly
    
    # Use validation days from last log
    print('WARNING: using validation days from last log')
    all_val_days = [day11 for day11 in all_val_days if day11 in log_val_days[-1]]

    #############################
    # Step 3: Start plot function
    #############################

    if plotting.startswith('PR'):

        if plotting == 'PR':
            plt_chkp = -1
        else:
            plt_chkp = int(plotting.split('-')[-1])

        # Comparison of the performances with good metrics
        comparison_metrics_multival(logs,
                                    logs_names,
                                    sorted_val_days,
                                    data_paths,
                                    plt_chkp=plt_chkp)




    return


def Fig_SOGM_SRM():
    
    
    ######################################
    # Step 1: Choose what you want to plot
    ######################################
    
    # Function returning the names of the log folders that we want to plot
    logs, logs_names = trained_models()

    # get test models
    test_model_names = ['Myhal-A1234',
                        'A+Sim_50',
                        'A+H',
                        'A+H+Sim_50']
    test_model_mask = np.sum([(logs_names == test_model_name) for test_model_name in test_model_names], axis=0)
    inds = np.where(test_model_mask > 0)
    logs = logs[inds]
    logs_names = logs_names[inds]

    ############################################
    # Step 2: See what validation we want to use
    ############################################

    log_val_days = []
    for log in logs:
        config = Config()
        config.load(log)
        val_path = join(log, 'val_preds')
        if exists(val_path):
            this_val_days = np.unique([f[:19] for f in listdir(val_path) if f.endswith('pots.ply')])
        else:
            this_val_days = np.array([])
        log_val_days += [this_val_days]
    all_val_days = np.unique(np.hstack(log_val_days))


    # List all possible datasets
    # data_folders = ['UTI3D_A', 'Simulation']
    data_folders = ['UTI3D_A', 'UTI3D_H', 'Simulation']
    data_paths = [join('../Data', f) for f in data_folders]

    sorted_val_days = []
    for data_path in data_paths:
        if 'Simulation' in data_path:
            val_days = [val_day for val_day in all_val_days
                        if exists(join(data_path, 'simulated_runs', val_day))]
        else:
            val_days = [val_day for val_day in all_val_days
                        if exists(join(data_path, 'runs', val_day))]

        sorted_val_days.append(val_days)


    print('\n')
    
    print('Possible Validation')
    print('*******************\n')

    print_sorted_val_table(logs_names, log_val_days, sorted_val_days, data_folders)


    #############################
    # Step 3: Start plot function
    #############################

    all_wanted_s = []
    all_wanted_f = []
    all_wanted_s = ['2021-12-10_13-06-09', '2021-12-10_13-06-09', '2022-05-20_12-47-48', '2022-05-20_12-47-48', '2022-05-20_12-47-48', '2022-05-20_12-47-48', '2022-05-31_19-34-18',
                    '2022-05-31_19-40-52', '2022-05-31_19-40-52', '2022-05-31_19-40-52', '2022-03-09_15-58-56', '2022-03-09_15-58-56', '2022-03-09_16-03-21', '2022-03-09_16-03-21',
                    '2022-03-09_16-03-21', '2022-03-09_16-03-21', '2022-03-09_16-03-21', '2022-03-09_16-03-21', '2022-03-09_16-03-21', '2022-03-09_16-03-21', '2022-03-09_16-03-21',
                    '2022-03-09_16-03-21', '2022-03-22_14-12-20', '2022-03-22_14-12-20', '2022-03-22_14-12-20', '2022-03-22_14-12-20', '2022-03-22_14-12-20', '2022-03-22_14-12-20',
                    '2022-03-22_14-12-20', '2022-03-22_16-08-09', '2022-03-22_16-08-09', '2022-03-22_16-08-09', '2022-03-22_16-08-09', '2022-03-22_16-08-09', '2022-03-22_16-08-09',
                    '2022-03-22_16-08-09', '2022-03-22_16-08-09', '2022-03-22_16-08-09', '2022-03-22_16-08-09', '2022-03-22_16-08-09', '2022-03-22_16-08-09', '2022-03-22_16-08-09',
                    '2022-03-22_16-08-09', '2022-03-28_14-53-33', '2022-03-28_14-53-33', '2022-03-28_14-53-33', '2022-03-28_14-53-33', '2022-03-28_14-53-33', '2022-03-28_16-56-52',
                    '2022-03-28_16-56-52', '2022-03-28_16-56-52', '2022-03-28_16-56-52', '2022-03-28_16-56-52', '2022-03-28_16-56-52', '2022-03-28_16-56-52',
                    '2022-04-01_14-00-06', '2022-04-01_14-00-06', '2022-04-01_14-00-06', '2022-04-01_14-00-06', '2022-04-01_14-00-06', '2022-04-01_14-57-35', '2022-04-01_14-57-35',
                    '2022-04-01_15-06-55', '2022-04-01_15-06-55', '2022-04-01_15-06-55', '2022-04-01_15-06-55', '2022-04-01_15-06-55', '2022-04-01_15-06-55', '2022-04-01_15-06-55',
                    '2022-04-01_15-06-55', '2022-04-01_15-06-55', '2022-04-01_15-06-55', '2022-04-01_15-06-55', '2022-04-01_15-06-55', '2022-04-01_15-06-55', '2022-04-01_15-06-55',
                    '2022-04-01_15-06-55', '2022-04-01_15-06-55', '2022-04-01_15-06-55', '2022-04-01_15-06-55', '2022-04-01_15-06-55', '2022-04-01_15-06-55', '2022-04-01_15-06-55',
                    '2022-04-01_15-06-55', '2022-04-01_15-06-55', '2022-04-01_15-06-55', '2022-04-01_15-06-55', '2022-04-01_15-06-55', '2022-04-01_15-06-55', '2022-04-01_15-06-55',
                    '2022-04-01_15-06-55', '2022-04-01_15-11-29', '2022-04-01_15-11-29', '2022-04-01_15-11-29', '2022-04-01_15-11-29', '2022-04-01_15-11-29', '2022-04-01_15-11-29',
                    '2022-04-01_15-11-29', '2022-04-01_15-11-29', '2022-04-01_15-11-29', '2022-04-01_15-11-29', '2022-04-01_15-11-29', '2022-04-01_15-11-29', '2022-04-01_15-11-29']
    all_wanted_f = [1694, 1718, 265, 648, 1171, 1271, 995, 168, 206, 874, 839, 878, 26, 540, 878, 1142, 1369, 1470, 1590, 1598, 1613, 1615, 111, 646, 900, 937, 958, 1034, 1199,
                    18, 33, 289, 306, 312, 478, 516, 617, 630, 837, 1033, 1538, 1550, 1565, 152, 941, 955, 1191, 1224, 36, 39, 44, 65, 67, 303, 312, 38, 9, 477, 1095, 1117,
                    984, 1005, 5, 27, 41, 51, 64, 73, 107, 93, 120, 162, 243, 261, 278, 324, 355, 363, 409, 566, 608, 770, 788, 971, 975, 981, 1020, 1226, 1243, 1291, 1333, 202,
                    242, 226, 468, 563, 573, 585, 598, 604, 615, 628, 641, 882]


    print('\nGif Selection')
    print('*************\n')
    wanted_inds = wanted_gifs(logs[0], sorted_val_days, data_paths, all_wanted_s=all_wanted_s, all_wanted_f=all_wanted_f)

    print('\n')

    # Get the gif predictions
    comp_preds, comp_gts, comp_ingts = comparison_gifs(logs,
                                                       logs_names,
                                                       sorted_val_days,
                                                       data_paths,
                                                       wanted_inds=wanted_inds,
                                                       wanted_chkp=[-1],
                                                       redo=False)

    # show_SRM_gifs(logs, logs_names,
    #               sorted_val_days, data_paths, wanted_inds,
    #               comp_preds, comp_gts, comp_ingts)

    show_SOGM_gifs(logs, logs_names,
                   sorted_val_days, data_paths, wanted_inds,
                   comp_preds, comp_gts, comp_ingts)

    return


def Exp_real_comp():

    dataset_path, map_day, refine_sessions, train_sessions, train_comments = UTI3D_A_sessions_v2()

    inspect_sogm_sessions(dataset_path, map_day, train_sessions, train_comments)

    return


def test_data_compression():


    # Get a list of test frame
    dataset_path, _, _, train_sessions, train_comments = UTI3D_A_sessions_v2()

    n_runs = 2
    if n_runs < len(train_sessions):
        train_sessions = train_sessions[-n_runs:]
        train_comments = train_comments[-n_runs:]

    # Initialize datasets (dummy validation)
    config = MyhalCollisionConfig()
    dataset = MyhalCollisionDataset(config,
                                    train_sessions,
                                    chosen_set='training',
                                    dataset_path=dataset_path,
                                    balance_classes=True)


    # Get a list of test frames
    s_ind = 0
    n_f = 10

    print('------------------')
    print('Test_frames:')
    test_frames = []
    test_names = []
    for frame_name in dataset.frames[s_ind][:n_f]:
        test_frames.append(join(dataset.seq_path[s_ind], frame_name + '.ply'))
        test_names.append(frame_name)
        print(test_frames[-1])
    print('------------------')

    # Measure total size
    size0 = 0
    for frame_path in test_frames:
        size0 += os.path.getsize(frame_path)
    print('Total size: {:12d}  - {:6.1f}%%'.format(size0, 100))

    # Read points
    f_points = []
    f_ts = []
    f_is = []
    f_rs = []
    for frame_path in test_frames:
        data = read_ply(frame_path)
        f_points.append(np.vstack((data['x'], data['y'], data['z'])).T)
        f_ts.append(data['time'])
        f_is.append(data['intensity'])
        f_rs.append(data['ring'])



    # Save normal
    dir1 = join('results', 'zip_1')
    if not exists(dir1):
        makedirs(dir1)
    for f_ind, frame_name in enumerate(test_names):
        write_ply(join(dir1, '{:s}.ply'.format(frame_name)),
                  [f_points[f_ind], f_ts[f_ind], f_is[f_ind], f_rs[f_ind]],
                  ['x', 'y', 'z', 'time', 'intensity', 'ring'])
    shutil.make_archive(dir1, 'zip', dir1)
    size1 = os.path.getsize(dir1 + '.zip')
    print(' zip1 size: {:12d}  - {:6.1f}%%'.format(size1, 100 * size1 / size0))

    # Save with ring as uint8
    dir2 = join('results', 'zip_2')
    if not exists(dir2):
        makedirs(dir2)
    for f_ind, frame_name in enumerate(test_names):
        write_ply(join(dir2, '{:s}.ply'.format(frame_name)),
                  [f_points[f_ind], f_ts[f_ind], f_is[f_ind], f_rs[f_ind].astype(np.uint8)],
                  ['x', 'y', 'z', 'time', 'intensity', 'ring'])
    shutil.make_archive(dir2, 'zip', dir2)
    size2 = os.path.getsize(dir2 + '.zip')
    print(' zip2 size: {:12d}  - {:6.1f}%%'.format(size2, 100 * size2 / size0))


    # Save as npz compressed
    dir3 = join('results', 'zip_3')
    if not exists(dir3):
        makedirs(dir3)
    for f_ind, frame_name in enumerate(test_names):
        np.savez_compressed(join(dir3, '{:s}.npz'.format(frame_name)),
                            xyz=f_points[f_ind],
                            time=f_ts[f_ind],
                            intensity=f_is[f_ind],
                            ring=f_rs[f_ind].astype(np.uint8))
    shutil.make_archive(dir3, 'zip', dir3)
    size3 = os.path.getsize(dir3 + '.zip')
    print(' zip3 size: {:12d}  - {:6.1f}%%'.format(size3, 100 * size3 / size0))

    print('------------------')
    print('Time for ply loading')

    t1 = time.time()
    total = 0
    for step in range(2):
        f_points = []
        f_ts = []
        f_is = []
        f_rs = []
        for f_ind, frame_name in enumerate(test_names):
            data = read_ply(join(dir2, '{:s}.ply'.format(frame_name)))
            f_points.append(np.vstack((data['x'], data['y'], data['z'])).T)
            f_ts.append(data['time'])
            f_is.append(data['intensity'])
            f_rs.append(data['ring'])
        total += len(f_points)
    t2 = time.time()
    print('{:.1f} ms / frame'.format(1000 * (t2 - t1) / total))
    print('------------------')


    print('------------------')
    print('Time for npz loading')

    t1 = time.time()
    total = 0
    for step in range(2):
        f_points = []
        f_ts = []
        f_is = []
        f_rs = []
        for f_ind, frame_name in enumerate(test_names):
            data = np.load(join(dir3, '{:s}.npz'.format(frame_name)))
            f_points.append(data['xyz'])
            f_ts.append(data['time'])
            f_is.append(data['intensity'])
            f_rs.append(data['ring'])
        total += len(f_points)
    t2 = time.time()
    print('{:.1f} ms / frame'.format(1000 * (t2 - t1) / total))
    print('------------------')


    print('\n****************************\n')
    print('Conclusion')
    print('The npz format is better but not so much better than the ply format after zip')
    print('we keep ply and zip as it is much more simple')
    print('\n****************************\n')


    return

# ----------------------------------------------------------------------------------------------------------------------
#
#           Main call
#       \***************/
#


if __name__ == '__main__':


    # Exp_lifelong()

    Fig_SOGM_SRM()

    # Exp_real_comp()  # Remember to uncomment the runs in UTI3D-A

    # test_data_compression()





















