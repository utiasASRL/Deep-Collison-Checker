#
#
#      0=================================0
#      |    Kernel Point Convolutions    |
#      0=================================0
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
import os
import torch
os.environ.update(OMP_NUM_THREADS='1',
                  OPENBLAS_NUM_THREADS='1',
                  NUMEXPR_NUM_THREADS='1',
                  MKL_NUM_THREADS='1',)
import numpy as np
import matplotlib.pyplot as plt
from os.path import isfile, join, exists
from os import listdir, remove, getcwd, makedirs
from sklearn.metrics import confusion_matrix
import time
import pickle
from torch.utils.data import DataLoader
from matplotlib.animation import FuncAnimation
import matplotlib.patches as patches
from matplotlib.widgets import Slider, Button
import imageio

import scipy
from scipy import ndimage
import scipy.ndimage.filters as filters

# My libs
from utils.config import Config
from utils.metrics import IoU_from_confusions, smooth_metrics, fast_confusion, fast_threshold_stats
from utils.ply import read_ply
from models.architectures import FakeColliderLoss, KPCollider
from utils.tester import ModelTester
from utils.mayavi_visu import fast_save_future_anim, save_zoom_img, colorize_collisions, zoom_collisions, superpose_gt, \
    show_local_maxima, show_risk_diffusion, superpose_gt_contour, superpose_and_merge

# Datasets
from datasets.MyhalCollision import MyhalCollisionDataset, MyhalCollisionSampler, MyhalCollisionCollate, MyhalCollisionSamplerTest

# ----------------------------------------------------------------------------------------------------------------------
#
#           Utility functions
#       \***********************/
#


def running_mean(signal, n, axis=0, stride=1):

    # Create the smoothing convolution
    torch_conv = torch.nn.Conv1d(1, 1, kernel_size=2 * n + 1, stride=stride, bias=False)
    torch_conv.weight.requires_grad_(False)
    torch_conv.weight *= 0
    torch_conv.weight += 1 / (2 * n + 1)

    signal = np.array(signal)
    if signal.ndim == 1:

        # Reshape signal to torch Tensor
        signal = np.expand_dims(np.expand_dims(signal, 0), 1).astype(np.float32)
        torch_signal = torch.from_numpy(signal)

        # Get result
        smoothed = torch_conv(torch_signal).squeeze().numpy()

        return smoothed

    elif signal.ndim == 2:

        # transpose if we want axis 0
        if axis == 0:
            signal = signal.T

        # Reshape signal to torch Tensor
        signal = np.expand_dims(signal, 1).astype(np.float32)
        torch_signal = torch.from_numpy(signal)

        # Get result
        smoothed = torch_conv(torch_signal).squeeze().numpy()

        # transpose if we want axis 0
        if axis == 0:
            smoothed = smoothed.T

        return smoothed

    else:
        print('wrong dimensions')
        return None


def IoU_multi_metrics(all_IoUs, smooth_n):

    # Get mean IoU for consecutive epochs to directly get a mean
    all_mIoUs = [np.hstack([np.mean(obj_IoUs, axis=1) for obj_IoUs in epoch_IoUs]) for epoch_IoUs in all_IoUs]
    smoothed_mIoUs = []
    for epoch in range(len(all_mIoUs)):
        i0 = max(epoch - smooth_n, 0)
        i1 = min(epoch + smooth_n + 1, len(all_mIoUs))
        smoothed_mIoUs += [np.mean(np.hstack(all_mIoUs[i0:i1]))]

    # Get mean for each class
    all_objs_mIoUs = [[np.mean(obj_IoUs, axis=1) for obj_IoUs in epoch_IoUs] for epoch_IoUs in all_IoUs]
    smoothed_obj_mIoUs = []
    for epoch in range(len(all_objs_mIoUs)):
        i0 = max(epoch - smooth_n, 0)
        i1 = min(epoch + smooth_n + 1, len(all_objs_mIoUs))

        epoch_obj_mIoUs = []
        for obj in range(len(all_objs_mIoUs[0])):
            epoch_obj_mIoUs += [np.mean(np.hstack([objs_mIoUs[obj] for objs_mIoUs in all_objs_mIoUs[i0:i1]]))]

        smoothed_obj_mIoUs += [epoch_obj_mIoUs]

    return np.array(smoothed_mIoUs), np.array(smoothed_obj_mIoUs)


def IoU_class_metrics(all_IoUs, smooth_n):

    # Get mean IoU per class for consecutive epochs to directly get a mean without further smoothing
    smoothed_IoUs = []
    for epoch in range(len(all_IoUs)):
        i0 = max(epoch - smooth_n, 0)
        i1 = min(epoch + smooth_n + 1, len(all_IoUs))
        smoothed_IoUs += [np.mean(np.vstack(all_IoUs[i0:i1]), axis=0)]
    smoothed_IoUs = np.vstack(smoothed_IoUs)
    smoothed_mIoUs = np.mean(smoothed_IoUs, axis=1)

    return smoothed_IoUs, smoothed_mIoUs


def load_confusions(filename, n_class):

    with open(filename, 'r') as f:
        lines = f.readlines()

    confs = np.zeros((len(lines), n_class, n_class))
    for i, line in enumerate(lines):
        C = np.array([int(value) for value in line.split()])
        confs[i, :, :] = C.reshape((n_class, n_class))

    return confs


def load_training_results(path):

    filename = join(path, 'training.txt')
    with open(filename, 'r') as f:
        lines = f.readlines()

    epochs = []
    steps = []
    L_out = []
    L_p = []
    acc = []
    t = []
    L_2D_init = []
    L_2D_prop = []
    for line in lines[1:]:
        line_info = line.split()
        if (len(line) > 0):
            epochs += [int(line_info[0])]
            steps += [int(line_info[1])]
            L_out += [float(line_info[2])]
            L_p += [float(line_info[3])]
            acc += [float(line_info[4])]
            t += [float(line_info[5])]
            if len(line_info) > 6:
                L_2D_init += [float(line_info[6])]
                L_2D_prop += [float(line_info[7])]

        else:
            break

    ret_list = [epochs, steps, L_out, L_p, acc, t]

    if L_2D_init:
        ret_list.append(L_2D_init)
    if L_2D_prop:
        ret_list.append(L_2D_prop)

    return ret_list


def load_single_IoU(filename, n_parts):

    with open(filename, 'r') as f:
        lines = f.readlines()

    # Load all IoUs
    all_IoUs = []
    for i, line in enumerate(lines):
        all_IoUs += [np.reshape([float(IoU) for IoU in line.split()], [-1, n_parts])]
    return all_IoUs


def load_snap_clouds(path, dataset, only_last=False):

    cloud_folders = np.array([join(path, str(f, 'utf-8')) for f in listdir(path)
                              if str(f, 'utf-8').startswith('val_preds')])
    cloud_epochs = np.array([int(f.split('_')[-1]) for f in cloud_folders])
    epoch_order = np.argsort(cloud_epochs)
    cloud_epochs = cloud_epochs[epoch_order]
    cloud_folders = cloud_folders[epoch_order]

    Confs = np.zeros((len(cloud_epochs), dataset.num_classes, dataset.num_classes), dtype=np.int32)
    for c_i, cloud_folder in enumerate(cloud_folders):
        if only_last and c_i < len(cloud_epochs) - 1:
            continue

        # Load confusion if previously saved
        conf_file = join(cloud_folder, 'conf.txt')
        if isfile(conf_file):
            Confs[c_i] += np.loadtxt(conf_file, dtype=np.int32)

        else:
            for f in listdir(cloud_folder):
                f = str(f, 'utf-8')
                if f.endswith('.ply') and not f.endswith('sub.ply'):
                    data = read_ply(join(cloud_folder, f))
                    labels = data['class']
                    preds = data['preds']
                    Confs[c_i] += fast_confusion(labels, preds, dataset.label_values).astype(np.int32)

            np.savetxt(conf_file, Confs[c_i], '%12d')

        # Erase ply to save disk memory
        if c_i < len(cloud_folders) - 1:
            for f in listdir(cloud_folder):
                f = str(f, 'utf-8')
                if f.endswith('.ply'):
                    remove(join(cloud_folder, f))

    # Remove ignored labels from confusions
    for l_ind, label_value in reversed(list(enumerate(dataset.label_values))):
        if label_value in dataset.ignored_labels:
            Confs = np.delete(Confs, l_ind, axis=1)
            Confs = np.delete(Confs, l_ind, axis=2)

    return cloud_epochs, IoU_from_confusions(Confs)


def load_multi_snap_clouds(path, dataset, file_i, only_last=False):

    cloud_folders = np.array([join(path, f) for f in listdir(path) if f.startswith('val_preds')])
    cloud_epochs = np.array([int(f.split('_')[-1]) for f in cloud_folders])
    epoch_order = np.argsort(cloud_epochs)
    cloud_epochs = cloud_epochs[epoch_order]
    cloud_folders = cloud_folders[epoch_order]

    if len(cloud_folders) > 0:
        dataset_folders = [f for f in listdir(cloud_folders[0]) if dataset.name in f]
        cloud_folders = [join(f, dataset_folders[file_i]) for f in cloud_folders]

    Confs = np.zeros((len(cloud_epochs), dataset.num_classes, dataset.num_classes), dtype=np.int32)
    for c_i, cloud_folder in enumerate(cloud_folders):
        if only_last and c_i < len(cloud_epochs) - 1:
            continue

        # Load confusion if previously saved
        conf_file = join(cloud_folder, 'conf_{:s}.txt'.format(dataset.name))
        if isfile(conf_file):
            Confs[c_i] += np.loadtxt(conf_file, dtype=np.int32)

        else:
            for f in listdir(cloud_folder):
                if f.endswith('.ply') and not f.endswith('sub.ply'):
                    if np.any([cloud_path.endswith(f) for cloud_path in dataset.files]):
                        data = read_ply(join(cloud_folder, f))
                        labels = data['class']
                        preds = data['preds']
                        Confs[c_i] += confusion_matrix(labels, preds, dataset.label_values).astype(np.int32)

            np.savetxt(conf_file, Confs[c_i], '%12d')

        # Erase ply to save disk memory
        if c_i < len(cloud_folders) - 1:
            for f in listdir(cloud_folder):
                if f.endswith('.ply'):
                    remove(join(cloud_folder, f))

    # Remove ignored labels from confusions
    for l_ind, label_value in reversed(list(enumerate(dataset.label_values))):
        if label_value in dataset.ignored_labels:
            Confs = np.delete(Confs, l_ind, axis=1)
            Confs = np.delete(Confs, l_ind, axis=2)

    return cloud_epochs, IoU_from_confusions(Confs)


def load_multi_IoU(filename, n_parts):

    with open(filename, 'r') as f:
        lines = f.readlines()

    # Load all IoUs
    all_IoUs = []
    for i, line in enumerate(lines):
        obj_IoUs = [[float(IoU) for IoU in s.split()] for s in line.split('/')]
        obj_IoUs = [np.reshape(IoUs, [-1, n_parts[obj]]) for obj, IoUs in enumerate(obj_IoUs)]
        all_IoUs += [obj_IoUs]
    return all_IoUs


# ----------------------------------------------------------------------------------------------------------------------
#
#           Plot functions
#       \********************/
#


def compare_trainings(list_of_paths, list_of_labels=None, smooth_epochs=3.0):

    # Parameters
    # **********

    plot_lr = False
    stride = 2

    if list_of_labels is None:
        list_of_labels = [str(i) for i in range(len(list_of_paths))]

    # Read Training Logs
    # ******************

    all_epochs = []
    all_loss = []
    all_loss1 = []
    all_loss2 = []
    all_loss3 = []
    all_lr = []
    all_times = []
    all_RAMs = []

    for path in list_of_paths:

        # Check if log contains stuff
        check = 'val_IoUs.txt' in [str(f, 'utf-8') for f in listdir(path)]
        check = check or ('val_confs.txt' in [str(f, 'utf-8') for f in listdir(path)])
        check = check or ('val_RMSEs.txt' in [str(f, 'utf-8') for f in listdir(path)])

        if check:
            config = Config()
            config.load(path)
        else:
            continue

        # Load results
        training_res_list = load_training_results(path)
        if len(training_res_list) > 6:
            epochs, steps, L_out, L_p, acc, t, L_2D_init, L_2D_prop = training_res_list
        else:
            epochs, steps, L_out, L_p, acc, t = training_res_list
            L_2D_init = []
            L_2D_prop = []

        epochs = np.array(epochs, dtype=np.int32)
        epochs_d = np.array(epochs, dtype=np.float32)
        steps = np.array(steps, dtype=np.float32)

        # Compute number of steps per epoch
        max_e = np.max(epochs)
        first_e = np.min(epochs)
        epoch_n = []
        for i in range(first_e, max_e):
            bool0 = epochs == i
            e_n = np.sum(bool0)
            epoch_n.append(e_n)
            epochs_d[bool0] += steps[bool0] / e_n
        smooth_n = int(np.mean(epoch_n) * smooth_epochs)
        smooth_loss = running_mean(L_out, smooth_n, stride=stride)
        all_loss += [smooth_loss]
        if L_2D_init:
            all_loss2 += [running_mean(L_2D_init, smooth_n, stride=stride)]
            all_loss3 += [running_mean(L_2D_prop, smooth_n, stride=stride)]
            all_loss1 += [all_loss[-1] - all_loss2[-1] - all_loss3[-1]]
        all_epochs += [epochs_d[smooth_n:-smooth_n:stride]]
        all_times += [t[smooth_n:-smooth_n:stride]]

        # Learning rate
        if plot_lr:
            lr_decay_v = np.array([lr_d for ep, lr_d in config.lr_decays.items()])
            lr_decay_e = np.array([ep for ep, lr_d in config.lr_decays.items()])
            max_e = max(np.max(all_epochs[-1]) + 1, np.max(lr_decay_e) + 1)
            lr_decays = np.ones(int(np.ceil(max_e)), dtype=np.float32)
            lr_decays[0] = float(config.learning_rate)
            lr_decays[lr_decay_e] = lr_decay_v
            lr = np.cumprod(lr_decays)
            all_lr += [lr[np.floor(all_epochs[-1]).astype(np.int32)]]

        # Rescale losses
        rescale_losses = True
        if L_2D_init and rescale_losses:
            all_loss2[-1] *= 1 / config.power_2D_init_loss
            all_loss3[-1] *= 1 / config.power_2D_prop_loss

    # Plots learning rate
    # *******************

    if plot_lr:
        # Figure
        fig = plt.figure('lr')
        for i, label in enumerate(list_of_labels):
            plt.plot(all_epochs[i], all_lr[i], linewidth=1, label=label)

        # Set names for axes
        plt.xlabel('epochs')
        plt.ylabel('lr')
        plt.yscale('log')

        # Display legends and title
        plt.legend(loc=1)

        # Customize the graph
        ax = fig.gca()
        ax.grid(linestyle='-.', which='both')
        # ax.set_yticks(np.arange(0.8, 1.02, 0.02))

    # Plots loss
    # **********

    if all_loss2:

        fig, axes = plt.subplots(1, 3, sharey=False, figsize=(12, 5))

        for i, label in enumerate(list_of_labels):
            axes[0].plot(all_epochs[i], all_loss1[i], linewidth=1, label=label)
            axes[1].plot(all_epochs[i], all_loss2[i], linewidth=1, label=label)
            axes[2].plot(all_epochs[i], all_loss3[i], linewidth=1, label=label)

        # Set names for axes
        for ax in axes:
            ax.set_xlabel('epochs')
        axes[0].set_ylabel('loss')
        axes[0].set_yscale('log')

        # Display legends and title
        axes[2].legend(loc=1)
        axes[0].set_title('3D_net loss')
        axes[1].set_title('2D_init loss')
        axes[2].set_title('2D_prop loss')

        # Customize the graph
        for ax in axes:
            ax.grid(linestyle='-.', which='both')

    else:

        # Figure
        fig = plt.figure('loss')
        for i, label in enumerate(list_of_labels):
            plt.plot(all_epochs[i], all_loss[i], linewidth=1, label=label)

        # Set names for axes
        plt.xlabel('epochs')
        plt.ylabel('loss')
        plt.yscale('log')

        # Display legends and title
        plt.legend(loc=1)
        plt.title('Losses compare')

        # Customize the graph
        ax = fig.gca()
        ax.grid(linestyle='-.', which='both')
        # ax.set_yticks(np.arange(0.8, 1.02, 0.02))

    # Plot Times
    # **********

    # Figure
    fig = plt.figure('time')
    for i, label in enumerate(list_of_labels):
        plt.plot(all_epochs[i], np.array(all_times[i]) / 3600, linewidth=1, label=label)

    # Set names for axes
    plt.xlabel('epochs')
    plt.ylabel('time')
    # plt.yscale('log')

    # Display legends and title
    plt.legend(loc=0)

    # Customize the graph
    ax = fig.gca()
    ax.grid(linestyle='-.', which='both')
    # ax.set_yticks(np.arange(0.8, 1.02, 0.02))

    # Show all
    plt.show()


def compare_convergences_collision2D(list_of_paths, list_of_names=None, smooth_n=20):

    # Parameters
    # **********

    

    if list_of_names is None:
        list_of_names = [str(i) for i in range(len(list_of_paths))]

    # Read Logs
    # *********

    all_pred_epochs = []
    all_fe = []
    all_bce = []
    all_fp = []
    all_fp_bce = []
    all_fn = []
    all_fn_bce = []

    # Load parameters
    config = Config()
    config.load(list_of_paths[0])

    for path in list_of_paths:

        # Load config and saved results
        metric_list = []
        file_list = ['subpart_IoUs.txt',
                     'val_IoUs.txt',
                     'reconstruction_error.txt',
                     'future_error.txt',
                     'future_error_bce.txt',
                     'future_FP.txt',
                     'future_FN.txt',
                     'future_FP_bce.txt',
                     'future_FN_bce.txt']
        max_epoch = 0
        for filename in file_list:
            try:
                metric = np.loadtxt(join(path, filename))
                max_epoch = max(max_epoch, metric.shape[0])
                smoothed = running_mean(metric, smooth_n)
            except OSError as e:
                smoothed = np.zeros((0, 0), dtype=np.float64)
            metric_list.append(smoothed)
        (IoUs,
         val_IoUs,
         mean_recons_e,
         mean_future_e,
         mean_future_bce,
         mean_future_FP,
         mean_future_FN,
         mean_future_FP_bce,
         mean_future_FN_bce) = metric_list

        # Epoch count
        epochs_d = np.array([i for i in range(max_epoch)])

        # Aggregate results
        all_pred_epochs += [epochs_d[smooth_n:-smooth_n]]
        all_fe += [mean_future_e]
        all_bce += [mean_future_bce]
        all_fp += [mean_future_FP]
        all_fp_bce += [mean_future_FP_bce]
        all_fn += [mean_future_FN]
        all_fn_bce += [mean_future_FN_bce]

    # Plots
    # *****

    # create plots

    for reduc in ['mean']:
        for error, error_name in zip([all_fe, all_bce, all_fp, all_fp_bce, all_fn, all_fn_bce],
                                     ['all_fe', 'all_bce', 'all_fp', 'all_fp_bce', 'all_fn', 'all_fn_bce']):

            if 'bce' in error_name:
                continue

            fig = plt.figure(reduc + ' ' + error_name[4:])
            for i, name in enumerate(list_of_names):
                if error[i].shape[0] > 0:
                    if reduc == 'last':
                        plotted_e = error[i][:, -1]
                    else:
                        plotted_e = np.mean(error[i], axis=1)
                else:
                    plotted_e = all_pred_epochs[i] * 0
                p = plt.plot(all_pred_epochs[i], plotted_e, linewidth=1, label=name)

            plt.xlabel('epochs')
            plt.ylabel(reduc + ' ' + error_name[4:])

            # Set limits for y axis
            #plt.ylim(0.55, 0.95)

            # Display legends and title
            plt.legend()

            # Customize the graph
            ax = fig.gca()
            ax.grid(linestyle='-.', which='both')
            #ax.set_yticks(np.arange(0.8, 1.02, 0.02))

    # Show all -------------------------------------------------------------------
    plt.show()

    return


def evolution_gifs(chosen_log):

    ############
    # Parameters
    ############

    # Load parameters
    config = Config()
    config.load(chosen_log)

    # Find all checkpoints in the chosen training folder
    chkp_path = join(chosen_log, 'checkpoints')
    chkps = np.sort([join(chkp_path, f) for f in listdir(chkp_path) if f[:4] == 'chkp'])

    # Get training and validation days
    val_path = join(chosen_log, 'val_preds')
    val_days = np.unique([f.split('_')[0] for f in listdir(val_path) if f.endswith('pots.ply')])

    # Util ops
    softmax = torch.nn.Softmax(1)
    sigmoid_2D = torch.nn.Sigmoid()
    fake_loss = FakeColliderLoss(config)

    # Result folder
    visu_path = join(config.saving_path, 'test_visu')
    if not exists(visu_path):
        makedirs(visu_path)

    ##################################
    # Change model parameters for test
    ##################################

    # Change parameters for the test here. For example, you can stop augmenting the input data.
    config.augment_noise = 0
    config.augment_scale_min = 1.0
    config.augment_scale_max = 1.0
    config.augment_symmetries = [False, False, False]
    config.augment_rotation = 'none'
    config.validation_size = 100

    ##########################################
    # Choice of the image we want to visualize
    ##########################################

    # Dataset
    test_dataset = MyhalCollisionDataset(config, val_days, chosen_set='validation', balance_classes=False)

    wanted_inds = [700, 100, 150, 800]
    wanted_s_inds = [test_dataset.all_inds[ind][0] for ind in wanted_inds]
    wanted_f_inds = [test_dataset.all_inds[ind][1] for ind in wanted_inds]
    sf_to_i = {tuple(test_dataset.all_inds[ind]): i for i, ind in enumerate(wanted_inds)}

    ####################################
    # Preload to avoid long computations
    ####################################

    # List all precomputed preds:
    saved_preds = np.sort([f for f in listdir(visu_path) if f.endswith('.pkl')])
    saved_pred_inds = [int(f[:-4].split('_')[-1]) for f in saved_preds]

    # Load if available
    if np.all([ind in saved_pred_inds for ind in wanted_inds]):

        print('\nFound previous predictions, loading them')

        all_preds = []
        all_gts = []
        for ind in wanted_inds:
            wanted_ind_file = join(visu_path, 'preds_{:08d}.pkl'.format(ind))
            with open(wanted_ind_file, 'rb') as wfile:
                ind_preds, ind_gts = pickle.load(wfile)
            all_preds.append(ind_preds)
            all_gts.append(ind_gts)
        all_preds = np.stack(all_preds, axis=1)
        all_gts = np.stack(all_gts, axis=0)

    ########
    # Or ...
    ########

    else:

        ############
        # Choose GPU
        ############

        # Set which gpu is going to be used (auto for automatic choice)
        GPU_ID = 'auto'

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
        os.environ['CUDA_VISIBLE_DEVICES'] = GPU_ID
        chosen_gpu = int(GPU_ID)

        ###########################
        # Initialize model and data
        ###########################

        # Specific sampler with pred inds
        test_sampler = MyhalCollisionSamplerTest(test_dataset, wanted_inds)
        test_loader = DataLoader(test_dataset,
                                 batch_size=1,
                                 sampler=test_sampler,
                                 collate_fn=MyhalCollisionCollate,
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

        ######################################
        # Start predictions with ckpts weights
        ######################################

        all_preds = []
        all_gts = [None for _ in wanted_inds]

        for chkp_i, chkp in enumerate(chkps):

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
                    img0 = stck_init_preds[b_i, 0, :, :, :]
                    gt_im0 = np.copy(stck_future_gts[b_i, config.n_frames - 1, :, :, :])
                    gt_im1 = stck_future_gts[b_i, config.n_frames - 1, :, :, :]
                    gt_im1[:, :, 2] = np.max(stck_future_gts[b_i, :, :, :, 2], axis=0)
                    img1 = stck_init_preds[b_i, 1, :, :, :]
                    s_ind = f_inds[b_i, 0]
                    f_ind = f_inds[b_i, 1]

                    # Get the 2D predictions and gt (prop_2D)
                    img = stck_future_preds[b_i, :, :, :, :]
                    gt_im = stck_future_gts[b_i, config.n_frames:, :, :, :]

                    # # Future errors defined the same as the loss
                    if sf_to_i[(s_ind, f_ind)] == 0:
                        future_errors_bce = fake_loss.apply(gt_im, stck_future_logits[b_i, :, :, :, :], error='bce')
                        a = 1/0
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
                    if chkp_i == 0:
                        all_gts[sf_to_i[(s_ind, f_ind)]] = gt_im

                    if np.all([chkp_pred is not None for chkp_pred in chkp_preds]):
                        break

                if np.all([chkp_pred is not None for chkp_pred in chkp_preds]):
                    break

            # Store all predictions
            chkp_preds = np.stack(chkp_preds, axis=0)
            all_preds.append(chkp_preds)

        # All predictions shape: [chkp_n, frames_n, T, H, W, 3]
        all_preds = np.stack(all_preds, axis=0)

        # All gts shape: [frames_n, T, H, W, 3]
        all_gts = np.stack(all_gts, axis=0)

        # Save each preds
        for ind_i, ind in enumerate(wanted_inds):
            wanted_ind_file = join(visu_path, 'preds_{:08d}.pkl'.format(ind))
            with open(wanted_ind_file, 'wb') as wfile:
                pickle.dump((all_preds[:, ind_i], all_gts[ind_i]), wfile)

    ################
    # Visualizations
    ################

    # First idea: future for different chkp
    idea1 = True
    if idea1:

        for frame_i, _ in enumerate(wanted_inds):

            # Colorize and zoom both preds and gts
            showed_preds = colorize_collisions(all_preds[:, frame_i])
            showed_preds = zoom_collisions(showed_preds, 5)
            showed_gts = colorize_collisions(all_gts[frame_i])
            showed_gts = zoom_collisions(showed_gts, 5)
            
            # Repeat gt for all checkpoints and merge with preds
            showed_gts = np.expand_dims(showed_gts, 0)
            showed_gts = np.tile(showed_gts, (showed_preds.shape[0], 1, 1, 1, 1))
            merged_imgs = superpose_gt(showed_preds, showed_gts)

            c_showed = [0, 5, 10, -1]
            n_showed = len(c_showed)

            fig, axes = plt.subplots(1, n_showed)
            images = []
            for ax_i, chkp_i in enumerate(c_showed):
                images.append(axes[ax_i].imshow(merged_imgs[chkp_i, 0]))

            def animate(i):
                for ax_i, chkp_i in enumerate(c_showed):
                    images[ax_i].set_array(merged_imgs[chkp_i, i])
                return images

            anim = FuncAnimation(fig, animate,
                                 frames=np.arange(merged_imgs.shape[1]),
                                 interval=50,
                                 blit=True)

            plt.show()

            # SAME BUT COMPARE MULTIPLE LOGS AT THE END OF THEIR CONFERGENCE

    # Second idea: evolution of prediction for different timestamps
    idea2 = False
    if idea2:

        for frame_i, _ in enumerate(wanted_inds):

            # Colorize and zoom both preds and gts
            showed_preds = colorize_collisions(all_preds[:, frame_i])
            showed_preds = zoom_collisions(showed_preds, 5)
            showed_gts = colorize_collisions(all_gts[frame_i])
            showed_gts = zoom_collisions(showed_gts, 5)

            # Repeat gt for all checkpoints and merge with preds
            showed_gts = np.expand_dims(showed_gts, 0)
            showed_gts = np.tile(showed_gts, (showed_preds.shape[0], 1, 1, 1, 1))
            merged_imgs = superpose_gt(showed_preds, showed_gts)

            t_showed = [2, 10, 18, 26]
            n_showed = len(t_showed)

            fig, axes = plt.subplots(1, n_showed)
            images = []
            for t, ax in zip(t_showed, axes):
                images.append(ax.imshow(merged_imgs[0, t]))

            # Add progress rectangles
            xy = (0.2 * merged_imgs.shape[-3], 0.015 * merged_imgs.shape[-2])
            dx = 0.6 * merged_imgs.shape[-3]
            dy = 0.025 * merged_imgs.shape[-2]
            rect1 = patches.Rectangle(xy, dx, dy, linewidth=1, edgecolor='white', facecolor='white')
            rect2 = patches.Rectangle(xy, dx * 0.01, dy, linewidth=1, edgecolor='white', facecolor='green')
            axes[0].add_patch(rect1)
            axes[0].add_patch(rect2)
            images.append(rect1)
            images.append(rect2)

            def animate(i):
                for t_i, t in enumerate(t_showed):
                    images[t_i].set_array(merged_imgs[i, t])
                progress = float(i + 1) / merged_imgs.shape[0]
                images[-1].set_width(dx * progress)
                return images

            n_gif = merged_imgs.shape[0]
            animation_frames = np.arange(n_gif)
            animation_frames = np.pad(animation_frames, 10, mode='edge')
            anim = FuncAnimation(fig, animate,
                                 frames=animation_frames,
                                 interval=100,
                                 blit=True)

            plt.show()

    # # Create superposition of gt and preds
    # r = preds[:, :, :, 0]
    # g = preds[:, :, :, 1]
    # b = preds[:, :, :, 2]
    # r[gt_mask] += 0
    # g[gt_mask] += 0
    # b[gt_mask] += 255

    # # Compute precision recall curves
    # figPR = show_PR(p, gt)

    # #fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    # #anim0 = anim_multi_PR(p, gt, axis=axes[0])
    # #anim = show_future_anim(preds, axis=axes[1])
    # fig, double_anim = anim_PR_gif(preds, p, gt)

    # plt.show()

    a = 1 / 0

    return


def comparison_gifs(list_of_paths, wanted_inds=[]):

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
    for chosen_log in list_of_paths:

        ############
        # Parameters
        ############

        # Load parameters
        config = Config()
        config.load(chosen_log)

        # Find all checkpoints in the chosen training folder
        chkp_path = join(chosen_log, 'checkpoints')
        chkps = np.sort([join(chkp_path, f) for f in listdir(chkp_path) if f[:4] == 'chkp'])

        # Get training and validation days
        val_path = join(chosen_log, 'val_preds')
        val_days = np.unique([f.split('_')[0] for f in listdir(val_path) if f.endswith('pots.ply')])

        # Util ops
        softmax = torch.nn.Softmax(1)
        sigmoid_2D = torch.nn.Sigmoid()
        fake_loss = FakeColliderLoss(config)

        # Result folder
        visu_path = join(config.saving_path, 'test_visu')
        if not exists(visu_path):
            makedirs(visu_path)

        ####################################
        # Preload to avoid long computations
        ####################################

        # List all precomputed preds:
        saved_preds = np.sort([f for f in listdir(visu_path) if f.endswith('.pkl')])
        saved_pred_inds = [int(f[:-4].split('_')[-1]) for f in saved_preds]

        # Load if available
        if np.all([ind in saved_pred_inds for ind in wanted_inds]):

            print('\nFound previous predictions, loading them')

            all_preds = []
            all_gts = []
            all_ingts = []
            for ind in wanted_inds:
                wanted_ind_file = join(visu_path, 'preds_{:08d}.pkl'.format(ind))
                with open(wanted_ind_file, 'rb') as wfile:
                    ind_preds, ind_gts, ind_ingts = pickle.load(wfile)
                all_preds.append(np.copy(ind_preds))
                all_gts.append(np.copy(ind_gts))
                all_ingts.append(np.copy(ind_ingts))

            #print([ppp.shape for ppp in all_preds])
            all_preds = np.stack(all_preds, axis=1)
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

            ##################################
            # Change model parameters for test
            ##################################

            # Change parameters for the test here. For example, you can stop augmenting the input data.
            config.augment_noise = 0
            config.augment_scale_min = 1.0
            config.augment_scale_max = 1.0
            config.augment_symmetries = [False, False, False]
            config.augment_rotation = 'none'
            config.validation_size = 100

            ##########################################
            # Choice of the image we want to visualize
            ##########################################

            # Dataset
            test_dataset = MyhalCollisionDataset(config, val_days, chosen_set='validation', balance_classes=False)

            wanted_s_inds = [test_dataset.all_inds[ind][0] for ind in wanted_inds]
            wanted_f_inds = [test_dataset.all_inds[ind][1] for ind in wanted_inds]
            sf_to_i = {tuple(test_dataset.all_inds[ind]): i for i, ind in enumerate(wanted_inds)}

            ###########################
            # Initialize model and data
            ###########################

            # Specific sampler with pred inds
            test_sampler = MyhalCollisionSamplerTest(test_dataset, wanted_inds)
            test_loader = DataLoader(test_dataset,
                                     batch_size=1,
                                     sampler=test_sampler,
                                     collate_fn=MyhalCollisionCollate,
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

            all_preds = []
            all_gts = [None for _ in wanted_inds]
            all_ingts = [None for _ in wanted_inds]

            for chkp_i, chkp in enumerate(chkps):

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

                        # Store all predictions
                        chkp_preds[sf_to_i[(s_ind, f_ind)]] = img
                        if chkp_i == 0:
                            all_gts[sf_to_i[(s_ind, f_ind)]] = gt_im
                            all_ingts[sf_to_i[(s_ind, f_ind)]] = ingt_im

                        if np.all([chkp_pred is not None for chkp_pred in chkp_preds]):
                            break

                    if np.all([chkp_pred is not None for chkp_pred in chkp_preds]):
                        break

                # Store all predictions
                chkp_preds = np.stack(chkp_preds, axis=0)
                all_preds.append(chkp_preds)

            # All predictions shape: [chkp_n, frames_n, T, H, W, 3]
            all_preds = np.stack(all_preds, axis=0)

            # All gts shape: [frames_n, T, H, W, 3]
            all_gts = np.stack(all_gts, axis=0)
            all_ingts = np.stack(all_ingts, axis=0)

            # Save each preds
            for ind_i, ind in enumerate(wanted_inds):
                wanted_ind_file = join(visu_path, 'preds_{:08d}.pkl'.format(ind))
                with open(wanted_ind_file, 'wb') as wfile:
                    pickle.dump((np.copy(all_preds[:, ind_i]),
                                 np.copy(all_gts[ind_i]),
                                 np.copy(all_ingts[ind_i])), wfile)

        comparison_preds.append(all_preds)
        comparison_gts.append(all_gts)
        comparison_ingts.append(all_ingts)

        # Free cuda memory
        torch.cuda.empty_cache()

    # All predictions shape: [log_n, frames_n, T, H, W, 3]
    comparison_preds = np.stack([cp[-1] for cp in comparison_preds], axis=0)

    # All gts shape: [frames_n, T, H, W, 3]
    comparison_gts = comparison_gts[0]
    comparison_ingts = comparison_ingts[0]

    # ####################### DEBUG #######################
    # #
    # # Show the diffusing function here
    # #

    # for frame_i, w_i in enumerate(wanted_inds):
    #     collision_risk = comparison_preds[0, frame_i]

    #     fig1, anim1 = show_local_maxima(collision_risk[..., 2], neighborhood_size=5, threshold=0.1, show=False)
    #     fig2, anim2 = show_risk_diffusion(collision_risk, dl=0.12, diff_range=2.5, show=False)
    #     plt.show()

    # a = 1/0

    # #
    # #
    # ####################### DEBUG #######################

    ################
    # Visualizations
    ################

    
    test_dataset = MyhalCollisionDataset(config, val_days, chosen_set='validation', balance_classes=False)
    wanted_s_inds = [test_dataset.all_inds[ind][0] for ind in wanted_inds]
    wanted_f_inds = [test_dataset.all_inds[ind][1] for ind in wanted_inds]

    for frame_i, w_i in enumerate(wanted_inds):
           

        if True:

            # Colorize and zoom both preds and gts
            showed_preds = zoom_collisions(comparison_preds[:, frame_i], 5)
            showed_gts = zoom_collisions(comparison_gts[frame_i], 5)
            showed_ingts = zoom_collisions(comparison_ingts[frame_i], 5)

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

            c_showed = np.arange(showed_preds.shape[0])
            n_showed = len(c_showed)

            # fig, axes = plt.subplots(1, n_showed)
            # if n_showed == 1:
            #     axes = [axes]

            images = []
            for ax_i, log_i in enumerate(c_showed):

                # # Init plt
                # images.append(axes[ax_i].imshow(merged_imgs[log_i, 0]))

                # Save gif for the videos
                seq_name = test_dataset.sequences[wanted_s_inds[frame_i]]
                frame_name = test_dataset.frames[wanted_s_inds[frame_i]][wanted_f_inds[frame_i]]
                im_name = 'results/gif_{:s}_{:s}_{:d}.gif'.format(seq_name, frame_name, ax_i)
                imageio.mimsave(im_name, merged_imgs[log_i], fps=20)

            # def animate(i):
            #     for ax_i, log_i in enumerate(c_showed):
            #         images[ax_i].set_array(merged_imgs[log_i, i])
            #     return images

            # anim = FuncAnimation(fig, animate,
            #                      frames=np.arange(merged_imgs.shape[1]),
            #                      interval=50,
            #                      blit=True)

            # # Save last iamge for the paper
            # seq_name = test_dataset.sequences[wanted_s_inds[frame_i]]
            # frame_name = test_dataset.frames[wanted_s_inds[frame_i]][wanted_f_inds[frame_i]]
            # im_name = 'results/last_{:s}_{:s}.png'.format(seq_name, frame_name)
            # imageio.imsave(im_name, merged_imgs[0, -4])

            #plt.show()


        if False:

            # Colorize and zoom both preds and gts
            showed_preds = zoom_collisions(comparison_preds[:, frame_i], 5)
            showed_gts = zoom_collisions(comparison_gts[frame_i], 5)
            showed_ingts = zoom_collisions(comparison_ingts[frame_i], 5)

            # Repeat gt for all checkpoints and merge with preds
            showed_gts = np.expand_dims(showed_gts, 0)
            showed_gts = np.tile(showed_gts, (showed_preds.shape[0], 1, 1, 1, 1))
            showed_ingts = np.expand_dims(showed_ingts, 0)
            showed_ingts = np.tile(showed_ingts, (showed_preds.shape[0], 1, 1, 1, 1))

            # Merge colors
            #merged_imgs = superpose_gt(showed_preds, showed_gts * 0, showed_ingts, ingts_fade=(255, 0))
            #merged_imgs = superpose_gt(showed_preds, showed_gts * 0, showed_ingts, ingts_fade=(0, 0))
            
            # merged_imgs = superpose_gt_contour(showed_preds, showed_gts, showed_ingts, ingts_fade=(255, 0))

            merged_imgs1 = superpose_and_merge(showed_preds, showed_gts, showed_ingts, traj=True, contour=False)
            #merged_imgs2 = superpose_and_merge(showed_preds, showed_gts, showed_ingts, traj=False, contour=True)
            #merged_imgs3 = superpose_and_merge(showed_preds, showed_gts, showed_ingts, traj=True, contour=True)

            

            # # To show gt images
            # showed_preds = (showed_gts > 0.05).astype(showed_gts.dtype)
            # showed_preds[..., 2] *= 0.5
            # merged_imgs = superpose_gt(showed_preds, showed_gts * 0, showed_ingts, ingts_fade=(0, 0))

            c_showed = np.arange(showed_preds.shape[0])
            n_showed = len(c_showed)

            # # Init plt
            # fig, axes = plt.subplots(n_showed, 2)
            # if axes.ndim == 1:
            #     axes = np.expand_dims(axes, 0)
            # for ax_i, log_i in enumerate(c_showed):
            #     axes[ax_i, 0].imshow(merged_imgs1[log_i])
            #     # axes[ax_i, 1].imshow(merged_imgs2[log_i])
            #     # axes[ax_i, 2].imshow(merged_imgs3[log_i])


            seq_name = test_dataset.sequences[wanted_s_inds[frame_i]]
            frame_name = test_dataset.frames[wanted_s_inds[frame_i]][wanted_f_inds[frame_i]]

            im_name = 'results/spps_{:s}_{:s}.png'.format(seq_name, frame_name)
            imageio.imsave(im_name, merged_imgs1[0])
            print(w_i, seq_name, frame_name)

    plt.show()


    return


def comparison_metrics(list_of_paths, list_of_names=None):

    ############
    # Parameters
    ############

    if list_of_names is None:
        list_of_names = [str(i) for i in range(len(list_of_paths))]

    # Set which gpu is going to be used (auto for automatic choice)
    GPU_ID = 'auto'

    comparison_TP_FP_FN = []

    for chosen_log in list_of_paths:

        ############
        # Parameters
        ############

        # Load parameters
        config = Config()
        config.load(chosen_log)

        # Find all checkpoints in the chosen training folder
        chkp_path = join(chosen_log, 'checkpoints')
        chkps = np.sort([join(chkp_path, f) for f in listdir(chkp_path) if f[:4] == 'chkp'])

        # Only deal with one checkpoint (for faster computations)
        chkps = chkps[-2:-1]

        # Get the chkp_inds
        chkp_inds = [int(f[:-4].split('_')[-1]) for f in chkps]

        # Get training and validation days
        val_path = join(chosen_log, 'val_preds')
        val_days = np.unique([f.split('_')[0] for f in listdir(val_path) if f.endswith('pots.ply')])

        # Util ops
        softmax = torch.nn.Softmax(1)
        sigmoid_2D = torch.nn.Sigmoid()
        fake_loss = FakeColliderLoss(config)

        # Result folder
        visu_path = join(config.saving_path, 'test_metrics')
        if not exists(visu_path):
            makedirs(visu_path)

        ####################################
        # Preload to avoid long computations
        ####################################

        # List all precomputed preds:
        saved_res = np.sort([f for f in listdir(visu_path) if f.startswith('metrics_chkp') and f.endswith('.pkl')])
        saved_res_inds = [int(f[:-4].split('_')[-1]) for f in saved_res]

        #saved_res_inds = []

        # List of the chkp to do
        to_do_inds = [ind for ind in chkp_inds if ind not in saved_res_inds]
        to_load_inds = [ind for ind in chkp_inds if ind in saved_res_inds]

        # Results
        all_TP_FP_FN = [None for ind in chkp_inds]

        # Load if available
        if len(to_load_inds) > 0:

            print('\nFound previous predictions, loading them')

            for chkp_i, chkp in enumerate(chkps):

                if chkp_inds[chkp_i] not in to_load_inds:
                    continue

                # Save preds for this chkp
                chkp_stat_file = join(visu_path, 'metrics_chkp_{:04d}.pkl'.format(chkp_inds[chkp_i]))
                with open(chkp_stat_file, 'rb') as rfile:
                    chkp_TP_FP_FN = pickle.load(rfile)
                   
                # Store all predictions 
                all_TP_FP_FN[chkp_i] = chkp_TP_FP_FN

        ########
        # Or ...
        ########

        if len(to_do_inds) > 0:

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

            ##################################
            # Change model parameters for test
            ##################################

            # Change parameters for the test here. For example, you can stop augmenting the input data.
            config.augment_noise = 0
            config.augment_scale_min = 1.0
            config.augment_scale_max = 1.0
            config.augment_symmetries = [False, False, False]
            config.augment_rotation = 'none'
            config.validation_size = 1000

            ##########################################
            # Choice of the image we want to visualize
            ##########################################

            # Dataset
            test_dataset = MyhalCollisionDataset(config, val_days, chosen_set='validation', balance_classes=False)

            ###########################
            # Initialize model and data
            ###########################

            # Specific sampler with pred inds
            wanted_inds = np.arange(0, test_dataset.all_inds.shape[0], 10)
            test_sampler = MyhalCollisionSamplerTest(test_dataset, wanted_inds)
            test_loader = DataLoader(test_dataset,
                                     batch_size=1,
                                     sampler=test_sampler,
                                     collate_fn=MyhalCollisionCollate,
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

                if chkp_inds[chkp_i] not in to_do_inds:
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
                chkp_done = []
                for s_ind, seq_frames in enumerate(test_dataset.frames):
                    chkp_TP_FP_FN.append(np.zeros((len(seq_frames), config.n_2D_layers, PR_resolution, 3), dtype=np.int32))
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

                            gt_flat = np.reshape(stck_future_gts[b_i, config.n_frames:, :, :, 2], (config.n_2D_layers, -1)) > 0.01
                            p_flat = np.reshape(stck_future_preds[b_i, :, :, :, 2], (config.n_2D_layers, -1))

                            # Get the result metrics [T, n_thresh, 3]
                            res_TP_FP_FN = fast_threshold_stats(gt_flat, p_flat, n_thresh=PR_resolution)

                            # Store result in container [seqs][frames, T, n_thresh, 3] 
                            chkp_TP_FP_FN[s_ind][f_ind, :, :, :] = res_TP_FP_FN
                            chkp_done[s_ind][f_ind] = True

                        # print([np.sum(c_done.astype(np.int32)) / np.prod(c_done.shape) for c_done in chkp_done])

                        count = np.sum([np.sum(c_done.astype(np.int32)) for c_done in chkp_done])
                        if count == last_count:
                            break
                        else:
                            last_count = count

                # Store all predictions
                chkp_TP_FP_FN = np.concatenate(chkp_TP_FP_FN, axis=0)
                all_TP_FP_FN[chkp_i] = chkp_TP_FP_FN
                
                # Save preds for this chkp
                chkp_stat_file = join(visu_path, 'metrics_chkp_{:04d}.pkl'.format(chkp_inds[chkp_i]))
                with open(chkp_stat_file, 'wb') as wfile:
                    pickle.dump(np.copy(chkp_TP_FP_FN), wfile)

            # Free cuda memory
            torch.cuda.empty_cache()

        # All TP_FP_FN shape: [chkp_n, frames_n, T, nt, 3]
        all_TP_FP_FN = np.stack(all_TP_FP_FN, axis=0)
        comparison_TP_FP_FN.append(all_TP_FP_FN)

    ################
    # Visualizations
    ################



    if False:

        # Figure
        figB, axB = plt.subplots(1, 1, figsize=(8, 6))

        # Plot last PR curve for each log
        for i, name in enumerate(list_of_names):

            # [frames_n, T, nt, 3]
            all_TP_FP_FN = comparison_TP_FP_FN[i][-1]

            # [nt, 3]
            chosen_TP_FP_FN = np.sum(all_TP_FP_FN, axis=(0, 1))

            tps = chosen_TP_FP_FN[..., 0]
            fps = chosen_TP_FP_FN[..., 1]
            fns = chosen_TP_FP_FN[..., 2]

            pre = tps / (fns + tps + 1e-6)
            rec = tps / (fps + tps + 1e-6)

            axB.plot(rec, pre, linewidth=1, label=name)
        
        # Customize the graph
        axB.grid(linestyle='-.', which='both')
        axB.set_xlim(0, 1)
        axB.set_ylim(0, 1)

        # Set names for axes
        plt.xlabel('recall')
        plt.ylabel('precision')

        # Display legends and title
        plt.legend()

    if True:

        # Figure
        figA, axA = plt.subplots(1, 1, figsize=(10, 7))
        plt.subplots_adjust(bottom=0.25)

        # Init threshold
        max_Nt = np.max([aaa.shape[-3] for aaa in comparison_TP_FP_FN])
        allowed_times = (np.arange(max_Nt, dtype=np.float32) + 1) * config.T_2D / config.n_2D_layers
        time_step = config.T_2D / config.n_2D_layers
        time_ind_0 = 9
        time_0 = allowed_times[time_ind_0]

        # Plot last PR curve for each log
        plotsA = []
        all_preA = []
        all_recA = []
        for i, name in enumerate(list_of_names):

            # [frames_n, T, nt, 3]
            all_TP_FP_FN = comparison_TP_FP_FN[i][-1]

            # Chosen timestamps [T, nt, 3]
            chosen_TP_FP_FN = np.sum(all_TP_FP_FN, axis=0)

            tps = chosen_TP_FP_FN[..., 0]
            fps = chosen_TP_FP_FN[..., 1]
            fns = chosen_TP_FP_FN[..., 2]

            pre = tps / (fns + tps + 1e-6)
            rec = tps / (fps + tps + 1e-6)

            plotsA += axA.plot(rec[time_ind_0], pre[time_ind_0], linewidth=1, label=name)
            all_preA.append(pre)
            all_recA.append(rec)
        
        # Customize the graph
        axA.grid(linestyle='-.', which='both')
        axA.set_xlim(0, 1)
        axA.set_ylim(0, 1)

        # Set names for axes
        plt.xlabel('recall')
        plt.ylabel('precision')

        # Display legends and title
        plt.legend()
        
        # Make a horizontal slider to control the frequency.
        axcolor = 'lightgoldenrodyellow'
        axtime = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor=axcolor)
        time_slider = Slider(ax=axtime,
                             label='Time',
                             valmin=np.min(allowed_times),
                             valmax=np.max(allowed_times),
                             valinit=time_0,
                             valstep=time_step)

        # The function to be called anytime a slider's value changes
        def update_PR(val):
            time_ind = (int)(np.round(val * config.n_2D_layers / config.T_2D)) - 1
            for plot_i, log_plot in enumerate(plotsA):
                if time_ind < all_preA[plot_i].shape[0]:
                    log_plot.set_xdata(all_recA[plot_i][time_ind])
                    log_plot.set_ydata(all_preA[plot_i][time_ind])

        # register the update function with each slider
        time_slider.on_changed(update_PR)

    if False:

        # Figure
        figB, axes = plt.subplots(1, 3, figsize=(10, 5))
        
        plt.subplots_adjust(bottom=0.25)

        # Init threshold
        n_thresh = comparison_TP_FP_FN[0].shape[-2]
        thresholds = np.linspace(0.0, 1.0, n_thresh, endpoint=False)
        thresholds = 1 - thresholds ** 1
        thresh = 0.5
        thresh_i = (n_thresh - 1) - np.searchsorted(thresholds[::-1], thresh)


        # Plot last PR curve for each log
        plots = []
        all_pre = []
        all_rec = []
        all_f1s = []
        for i, name in enumerate(list_of_names):

            # [frames_n, T, nt, 3]
            all_TP_FP_FN = comparison_TP_FP_FN[i][-1]

            # Init x-axis values
            times = np.arange(comparison_TP_FP_FN[i].shape[-3])

            # Chosen timestamps [T, nt, 3]
            chosen_TP_FP_FN = np.sum(all_TP_FP_FN, axis=0)

            # Chosen timestamps [nt, T, 3]
            chosen_TP_FP_FN = np.transpose(chosen_TP_FP_FN, (1, 0, 2))

            tps = chosen_TP_FP_FN[..., 0]
            fps = chosen_TP_FP_FN[..., 1]
            fns = chosen_TP_FP_FN[..., 2]

            pre = tps / (fns + tps + 1e-6)
            rec = tps / (fps + tps + 1e-6)
            f1s = 2 * tps / (2 * tps + fps + fns + 1e-6)

            log_plots = []
            log_plots += axes[0].plot(times, pre[thresh_i], linewidth=1, label=name)
            log_plots += axes[1].plot(times, rec[thresh_i], linewidth=1, label=name)
            log_plots += axes[2].plot(times, f1s[thresh_i], linewidth=1, label=name)

            plots.append(log_plots)
            all_pre.append(pre)
            all_rec.append(rec)
            all_f1s.append(f1s)
                
        # Customize the graph
        for ax_i, ax in enumerate(axes):
            ax.grid(linestyle='-.', which='both')
            ax.set_ylim(0, 1)

        # Set names for axes
        axes[0].set_xlabel('precision')
        axes[1].set_xlabel('recall')
        axes[2].set_xlabel('F1-Score')

        # Display legends and title
        plt.legend()

        # Make a horizontal slider to control the frequency.
        axcolor = 'lightgoldenrodyellow'
        axthresh = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor=axcolor)
        thresh_slider = Slider(ax=axthresh,
                               label='Threshold',
                               valmin=0.0,
                               valmax=1.0,
                               valinit=thresh)

        # The function to be called anytime a slider's value changes
        def update_F1(val):
            thresh_i = (n_thresh - 1) - np.searchsorted(thresholds[::-1], val)
            for plot_i, log_plots in enumerate(plots):
                log_plots[0].set_ydata(all_pre[plot_i][thresh_i])
                log_plots[1].set_ydata(all_rec[plot_i][thresh_i])
                log_plots[2].set_ydata(all_f1s[plot_i][thresh_i])

        # register the update function with each slider
        thresh_slider.on_changed(update_F1)

    
    if True:

        # Figure
        figC, axesC = plt.subplots(1, 1, figsize=(4, 3))

        # Plot last PR curve for each log
        for i, name in enumerate(list_of_names):

            # [frames_n, T, nt, 3]
            all_TP_FP_FN = comparison_TP_FP_FN[i][-1]

            # Init x-axis values
            times = np.arange(comparison_TP_FP_FN[i].shape[-3])
            times = times.astype(np.float32) / 10

            # Chosen timestamps [T, nt, 3]
            chosen_TP_FP_FN = np.sum(all_TP_FP_FN, axis=0)

            # Chosen timestamps [nt, T, 3]
            chosen_TP_FP_FN = np.transpose(chosen_TP_FP_FN, (1, 0, 2))

            # PR [nt, T]
            tps = chosen_TP_FP_FN[..., 0]
            fps = chosen_TP_FP_FN[..., 1]
            fns = chosen_TP_FP_FN[..., 2]

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

            # Average precision as computed by scikit
            AP = np.sum((rec[1:] - rec[:-1]) * pre[1:], axis=0)

            print(name, 'mAP=', np.mean(AP))

            axesC.plot(times, f1s[best_mean], linewidth=1, label=name)

                
        # Customize the graph
        axesC.grid(linestyle='-.', which='both')
        #axesC.set_ylim(0, 1)

        # Set names for axes
        axesC.set_xlabel('Time Layer in SOGM (sec)')
        axesC.set_ylabel('AP')

        # Display legends and title
        plt.legend()

        fname = 'results/AP_fig.pdf'
        plt.savefig(fname,
                    bbox_inches='tight')



    plt.show()

    return


# ----------------------------------------------------------------------------------------------------------------------
#
#           Experiments
#       \*****************/
#


def collider_tests_1(old_result_limit):
    """
    A lot have been going on, we know come back to basics and experiment with bouncers. In this experiment, bouncers have various speeds, and we try different loss weights
    """

    # Using the dates of the logs, you can easily gather consecutive ones. All logs should be of the same dataset.
    start = 'Log_2021-04-14_18-48-28'
    end = 'Log_2021-04-21_15-05-54'

    if end < old_result_limit:
        res_path = 'old_results'
    else:
        res_path = 'results'

    logs = np.sort([join(res_path, log) for log in listdir(res_path) if start <= log <= end])
    logs = logs.astype('<U50')

    # Give names to the logs (for legends)
    logs_names = ['Bouncer-Loss=1/1',
                  'Bouncer-Loss=0.5/4',
                  'Bouncer-Loss=0.5/4-NEW_METRIC',
                  'Bouncer-Loss=1/2-NEW_METRIC',
                  'Bouncer-Loss=1/2-NEW_METRIC',
                  'Bouncer-Loss=1/2-NEW_METRIC',
                  'test'
                  ]

    logs_names = np.array(logs_names[:len(logs)])

    return logs, logs_names


def collider_tests_2(old_result_limit):
    """
    Nouveau loss, et nouvelles valeur de validation pour ces tests
    """

    # Using the dates of the logs, you can easily gather consecutive ones. All logs should be of the same dataset.
    start = 'Log_2021-04-21_15-05-54'
    end = 'Log_2021-04-30_11-07-40'

    if end < old_result_limit:
        res_path = 'old_results'
    else:
        res_path = 'results'

    logs = np.sort([join(res_path, log) for log in listdir(res_path) if start <= log <= end])
    logs = logs.astype('<U50')

    # Give names to the logs (for legends)
    logs_names = ['Bouncer-Loss=1/2-v2',
                  'Bouncer-Loss=1/2-v0',
                  'Bouncer-Loss=1/2-v1',
                  'Bouncer-5frames-failed',
                  'Bouncer-5frames-Loss=1/8-v2',
                  ]

    logs_names = np.array(logs_names[:len(logs)])

    return logs, logs_names


def collider_tests_Bouncers(old_result_limit):
    """
    Nouveau loss, et nouvelles valeur de validation pour ces tests
    """

    # Using the dates of the logs, you can easily gather consecutive ones. All logs should be of the same dataset.
    start = 'Log_2021-04-30_11-07-41'
    end = 'Log_2021-05-02_10-19-22'

    if end < old_result_limit:
        res_path = 'old_results'
    else:
        res_path = 'results'

    logs = np.sort([join(res_path, log) for log in listdir(res_path) if start <= log <= end])
    logs = logs.astype('<U50')

    # Give names to the logs (for legends)
    logs_names = ['Bouncer-5frames',
                  'Bouncer-3frames']

    logs_names = np.array(logs_names[:len(logs)])

    return logs, logs_names


def collider_tests_Wanderers(old_result_limit):
    """
    Nouveau loss, et nouvelles valeur de validation pour ces tests
    """

    # Using the dates of the logs, you can easily gather consecutive ones. All logs should be of the same dataset.
    start = 'Log_2021-05-03_17-57-57'
    end = 'Log_2021-05-10_10-00-22'

    if end < old_result_limit:
        res_path = 'old_results'
    else:
        res_path = 'results'

    logs = np.sort([join(res_path, log) for log in listdir(res_path) if start <= log <= end])
    logs = logs.astype('<U50')

    # Give names to the logs (for legends)
    logs_names = ['Wand-indep',
                  'Wand-shared',
                  'Wand-indep-5frames',
                  'Wand-indep-d120',
                  'Wand-indep-d120-loss*50',
                  'test']

    logs_names = np.array(logs_names[:len(logs)])

    return logs, logs_names


def collider_tests_Followers(old_result_limit):
    """
    Nouveau loss, et nouvelles valeur de validation pour ces tests
    """

    # Using the dates of the logs, you can easily gather consecutive ones. All logs should be of the same dataset.
    start = 'Log_2021-05-10_10-57-57'
    end = 'Log_2021-05-13_10-00-22'

    if end < old_result_limit:
        res_path = 'old_results'
    else:
        res_path = 'results'

    logs = np.sort([join(res_path, log) for log in listdir(res_path) if start <= log <= end])
    logs = logs.astype('<U50')
    logs = np.insert(logs, 5, 'results/Log_2021-06-01_21-33-07')

    # Give names to the logs (for legends)
    logs_names = ['Followers-v2',
                  'Followers-v0',
                  'Followers-v1',
                  'Followers-v1',
                  'Followers-v2',
                  'Followers-v2-pred>0.1',
                  'test']

    logs_names = np.array(logs_names[:len(logs)])

    return logs, logs_names


def collider_final_Bouncers(old_result_limit):
    """
    Nouveau loss, et nouvelles valeur de validation pour ces tests
    """

    # Using the dates of the logs, you can easily gather consecutive ones. All logs should be of the same dataset.
    start = 'Log_2021-05-24_13-59-19'
    end = 'Log_2021-05-24_13-59-20'

    if end < old_result_limit:
        res_path = 'old_results'
    else:
        res_path = 'results'

    logs = np.sort([join(res_path, log) for log in listdir(res_path) if start <= log <= end])
    logs = np.insert(logs, 0, 'results/Log_2021-05-27_17-20-02')
    logs = np.insert(logs, 1, 'results/Log_2021-05-27_17-23-43')
    logs = np.insert(logs, 3, 'results/Log_2021-05-31_18-55-24')
    logs = logs.astype('<U50')

    # Give names to the logs (for legends)
    logs_names = ['Bouncer-v0',
                  'Bouncer-v1',
                  'Bouncer-v2-pred>0.03',
                  'Bouncer-v2-pred>0.1']

    # logs_names = ['No Mask',
    #               'GT-Mask',
    #               'Active-Mask']

    logs_names = np.array(logs_names[:len(logs)])

    return logs, logs_names


def collider_final_Wanderers(old_result_limit):
    """
    Nouveau loss, et nouvelles valeur de validation pour ces tests
    """

    # Using the dates of the logs, you can easily gather consecutive ones. All logs should be of the same dataset.
    start = 'Log_2021-05-24_15-46-47'
    end = 'Log_2021-05-24_15-46-48'

    if end < old_result_limit:
        res_path = 'old_results'
    else:
        res_path = 'results'

    logs = np.sort([join(res_path, log) for log in listdir(res_path) if start <= log <= end])
    logs = np.insert(logs, 0, 'results/Log_2021-05-29_11-14-27')
    logs = np.insert(logs, 1, 'results/Log_2021-05-29_11-15-00')
    # logs = np.insert(logs, 3, 'results/Log_2021-05-31_18-56-33')
    logs = logs.astype('<U50')

    # Give names to the logs (for legends)
    logs_names = ['Wanderer-v0',
                  'Wanderer-v1',
                  'Wanderer-v2-pred>0.03',
                  'Wanderer-v2-pred>0.1']

    # logs_names = ['No Mask',
    #               'GT-Mask',
    #               'Active-Mask']

    logs_names = np.array(logs_names[:len(logs)])

    return logs, logs_names


def collider_final_Followers(old_result_limit):
    """
    Nouveau loss, et nouvelles valeur de validation pour ces tests
    """

    # Using the dates of the logs, you can easily gather consecutive ones. All logs should be of the same dataset.
    start = 'Log_2021-06-04_11-13-12'
    end = 'Log_2021-08-05_11-13-12'

    if end < old_result_limit:
        res_path = 'old_results'
    else:
        res_path = 'results'

    logs = np.sort([join(res_path, log) for log in listdir(res_path) if start <= log <= end])
    logs = logs.astype('<U50')
    logs = np.insert(logs, 0, 'results/Log_2021-06-01_21-33-07')

    # Give names to the logs (for legends)
    logs_names = ['old-Followers',
                  'Followers-v2-f-loss-10(322)',
                  'Followers-v2-f-loss-1',
                  'Followers-v2-f-loss-100',
                  'Followers-v2-10-2Dnet221',
                  'Followers-v2-10-2Dnet443',
                  'Followers-no-3D-loss',
                  'Followers-shared_weights',
                  'Followers-T=5.0',
                  'Followers-443-loss50.0<0.03',
                  'test']

    logs_names = np.array(logs_names[:len(logs)])

    return logs, logs_names

    
def collider_behaviors(old_result_limit):
    """
    Nouveau loss, et nouvelles valeur de validation pour ces tests
    """

    # Using the dates of the logs, you can easily gather consecutive ones. All logs should be of the same dataset.
    start = 'Log_2021-06-04_11-13-12'
    end = 'Log_2021-06-04_11-13-14'

    if end < old_result_limit:
        res_path = 'old_results'
    else:
        res_path = 'results'

    logs = np.sort([join(res_path, log) for log in listdir(res_path) if start <= log <= end])
    logs = logs.astype('<U50')
    logs = np.insert(logs, 0, 'results/Log_2021-05-24_13-59-19')
    logs = np.insert(logs, 1, 'results/Log_2021-05-24_15-46-47')

    # Give names to the logs (for legends)
    logs_names = ['Bouncers',
                  'Wanderers',
                  'FlowFollowers']

    logs_names = np.array(logs_names[:len(logs)])

    return logs, logs_names

    
# ----------------------------------------------------------------------------------------------------------------------
#
#           Main call
#       \***************/
#


def wanted_Bouncers(chosen_log):

    # Get training and validation days
    config = Config()
    config.load(chosen_log)
    val_path = join(chosen_log, 'val_preds')
    val_days = np.unique([f.split('_')[0] for f in listdir(val_path) if f.endswith('pots.ply')])

    test_dataset = MyhalCollisionDataset(config, val_days, chosen_set='validation', balance_classes=False)
    seq_inds = test_dataset.all_inds[:, 0]
    frame_inds = test_dataset.all_inds[:, 1]

    wanted_s = ['2021-05-15-23-15-09',
                '2021-05-15-23-33-25',
                '2021-05-15-23-33-25',
                '2021-05-15-23-54-50',
                '2021-05-15-23-54-50']
    wanted_f = [300,
                600,
                750,
                100,
                500]

    wanted_inds = []
    for seq, f_i in zip(wanted_s, wanted_f):
        s_i = np.argwhere(val_days == seq)[0][0]
        mask = np.logical_and(seq_inds == s_i, frame_inds == f_i)
        w_i = np.argwhere(mask)[0][0]
        wanted_inds += [w_i - 4, w_i - 2, w_i, w_i + 2, w_i + 4]

    # TMP
    wanted_inds = [1400] + wanted_inds

    return wanted_inds


def wanted_Wanderers(chosen_log):

    # Get training and validation days
    config = Config()
    config.load(chosen_log)
    val_path = join(chosen_log, 'val_preds')
    val_days = np.unique([f.split('_')[0] for f in listdir(val_path) if f.endswith('pots.ply')])

    test_dataset = MyhalCollisionDataset(config, val_days, chosen_set='validation', balance_classes=False)
    seq_inds = test_dataset.all_inds[:, 0]
    frame_inds = test_dataset.all_inds[:, 1]

    wanted_s = ['2021-05-17-14-04-52',
                '2021-05-17-14-04-52',
                '2021-05-17-14-04-52',
                '2021-05-17-14-04-52',
                '2021-05-17-14-04-52',
                '2021-05-17-14-21-56',
                '2021-05-17-14-44-46',
                '2021-05-17-14-44-46']
    wanted_f = [50,
                150,
                200,
                700,
                350,
                50,
                1550]

    wanted_inds = []
    for seq, f_i in zip(wanted_s, wanted_f):
        s_i = np.argwhere(val_days == seq)[0][0]
        mask = np.logical_and(seq_inds == s_i, frame_inds == f_i)
        w_i = np.argwhere(mask)[0][0]
        wanted_inds += [w_i - 4, w_i - 2, w_i, w_i + 2, w_i + 4]

    return wanted_inds


def wanted_Flow1(chosen_log):

    # Get training and validation days
    config = Config()
    config.load(chosen_log)
    val_path = join(chosen_log, 'val_preds')
    val_days = np.unique([f.split('_')[0] for f in listdir(val_path) if f.endswith('pots.ply')])

    test_dataset = MyhalCollisionDataset(config, val_days, chosen_set='validation', balance_classes=False)
    seq_inds = test_dataset.all_inds[:, 0]
    frame_inds = test_dataset.all_inds[:, 1]

    wanted_s = ['2021-05-06-23-59-54',
                '2021-05-06-23-59-54',
                '2021-05-06-23-59-54']
    wanted_f = [1400,
                1450,
                1900]

    wanted_inds = []
    for seq, f_i in zip(wanted_s, wanted_f):
        s_i = np.argwhere(val_days == seq)[0][0]
        mask = np.logical_and(seq_inds == s_i, frame_inds == f_i)
        w_i = np.argwhere(mask)[0][0]
        wanted_inds += [w_i - 4, w_i - 2, w_i, w_i + 2, w_i + 4]

    return wanted_inds


def wanted_Flow2(chosen_log):

    # Get training and validation days
    config = Config()
    config.load(chosen_log)
    val_path = join(chosen_log, 'val_preds')
    val_days = np.unique([f.split('_')[0] for f in listdir(val_path) if f.endswith('pots.ply')])

    test_dataset = MyhalCollisionDataset(config, val_days, chosen_set='validation', balance_classes=False)
    seq_inds = test_dataset.all_inds[:, 0]
    frame_inds = test_dataset.all_inds[:, 1]

    wanted_s = ['2021-06-02-21-09-48',
                '2021-06-02-21-09-48',
                '2021-06-02-21-09-48',
                '2021-06-02-20-33-09',
                '2021-06-02-20-33-09']
    wanted_f = [250,
                700,
                1350, 
                100, 
                350]

    wanted_inds = []
    for seq, f_i in zip(wanted_s, wanted_f):
        s_i = np.argwhere(val_days == seq)[0][0]
        mask = np.logical_and(seq_inds == s_i, frame_inds == f_i)
        w_i = np.argwhere(mask)[0][0]
        wanted_inds += [w_i - 4, w_i - 2, w_i, w_i + 2, w_i + 4]

    return wanted_inds


def wanted_Flow3(chosen_log):

    # Get training and validation days
    config = Config()
    config.load(chosen_log)
    val_path = join(chosen_log, 'val_preds')
    val_days = np.unique([f.split('_')[0] for f in listdir(val_path) if f.endswith('pots.ply')])

    test_dataset = MyhalCollisionDataset(config, val_days, chosen_set='validation', balance_classes=False)
    seq_inds = test_dataset.all_inds[:, 0]
    frame_inds = test_dataset.all_inds[:, 1]

    wanted_s = ['2021-06-02-20-33-09',
                '2021-06-02-20-33-09',
                '2021-06-02-20-33-09',
                '2021-06-02-20-33-09',
                '2021-06-02-20-33-09',
                '2021-06-02-21-09-48',
                '2021-06-02-21-09-48']
    wanted_f = [50,
                300,
                350, 
                560, 
                650, 
                450, 
                550]

    wanted_inds = []
    for seq, f_i in zip(wanted_s, wanted_f):
        s_i = np.argwhere(val_days == seq)[0][0]
        mask = np.logical_and(seq_inds == s_i, frame_inds == f_i)
        w_i = np.argwhere(mask)[0][0]
        wanted_inds += [w_i - 4, w_i - 2, w_i, w_i + 2, w_i + 4]

    return wanted_inds


if __name__ == '__main__':

    ######################################################
    # Choose a list of log to plot together for comparison
    ######################################################

    # Old result limit
    old_res_lim = 'Log_2020-05-04_19-17-59'

    # My logs: choose the logs to show
    # logs, logs_names = collider_final_Bouncers(old_res_lim)
    # wanted_inds = None

    # # FOR BOUNCER VISU (use collider_tests_Followers)
    # logs, logs_names = collider_final_Bouncers(old_res_lim)
    # logs = logs[[2]]
    # logs_names = logs_names[[2]]
    # wanted_inds = wanted_Bouncers(logs[0])

    # # FOR WANDERERS VISU (use collider_tests_Followers)
    # logs, logs_names = collider_final_Wanderers(old_res_lim)
    # logs = logs[[2]]
    # logs_names = logs_names[[2]]
    # wanted_inds = wanted_Wanderers(logs[0])

    # FOR FLOW1 VISU
    logs, logs_names = collider_tests_Followers(old_res_lim)
    logs = logs[[4]]
    logs_names = logs_names[[4]]
    wanted_inds = wanted_Flow1(logs[0])
    
    # # FOR FLOW2 VISU
    # logs, logs_names = collider_behaviors(old_res_lim)
    # logs = logs[[2]]
    # logs_names = logs_names[[2]]
    # wanted_inds = wanted_Flow2(logs[0])
    
    # # FOR FLOW3 VISU
    # logs, logs_names = collider_final_Followers(old_res_lim)
    # logs = logs[[9]]
    # logs_names = logs_names[[9]]
    # print(logs)
    # print(logs_names)
    # wanted_inds = wanted_Flow3(logs[0])


    ######################################################
    # Choose a list of log to plot together for comparison
    ######################################################

    # Check that all logs are of the same dataset. Different object can be compared
    plot_dataset = None
    config = None
    for log in logs:
        config = Config()
        config.load(log)
        this_dataset = config.dataset
        if plot_dataset:
            if plot_dataset == this_dataset:
                continue
            else:
                raise ValueError('All logs must share the same dataset to be compared')
        else:
            plot_dataset = this_dataset

    ################
    # Plot functions
    # ##############

    # plotting = 'evo'
    plotting = 'gifs'
    # plotting = 'PR'
    # plotting = 'conv'

    if plotting == 'evo':
        # Evolution of predictions from checkpoints to checkpoints
        evolution_gifs(logs[1])
        
    elif plotting == 'gifs':
        # Comparison of last checkpoints of each logs
        comparison_gifs(logs, wanted_inds=wanted_inds)
        #comparison_gifs(logs[[2]])

    elif plotting == 'PR':
        # Comparison of the performances with good metrics
        comparison_metrics(logs, logs_names)
        #comparison_metrics(logs[[1, 8]], logs_names[[1, 8]])

    else:

        # Plot the training loss and accuracy
        compare_trainings(logs, logs_names, smooth_epochs=3.0)

        # Plot the validation
        if config.dataset_task == 'collision_prediction':
            if config.dataset.startswith('MyhalCollision'):
                compare_convergences_collision2D(logs, logs_names, smooth_n=20)
        else:
            raise ValueError('Unsupported dataset : ' + plot_dataset)
