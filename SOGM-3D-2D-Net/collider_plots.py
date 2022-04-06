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


def print_logs_val_table(logs_names, all_val_days, log_val_days):

    # Create a table with the presence of validation days for each log
    n_fmt0 = np.max([len(log_name) for log_name in logs_names]) + 2
    lines = ['{:^{width}s}|'.format('     \\  Val', width=n_fmt0)]
    lines += ['{:^{width}s}|'.format('Logs  \\    ', width=n_fmt0)]
    lines += ['{:-^{width}s}|'.format('', width=n_fmt0)]
    for log_i, log in enumerate(logs_names):
        lines += ['{:^{width}s}|'.format(logs_names[log_i], width=n_fmt0)]

    n_fmt1 = 7
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
                
    for line_str in lines:
        print(line_str)
    
    return


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


def cleanup(res_path, max_clean_date, remove_tmp_test=False):

    # Removing data:
    #   > all checkpoints except last
    #   > all future_visu
    #   > all val_preds
    #   > (option) all test temp data

    # List results folders
    res_folders = np.sort([f for f in listdir(res_path) if f.startswith('Log_')])

    # Only consider folder up to max_clean_date
    res_folders = res_folders[res_folders < max_clean_date]

    for res_folder in res_folders:

        print('Erasing useless data for:', res_folder)

        # checkpoints
        chkp_path = join(res_path, res_folder, 'checkpoints')

        # Remove 'current_chkp.tar'
        current_chkp = join(chkp_path, 'current_chkp.tar')
        if exists(current_chkp):
            remove(current_chkp)

        # List checkpoints, keep last one
        chkps = np.sort([join(chkp_path, f) for f in listdir(chkp_path) if f.endswith('.tar')])
        for chkp in chkps[:-1]:
            remove(chkp)

        # Remove unused folders
        removed_folders = [join(res_path, res_folder, 'future_visu'),
                           join(res_path, res_folder, 'val_preds')]

        if remove_tmp_test:
            removed_folders += [join(res_path, res_folder, 'test_metrics'),
                                join(res_path, res_folder, 'test_visu')]

        for removed_folder in removed_folders:
            if (os.path.isdir(removed_folder)):
                shutil.rmtree(removed_folder)

    return

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
    all_mean_epoch_n = []
    all_batch_num = []

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
        t = np.array(t, dtype=np.float32)

        # Compute number of steps per epoch
        max_e = np.max(epochs)
        first_e = np.min(epochs)
        epoch_n = []
        for i in range(first_e, max_e):
            bool0 = epochs == i
            e_n = np.sum(bool0)
            epoch_n.append(e_n)
            epochs_d[bool0] += steps[bool0] / e_n
        mean_epoch_n = np.mean(epoch_n)
        smooth_n = int(mean_epoch_n * smooth_epochs)
        smooth_loss = running_mean(L_out, smooth_n, stride=stride)
        all_loss += [smooth_loss]
        if L_2D_init:
            all_loss2 += [running_mean(L_2D_init, smooth_n, stride=stride)]
            all_loss3 += [running_mean(L_2D_prop, smooth_n, stride=stride)]
            all_loss1 += [all_loss[-1] - all_loss2[-1] - all_loss3[-1]]
        all_epochs += [epochs_d[smooth_n:-smooth_n:stride]]
        all_times += [t[smooth_n:-smooth_n:stride]]
        all_mean_epoch_n += [mean_epoch_n]
        all_batch_num += [config.batch_num]

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

        fig3, axes3 = plt.subplots(1, 3, sharey=False, figsize=(12, 5))

        plots = []
        for i, label in enumerate(list_of_labels):
            pl_list = []
            pl_list += axes3[0].plot(all_epochs[i], all_loss1[i], linewidth=1, label=label)
            pl_list += axes3[1].plot(all_epochs[i], all_loss2[i], linewidth=1, label=label)
            pl_list += axes3[2].plot(all_epochs[i], all_loss3[i], linewidth=1, label=label)
            plots.append(pl_list)

        # Set names for axes
        for ax in axes3:
            ax.set_xlabel('epochs')
        axes3[0].set_ylabel('loss')
        axes3[0].set_yscale('log')

        # Display legends and title
        axes3[2].legend(loc=1)
        axes3[0].set_title('3D_net loss')
        axes3[1].set_title('2D_init loss')
        axes3[2].set_title('2D_prop loss')

        # Customize the graph
        for ax in axes3:
            ax.grid(linestyle='-.', which='both')

        # X-axis controller
        plt.subplots_adjust(left=0.14)

        axcolor = 'lightgrey'
        rax = plt.axes([0.02, 0.4, 0.1, 0.2], facecolor=axcolor)
        radio = RadioButtons(rax, ('epochs', 'steps', 'examples', 'time'))

    
        def x_func(label):
            if label == 'epochs':
                for i, label in enumerate(list_of_labels):
                    for pl in plots[i]:
                        pl.set_xdata(all_epochs[i])
            if label == 'steps':
                for i, label in enumerate(list_of_labels):
                    for pl in plots[i]:
                        all_mean_epoch_n
                        pl.set_xdata(all_epochs[i] * all_mean_epoch_n[i])
            if label == 'examples':
                for i, label in enumerate(list_of_labels):
                    for pl in plots[i]:
                        pl.set_xdata(all_epochs[i] * all_mean_epoch_n[i] * all_batch_num[i])
            if label == 'time':
                for i, label in enumerate(list_of_labels):
                    for pl in plots[i]:
                        pl.set_xdata(all_times[i] / 3600)
            
            for ax in axes3:
                ax.relim()
                ax.autoscale_view()
            plt.draw()

        radio.on_clicked(x_func)


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
        plt.plot(all_epochs[i], all_times[i] / 3600, linewidth=1, label=label)

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


def evolution_gifs(chosen_log, dataset_path='RealMyhal'):

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
    val_days = np.unique([f[:19] for f in listdir(val_path) if f.endswith('pots.ply')])

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
    test_dataset = MyhalCollisionDataset(config, val_days, chosen_set='validation', dataset_path=dataset_path, balance_classes=False)

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
                    i_frame0 = config.n_frames - 1
                    img0 = stck_init_preds[b_i, 0, :, :, :]
                    gt_im0 = np.copy(stck_future_gts[b_i, i_frame0, :, :, :])
                    gt_im1 = stck_future_gts[b_i, i_frame0, :, :, :]
                    gt_im1[:, :, 2] = np.max(stck_future_gts[b_i, i_frame0:, :, :, 2], axis=0)
                    img1 = stck_init_preds[b_i, 1, :, :, :]
                    s_ind = f_inds[b_i, 0]
                    f_ind = f_inds[b_i, 1]

                    # Get the 2D predictions and gt (prop_2D)
                    img = stck_future_preds[b_i, :, :, :, :]
                    gt_im = stck_future_gts[b_i, (i_frame0+1):, :, :, :]

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


def comparison_gifs(list_of_paths, list_of_names, real_val_days, sim_val_days, dataset_path, sim_path, wanted_inds=[], wanted_chkp=[], redo=False):

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
            test_dataset = MyhalCollisionDataset(config,
                                                 real_val_days,
                                                 chosen_set='validation',
                                                 dataset_path=dataset_path,
                                                 add_sim_path=sim_path,
                                                 add_sim_days=sim_val_days,
                                                 balance_classes=False)

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
            print(log_name)
            print('*' * len(log_name))
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
                        chkp_preds = np.stack(chkp_preds, axis=0)

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


    # ####################### DEBUG #######################
    # #
    # # Show the diffusing function here
    # #

    # for frame_i, w_i in enumerate(wanted_inds):
    #     collision_risk = comparison_preds[0][frame_i]

    #     fig1, anim1 = show_local_maxima(collision_risk[..., 2], neighborhood_size=5, threshold=0.1, show=False)
    #     fig2, anim2 = show_risk_diffusion(collision_risk, dl=0.12, diff_range=2.5, show=False)
    #     plt.show()

    # a = 1/0

    # #
    # #
    # ####################### DEBUG #######################

    #############
    # Preparation
    #############
    
    test_dataset = MyhalCollisionDataset(config,
                                         real_val_days,
                                         chosen_set='validation',
                                         dataset_path=dataset_path,
                                         add_sim_path=sim_path,
                                         add_sim_days=sim_val_days,
                                         balance_classes=False)
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
        # merged_imgs = superpose_gt(showed_preds, showed_gts, showed_ingts)
        merged_imgs = superpose_gt_contour(showed_preds, showed_gts, showed_ingts, no_in=True)
        
        # # To show gt images
        # showed_preds = np.copy(showed_gts)
        # showed_preds[..., 2] *= 0.6
        # merged_imgs = superpose_gt(showed_preds, showed_gts * 0, showed_ingts, ingts_fade=(100, -5))

        # Reverse image height axis so that imshow is consistent with plot
        merged_imgs = merged_imgs[:, :, ::-1, :, :]

        all_merged_imgs.append(merged_imgs)

        print('', end='\r')
        print(fmt_str.format('#' * (((frame_i + 1) * progress_n) // N), 100 * (frame_i + 1) / N), end='', flush=True)

    # Show a nice 100% progress bar
    print('', end='\r')
    print(fmt_str.format('#' * progress_n, 100), flush=True)
    print('\n')

    c_showed = np.arange(all_merged_imgs[0].shape[0])
    n_showed = len(c_showed)
    frame_i = 0

    fig, axes = plt.subplots(1, n_showed)
    if n_showed == 1:
        axes = [axes]

    images = []
    for ax_i, log_i in enumerate(c_showed):

        # Init plt
        images.append(axes[ax_i].imshow(all_merged_imgs[frame_i][log_i, 0]))
        # plt.axis('off')

        axes[ax_i].text(.5, -0.02, logs_names[log_i],
                        horizontalalignment='center',
                        verticalalignment='top',
                        transform=axes[ax_i].transAxes)

        axes[ax_i].axis('off')

        # # Save gif for the videos
        # seq_name = test_dataset.sequences[wanted_s_inds[frame_i]]
        # frame_name = test_dataset.frames[wanted_s_inds[frame_i]][wanted_f_inds[frame_i]]

        # sogm_folder = join(test_dataset.path, 'sogm_preds/Log_{:s}'.format(seq_name))
        # if not exists(sogm_folder):
        #     makedirs(sogm_folder)
        # im_name = join(sogm_folder, 'gif_{:s}_{:s}_{:d}.gif'.format(seq_name, frame_name, ax_i))
        # imageio.mimsave(im_name, all_merged_imgs[frame_i][log_i], fps=20)
      
    seq_name = test_dataset.sequences[wanted_s_inds[frame_i]]
    frame_name = test_dataset.frames[wanted_s_inds[frame_i]][wanted_f_inds[frame_i]]
    title = [fig.suptitle('Example {:d}/{:d} > {:s} {:s}'.format(frame_i + 1, len(all_merged_imgs), seq_name, frame_name))]

    # Make a vertically oriented slider to control the amplitude
    Nt = all_merged_imgs[frame_i].shape[-4] - 1
    allowed_times = (np.arange(Nt, dtype=np.float32) + 1) * config.T_2D / config.n_2D_layers
    time_step = config.T_2D / config.n_2D_layers
    axslide = plt.axes([0.02, 0.2, 0.01, 0.6])
    time_slider = Slider(ax=axslide,
                         label="T",
                         valmin=allowed_times[0],
                         valmax=allowed_times[-1],
                         valinit=allowed_times[0],
                         orientation="vertical")

    # The function to be called anytime a slider's value changes
    def slider_update(val, stop_loop=True):
        nonlocal frame_i, loop_running
        if stop_loop and loop_running:
            anim.event_source.stop()
            loop_running = False
        t_i = int(np.floor(val / time_step))
        for ax_i, log_i in enumerate(c_showed):
            images[ax_i].set_array(all_merged_imgs[frame_i][log_i, t_i])
        return images
        
    # register the update function with each slider
    time_slider.on_changed(slider_update)

    loop_running = True

    def animate(i):
        nonlocal frame_i
        for ax_i, log_i in enumerate(c_showed):
            images[ax_i].set_array(all_merged_imgs[frame_i][log_i, i])
        return images
        
    # Register event
    def onkey(event):
        nonlocal frame_i, loop_running
        
        if event.key == 'right':
            frame_i = (frame_i + 1) % len(all_merged_imgs)
            slider_update(0, stop_loop=False)

        elif event.key == 'left':
            frame_i = (frame_i - 1) % len(all_merged_imgs)
            slider_update(0, stop_loop=False)

        elif event.key == 'enter':
            if loop_running:
                anim.event_source.stop()
                loop_running = False
            else:
                anim.event_source.start()
                loop_running = True

        if event.key in ['right', 'left']:
                
            seq_name = test_dataset.sequences[wanted_s_inds[frame_i]]
            frame_name = test_dataset.frames[wanted_s_inds[frame_i]][wanted_f_inds[frame_i]]
            title[0].set_text('Example {:d}/{:d} > {:s} at {:s}'.format(frame_i + 1, len(all_merged_imgs), seq_name, frame_name))
            plt.draw()


        if event.key in ['s', 'S', 'g', 'G']:
            print('Saving in progress')
            
            # Save current gif
            seq_name = test_dataset.sequences[wanted_s_inds[frame_i]]
            frame_name = test_dataset.frames[wanted_s_inds[frame_i]][wanted_f_inds[frame_i]]
                    
            for log_i in c_showed:

                sogm_folder = join(test_dataset.path, 'sogm_preds')
                print(sogm_folder)
                if not exists(sogm_folder):
                    makedirs(sogm_folder)
                im_name = join(sogm_folder, 'gif_{:s}_{:s}_{:d}.gif'.format(seq_name, frame_name, log_i))
                imageio.mimsave(im_name, all_merged_imgs[frame_i][log_i], fps=20)
                    
                gif_folder = join(visu_paths[log_i], 'gifs')
                if not exists(gif_folder):
                    makedirs(gif_folder)
                im_name = join(gif_folder, 'gif_{:s}_{:s}.gif'.format(seq_name, frame_name))
                imageio.mimsave(im_name, all_merged_imgs[frame_i][log_i], fps=20)
            print('Done')

        return title

    anim = FuncAnimation(fig, animate,
                         frames=np.arange(all_merged_imgs[frame_i].shape[1]),
                         interval=50,
                         blit=True,
                         repeat=True,
                         repeat_delay=500)

    cid = fig.canvas.mpl_connect('key_press_event', onkey)
    print('\n---------------------------------------\n')
    print('Instructions:\n')
    print('> Use right and left arrows to navigate among examples.')
    print('> Use enter to start/stop animation loop.')
    print('> Use "g" to save as gif.')
    print('\n---------------------------------------\n')

    plt.show()



    return


def comparison_metrics(list_of_paths, list_of_names, real_val_days, sim_val_days, dataset_path='RealMyhal', sim_path='Simulation', wanted_chkps=[]):

    ############
    # Parameters
    ############

    if list_of_names is None:
        list_of_names = [str(i) for i in range(len(list_of_paths))]

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

        # Only deal with one checkpoint (for faster computations)
        if len(wanted_chkps) < 1:
            chkps = chkps[-2:-1]
        else:
            chkps = chkps[wanted_chkps]
            
        all_chkps.append(chkps)

        # Get the chkp_inds
        chkp_inds = [int(f[:-4].split('_')[-1]) for f in chkps]

        # # Get training and validation days
        # val_path = join(chosen_log, 'val_preds')
        # val_days = np.unique([f[:19] for f in listdir(val_path) if f.endswith('pots.ply')])

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

        # Dataset
        test_dataset = MyhalCollisionDataset(config,
                                             real_val_days,
                                             chosen_set='validation',
                                             dataset_path=dataset_path,
                                             add_sim_path=sim_path,
                                             add_sim_days=sim_val_days,
                                             balance_classes=False)

        # List all precomputed preds:
        saved_res = np.sort([f for f in listdir(visu_path) if f.startswith('metrics_chkp') and f.endswith('.pkl')])
        saved_res_inds = [int(f[:-4].split('_')[-1]) for f in saved_res]

        # # REDO
        # if i_chosen_log >= len(list_of_paths) - 20:
        #     saved_res_inds = []

        # List of the chkp to do
        todo_inds = [ind for ind in chkp_inds if ind not in saved_res_inds]
        to_load_inds = [ind for ind in chkp_inds if ind in saved_res_inds]

        # Results
        all_TP_FP_FN = [None for ind in chkp_inds]
        all_MSE = [None for ind in chkp_inds]

        # Load if available
        if len(to_load_inds) > 0:

            print('\nFound previous predictions, loading them')

            for chkp_i, chkp in enumerate(chkps):

                if chkp_inds[chkp_i] not in to_load_inds:
                    continue

                # Load preds for this chkp
                chkp_stat_file = join(visu_path, 'metrics_chkp_{:04d}.pkl'.format(chkp_inds[chkp_i]))
                with open(chkp_stat_file, 'rb') as rfile:
                    chkp_TP_FP_FN = pickle.load(rfile)

                chkp_mse_file = join(visu_path, 'mse_chkp_{:04d}.pkl'.format(chkp_inds[chkp_i]))
                with open(chkp_mse_file, 'rb') as rfile:
                    chkp_MSE = pickle.load(rfile)
                   
                # Store all predictions 
                all_TP_FP_FN[chkp_i] = chkp_TP_FP_FN
                all_MSE[chkp_i] = chkp_MSE

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
                            res_TP_FP_FN = fast_threshold_stats(gt_flat, p_flat, n_thresh=PR_resolution)

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

                # Store all predictions
                chkp_TP_FP_FN = np.concatenate(chkp_TP_FP_FN, axis=0)
                chkp_MSE = np.concatenate(chkp_MSE, axis=0)
                all_TP_FP_FN[chkp_i] = chkp_TP_FP_FN
                all_MSE[chkp_i] = chkp_MSE
                
                # Save preds for this chkp
                chkp_stat_file = join(visu_path, 'metrics_chkp_{:04d}.pkl'.format(chkp_inds[chkp_i]))
                with open(chkp_stat_file, 'wb') as wfile:
                    pickle.dump(np.copy(chkp_TP_FP_FN), wfile)

                chkp_mse_file = join(visu_path, 'mse_chkp_{:04d}.pkl'.format(chkp_inds[chkp_i]))
                with open(chkp_mse_file, 'wb') as wfile:
                    pickle.dump(np.copy(chkp_MSE), wfile)

            # Free cuda memory
            torch.cuda.empty_cache()

        # All TP_FP_FN shape: [chkp_n, frames_n, T, nt, 3]
        all_TP_FP_FN = np.stack(all_TP_FP_FN, axis=0)
        all_MSE = np.stack(all_MSE, axis=0)
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
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        plt.subplots_adjust(bottom=0.2)

        # Init threshold
        min_dt = np.min(dts)
        max_t = np.max(horizons)
        allowed_times = np.arange(min_dt, max_t + 0.1 * min_dt, min_dt)
        
        # for log_i, log_t in enumerate(log_times):
        #     t = horizons[log_i]
        #     dt = dts[log_i]
        #     log_t = np.arange(dt, t + 0.1 * dt, dt)
            
        time_step = min_dt
        time_ind_0 = 9
        time_0 = allowed_times[time_ind_0]

        # Plot last PR curve for each log
        real_plotsA = []
        real_preA = []
        real_recA = []
        sim_plotsA = []
        sim_preA = []
        sim_recA = []
        for i, name in enumerate(list_of_names):

            # log_n * [chkp_n, frames_n, T, nt, 3] => [frames_n, T, nt, 3]
            all_TP_FP_FN = comparison_TP_FP_FN[i][-1]

            # All stats from real and sim sequences [T, nt, 3]
            real_TP_FP_FN = np.sum(all_TP_FP_FN[np.logical_not(is_sim)], axis=0)
            sim_TP_FP_FN = np.sum(all_TP_FP_FN[is_sim], axis=0)

            for chosen_TP_FP_FN, preA, recA, ax, plotsA in zip([real_TP_FP_FN, sim_TP_FP_FN], [real_preA, sim_preA], [real_recA, sim_recA], axes, [real_plotsA, sim_plotsA]):
                tps = chosen_TP_FP_FN[..., 0]
                fps = chosen_TP_FP_FN[..., 1]
                fns = chosen_TP_FP_FN[..., 2]

                pre = tps / (fns + tps + 1e-6)
                rec = tps / (fps + tps + 1e-6)

                plotsA.append(ax.plot(rec[time_ind_0], pre[time_ind_0], linewidth=1, label=name)[0])
                preA.append(pre)
                recA.append(rec)
        
        # Customize the graph
        for axA in axes:
            axA.grid(linestyle='-.', which='both')
            axA.set_xlim(0, 1)
            axA.set_ylim(0, 1)
            axA.set_xlabel('recall')
            axA.set_ylabel('precision')

        # Display legends and title
        plt.legend()

        #################
        # Slider for time
        #################
        
        # Make a horizontal slider to control the frequency.
        axcolor = 'lightgoldenrodyellow'
        axtime = plt.axes([0.2, 0.04, 0.6, 0.02], facecolor=axcolor)
        time_slider = Slider(ax=axtime,
                             label='Time',
                             valmin=np.min(allowed_times),
                             valmax=np.max(allowed_times),
                             valinit=time_0,
                             valstep=time_step)

        # The function to be called anytime a slider's value changes
        def update_PR(val):
            for log_i, _ in enumerate(log_times):
                time_ind = (int)(np.round(val * n_2D_layers[log_i] / horizons[log_i])) - 1
                for preA, recA, plotsA in zip([real_preA, sim_preA], [real_recA, sim_recA], [real_plotsA, sim_plotsA]):
                    if time_ind < preA[log_i].shape[0]:
                        plotsA[log_i].set_xdata(recA[log_i][time_ind])
                        plotsA[log_i].set_ydata(preA[log_i][time_ind])

        # register the update function with each slider
        time_slider.on_changed(update_PR)

        #################################
        # Radio Button for chkp selection
        #################################
        
        # plt.subplots_adjust(left=0.14)

        # axcolor = 'lightgrey'
        # rax = plt.axes([0.02, 0.4, 0.1, 0.2], facecolor=axcolor)
        # radio = RadioButtons(rax, ('epochs', 'steps', 'examples', 'time'))

    
        # def x_func(label):
        #     if label == 'epochs':
        #         for i, label in enumerate(list_of_labels):
        #             for pl in plots[i]:
        #                 pl.set_xdata(all_epochs[i])
        #     if label == 'steps':
        #         for i, label in enumerate(list_of_labels):
        #             for pl in plots[i]:
        #                 all_mean_epoch_n
        #                 pl.set_xdata(all_epochs[i] * all_mean_epoch_n[i])
        #     if label == 'examples':
        #         for i, label in enumerate(list_of_labels):
        #             for pl in plots[i]:
        #                 pl.set_xdata(all_epochs[i] * all_mean_epoch_n[i] * all_batch_num[i])
        #     if label == 'time':
        #         for i, label in enumerate(list_of_labels):
        #             for pl in plots[i]:
        #                 pl.set_xdata(all_times[i] / 3600)
            
        #     for ax in axes3:
        #         ax.relim()
        #         ax.autoscale_view()
        #     plt.draw()

        # radio.on_clicked(x_func)









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
        
        print('\n')
        print('Table of results:')
        print('*****************\n')
        
        s = '{:^35}'.format('')
        s += '{:^42}'.format('Real sequences')
        s += '    '
        s += '{:^42}'.format('Simulation sequences')
        print(s)

        s = '{:^35}'.format('log')
        s += '{:^10}'.format('AP_1')
        s += '{:^10}'.format('AP_2')
        s += '{:^10}'.format('mAP')
        s += '{:^12}'.format('MSE')
        s += '    '

        s += '{:^10}'.format('AP_1')
        s += '{:^10}'.format('AP_2')
        s += '{:^10}'.format('mAP')
        s += '{:^12}'.format('MSE')
        print(s)

        # Figure
        figC, axesC = plt.subplots(1, 2, figsize=(9, 4))

        # Plot last PR curve for each log
        for i, name in enumerate(list_of_names):

            # [frames_n, T, nt, 3]
            all_TP_FP_FN = comparison_TP_FP_FN[i][-1]

            # Init x-axis values
            times = log_times[i]
            
            # All stats from real and sim sequences [T, nt, 3]
            real_TP_FP_FN = np.sum(all_TP_FP_FN[np.logical_not(is_sim)], axis=0)
            sim_TP_FP_FN = np.sum(all_TP_FP_FN[is_sim], axis=0)

            # Chosen timestamps [nt, T, 3]
            real_TP_FP_FN = np.transpose(real_TP_FP_FN, (1, 0, 2))
            sim_TP_FP_FN = np.transpose(sim_TP_FP_FN, (1, 0, 2))

            # MSE [T]
            real_MSE = np.mean(comparison_MSE[i][-1, np.logical_not(is_sim), :], axis=0)
            sim_MSE = np.mean(comparison_MSE[i][-1, is_sim, :], axis=0)
            
            s = ''
            for ax_i, (chosen_TP_FP_FN, MSE, ax) in enumerate(zip([real_TP_FP_FN, sim_TP_FP_FN], [real_MSE, sim_MSE], axesC)):
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

             
                
                if ax_i == 0:
                    s += '{:^35}'.format(name)
                else:
                    s += '    '

                ind1 = np.argmin(np.abs(times - 1.0))
                ind2 = np.argmin(np.abs(times - 2.0))
                ind3 = np.argmin(np.abs(times - np.min(horizons)))

                s += '{:^10.5f}'.format(100*AP[ind1])
                s += '{:^10.5f}'.format(100*AP[ind2])
                s += '{:^10.5f}'.format(100*np.mean(AP[:ind3]))
                s += '{:^12.5f}'.format(10000 * np.mean(MSE))

                ax.plot(times[:-1], AP[:-1], linewidth=1, label=name)
                #ax.plot(times, f1s[best_mean], linewidth=1, label=name)

            print(s)

                
        for ax in axesC:
            ax.grid(linestyle='-.', which='both')
            # axA.set_ylim(0, 1)
            ax.set_xlabel('Time Layer in SOGM (sec)')
            ax.set_ylabel('AP')

        # Display legends and title
        plt.legend()

        fname = 'results/AP_fig.pdf'
        plt.savefig(fname,
                    bbox_inches='tight')



    plt.show()

    return


# ----------------------------------------------------------------------------------------------------------------------
#
#           Choice of visualized gif
#       \******************************/
#


def wanted_gifs(chosen_log, real_val_days, sim_val_days, dataset_path, sim_path, adding_extra=False, all_wanted_s=[], all_wanted_f=[]):

    # Get training and validation days
    config = Config()
    config.load(chosen_log)
    # val_path = join(chosen_log, 'val_preds')
    # val_days = np.unique([f[:19] for f in listdir(val_path) if f.endswith('pots.ply')])

    test_dataset = MyhalCollisionDataset(config,
                                         real_val_days,
                                         chosen_set='validation',
                                         dataset_path=dataset_path,
                                         add_sim_path=sim_path,
                                         add_sim_days=sim_val_days,
                                         balance_classes=False)
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


# ----------------------------------------------------------------------------------------------------------------------
#
#           Choice of log to show
#       \***************************/
#


def your_logs():
    """
    """

    # Using the dates of the logs, you can easily gather consecutive ones. All logs should be of the same dataset.
    start = 'Log_2022-01-04_01-16-23'
    end = 'Log_2022-01-10_18-04-07'

    # Path to the results logs
    res_path = 'results'

    # Gathering names
    logs = np.sort([join(res_path, log) for log in listdir(res_path) if start <= log <= end])

    # Optinally add some specific folder that is not between start and end
    #logs = np.insert(logs, 0, 'results/Log_2021-05-27_17-20-02')
    logs = logs.astype('<U50')

    # Give names to the logs (for legends)
    logs_names = ['e200-rot',
                  'e500-no_rot',
                  'e500-rot-npr=0.5',
                  'etc',
                  'etc']

    # logs_names = ['No Mask',
    #               'GT-Mask',
    #               'Active-Mask']

    logs_names = np.array(logs_names[:len(logs)])

    return logs, logs_names


def Myhal_logs():
    """
    From this point, we modified the annotation process to remove noise
    We also changed the training/validation set
    """

    # Using the dates of the logs, you can easily gather consecutive ones. All logs should be of the same dataset.
    start = 'Log_2022-01-10_18-04-08'
    end = 'Log_2022-01-13_20-41-33'

    # Path to the results logs
    res_path = 'results'

    # Gathering names
    logs = np.sort([join(res_path, log) for log in listdir(res_path) if start <= log <= end])

    # Optinally add some specific folder that is not between start and end
    #logs = np.insert(logs, 0, 'results/Log_2021-05-27_17-20-02')
    logs = logs.astype('<U50')

    # Give names to the logs (for legends)
    logs_names = ['e500-rot',
                  'e200-rot',
                  'e200-norot',
                  'e200-norot-new_train_inds',
                  'e200-rot-new_train_inds',
                  'etc']

    # logs_names = ['No Mask',
    #               'GT-Mask',
    #               'Active-Mask']

    logs_names = np.array(logs_names[:len(logs)])

    return logs, logs_names


def Sim2Myhal_logs():
    """
    From this point, we modified the training to include simulation frames
    """

    # Using the dates of the logs, you can easily gather consecutive ones. All logs should be of the same dataset.
    start = 'Log_2022-01-13_20-41-34'
    end = 'Log_2022-01-19_18-54-31'

    # Path to the results logs
    res_path = 'results'

    # Gathering names
    logs = np.sort([join(res_path, log) for log in listdir(res_path) if start <= log <= end])

    # Optinally add some specific folder that is not between start and end
    #logs = np.insert(logs, 0, 'results/Log_2021-05-27_17-20-02')
    logs = logs.astype('<U50')

    # Give names to the logs (for legends)
    logs_names = ['e200-rot',
                  'e500-rot',
                  'e500-rot-noise=0.1',
                  'e500-rot-noise=0.05',
                  'e500-rot-noise=0.02',
                  'etc']

    # logs_names = ['No Mask',
    #               'GT-Mask',
    #               'Active-Mask']

    logs_names = np.array(logs_names[:len(logs)])
    
    # Copy here the indices you selected with gui
    # all_wanted_s = []
    # all_wanted_f = []
    all_wanted_s = ['2021-12-10_13-06-09',
                    '2021-12-10_13-06-09',
                    '2021-12-10_13-06-09',
                    '2021-12-10_13-06-09',
                    '2021-12-10_13-06-09',
                    '2021-12-13_18-16-27',
                    '2021-12-13_18-16-27',
                    '2021-12-13_18-16-27',
                    '2021-12-13_18-16-27',
                    '2021-12-13_18-16-27',
                    '2021-12-13_18-16-27',
                    '2021-12-13_18-16-27',
                    '2021-12-13_18-16-27',
                    '2021-12-13_18-16-27',
                    '2021-12-13_18-16-27',
                    '2021-12-13_18-16-27',
                    '2021-12-13_18-16-27',
                    '2021-12-13_18-16-27',
                    '2021-12-13_18-16-27',
                    '2021-12-13_18-16-27',
                    '2021-12-15_19-09-57',
                    '2021-12-15_19-09-57',
                    '2021-12-15_19-09-57',
                    '2021-12-15_19-09-57',
                    '2021-06-02-19-55-16',
                    '2021-06-02-20-33-09',
                    '2021-06-02-20-33-09',
                    '2021-06-02-20-33-09',
                    '2021-06-02-20-33-09',
                    '2021-06-02-21-09-48',
                    '2021-06-02-21-09-48']
    all_wanted_f = [1664,
                    1694,
                    1705,
                    1729,
                    1746,
                    560,
                    697,
                    716,
                    736,
                    751,
                    762,
                    777,
                    799,
                    825,
                    904,
                    936,
                    1034,
                    1082,
                    1126,
                    1193,
                    547,
                    591,
                    853,
                    927,
                    776,
                    73,
                    420,
                    461,
                    478,
                    94,
                    122]

    return logs, logs_names, all_wanted_s, all_wanted_f

    
def Sim2Myhal_gif_examples_1():
    """
    In this examples we compare the following:
    - training with old indices
    - training with new indices
    - training with new indices + Simulation examples
    """

    # Using the dates of the logs, you can easily gather consecutive ones. All logs should be of the same dataset.
    start = 'Log_2022-01-10_18-04-08'
    end = 'Log_2022-01-10_18-04-09'

    # Path to the results logs
    res_path = 'results'

    # Gathering names
    logs = np.sort([join(res_path, log) for log in listdir(res_path) if start <= log <= end])

    # Optinally add some specific folder that is not between start and end
    logs = np.insert(logs, 1, 'results/Log_2022-01-11_19-48-52')
    logs = np.insert(logs, 2, 'results/Log_2022-01-13_20-42-47')
    logs = logs.astype('<U50')

    # Give names to the logs (for legends)
    logs_names = ['e500-rot-old',
                  'e200-rot-new',
                  'e500-rot+sim',
                  'etc']

    logs_names = np.array(logs_names[:len(logs)])
    
    # Copy here the indices you selected with gui
    # all_wanted_s = []
    # all_wanted_f = []
    all_wanted_s = ['2021-12-10_13-06-09',
                    '2021-12-10_13-06-09',
                    '2021-12-10_13-06-09',
                    '2021-12-10_13-06-09',
                    '2021-12-10_13-06-09',
                    '2021-12-13_18-16-27',
                    '2021-12-13_18-16-27',
                    '2021-12-13_18-16-27',
                    '2021-12-13_18-16-27',
                    '2021-12-13_18-16-27',
                    '2021-12-13_18-16-27',
                    '2021-12-13_18-16-27',
                    '2021-12-13_18-16-27',
                    '2021-12-13_18-16-27',
                    '2021-12-13_18-16-27',
                    '2021-12-13_18-16-27',
                    '2021-12-13_18-16-27',
                    '2021-12-13_18-16-27',
                    '2021-12-13_18-16-27',
                    '2021-12-13_18-16-27',
                    '2021-12-15_19-09-57',
                    '2021-12-15_19-09-57',
                    '2021-12-15_19-09-57',
                    '2021-12-15_19-09-57',
                    '2021-06-02-19-55-16',
                    '2021-06-02-20-33-09',
                    '2021-06-02-20-33-09',
                    '2021-06-02-20-33-09',
                    '2021-06-02-20-33-09',
                    '2021-06-02-21-09-48',
                    '2021-06-02-21-09-48']
    all_wanted_f = [1664,
                    1694,
                    1705,
                    1729,
                    1746,
                    560,
                    697,
                    716,
                    736,
                    751,
                    762,
                    777,
                    799,
                    825,
                    904,
                    936,
                    1034,
                    1082,
                    1126,
                    1193,
                    547,
                    591,
                    853,
                    927,
                    776,
                    73,
                    420,
                    461,
                    478,
                    94,
                    122]

    return logs, logs_names, all_wanted_s, all_wanted_f


def Controlled_Exp_logs():
    """
    This logs are for the controlled scenario experiment
    """

    # Using the dates of the logs, you can easily gather consecutive ones. All logs should be of the same dataset.
    start = 'Log_2022-01-19_18-54-32'
    end = 'Log_2022-01-26_18-47-29'

    # Path to the results logs
    res_path = 'results'

    # Gathering names
    logs = np.sort([join(res_path, log) for log in listdir(res_path) if start <= log <= end])

    # Optinally add some specific folder that is not between start and end
    #logs = np.insert(logs, 0, 'results/Log_2021-05-27_17-20-02')
    logs = logs.astype('<U50')

    # Give names to the logs (for legends). These logs were all done with e500 and rot augment
    logs_names = ['dl=0.06_mixed',
                  'dl=0.06_real',
                  'dl=0.09_mixed',
                  'dl=0.12_mixed',
                  'dl=0.12_sim',
                  'dl=0.12_mixed(allrandom)',
                  'dl=0.12_real',
                  'dl=0.12_mixed_FAST',
                  'etc']
    
    # Copy here the indices you selected with gui
    # all_wanted_s = []
    # all_wanted_f = []
    all_wanted_s = ['2021-12-10_13-06-09',
                    '2021-12-10_13-06-09',
                    '2021-12-10_13-06-09',
                    '2021-12-13_18-16-27',
                    '2021-12-13_18-16-27',
                    '2021-12-13_18-16-27',
                    '2022-01-18_11-15-40',
                    '2022-01-18_11-15-40',
                    '2022-01-18_11-15-40',
                    '2022-01-18_11-15-40',
                    '2022-01-18_11-15-40',
                    '2022-01-18_11-15-40',
                    '2022-01-18_11-15-40',
                    '2022-01-18_11-15-40',
                    '2022-01-18_11-20-21',
                    '2022-01-18_11-20-21',
                    '2022-01-18_11-20-21',
                    '2022-01-18_11-20-21',
                    '2022-01-18_11-20-21',
                    '2022-01-18_11-20-21',
                    '2021-06-02-20-33-09',
                    '2021-06-02-20-33-09']
    all_wanted_f = [1656,
                    1716,
                    1746,
                    714,
                    743,
                    919,
                    438,
                    462,
                    576,
                    608,
                    619,
                    1110,
                    1153,
                    1355,
                    324,
                    354,
                    1135,
                    1159,
                    1549,
                    1580,
                    74,
                    648]

    logs_names = np.array(logs_names[:len(logs)])

    return logs, logs_names, all_wanted_s, all_wanted_f


def Controlled_Fast_Exp_logs():
    """
    From this point we use fast convergence strategy (decay = 50), dl = 0.12
    """

    # Using the dates of the logs, you can easily gather consecutive ones. All logs should be of the same dataset.
    start = 'Log_2022-01-26_18-47-27'
    end = 'Log_2022-02-25_21-21-56'

    # Path to the results logs
    res_path = 'results'

    # Gathering names
    logs = np.sort([join(res_path, log) for log in listdir(res_path) if start <= log <= end])

    # Optinally add some specific folder that is not between start and end
    #logs = np.insert(logs, 0, 'results/Log_2021-05-27_17-20-02')
    logs = logs.astype('<U50')

    # Give names to the logs (for legends). These logs were all done with e500 and rot augment
    logs_names = ['Real/Sim_50/50',
                  'Real/Sim_85/15',
                  'Real/Sim_70/30',
                  'Real/Sim_30/70',
                  'Real/Sim_15/85',
                  'Sim_only',
                  'Real_only',
                  'Real/Sim_30/70',
                  'etc']

    # Copy here the indices you selected with gui
    # all_wanted_s = []
    # all_wanted_f = []
    all_wanted_s = ['2021-12-10_13-06-09',
                    '2021-12-10_13-06-09',
                    '2021-12-10_13-06-09',
                    '2021-12-13_18-16-27',
                    '2021-12-13_18-16-27',
                    '2021-12-13_18-16-27',
                    '2022-01-18_11-15-40',
                    '2022-01-18_11-15-40',
                    '2022-01-18_11-15-40',
                    '2022-01-18_11-15-40',
                    '2022-01-18_11-15-40',
                    '2022-01-18_11-15-40',
                    '2022-01-18_11-15-40',
                    '2022-01-18_11-15-40',
                    '2022-01-18_11-20-21',
                    '2022-01-18_11-20-21',
                    '2022-01-18_11-20-21',
                    '2022-01-18_11-20-21',
                    '2022-01-18_11-20-21',
                    '2022-01-18_11-20-21',
                    '2021-06-02-20-33-09',
                    '2021-06-02-20-33-09']
    all_wanted_f = [1656,
                    1716,
                    1746,
                    714,
                    743,
                    919,
                    438,
                    462,
                    576,
                    608,
                    619,
                    1110,
                    1153,
                    1355,
                    324,
                    354,
                    1135,
                    1159,
                    1549,
                    1580,
                    74,
                    648]

    logs_names = np.array(logs_names[:len(logs)])

    return logs, logs_names, all_wanted_s, all_wanted_f


def Controlled_v2_logs():
    """
    Here we use v2 of the data, including three more runs for face to face scenario
    """

    # Using the dates of the logs, you can easily gather consecutive ones. All logs should be of the same dataset.
    start = 'Log_2022-02-25_21-21-57'
    end = 'Log_2022-03-02_14-32-20'

    # Path to the results logs
    res_path = 'results'

    # Gathering names
    logs = np.sort([join(res_path, log) for log in listdir(res_path) if start <= log <= end])

    # Optinally add some specific folder that is not between start and end
    #logs = np.insert(logs, 0, 'results/Log_2021-05-27_17-20-02')
    logs = logs.astype('<U50')

    # Give names to the logs (for legends). These logs were all done with e500 and rot augment
    logs_names = ['60/40_3s/30',
                  '60/40_4s/20',
                  '60/40_5s/25',
                  '60/40_4s/40',
                  '60/40_5s/50',
                  'etc']

    # Copy here the indices you selected with gui
    # all_wanted_s = []
    # all_wanted_f = []
    all_wanted_s = ['2021-12-10_13-06-09',
                    '2021-12-10_13-06-09',
                    '2021-12-10_13-06-09',
                    '2021-12-13_18-16-27',
                    '2021-12-13_18-16-27',
                    '2021-12-13_18-16-27',
                    '2022-01-18_11-15-40',
                    '2022-01-18_11-15-40',
                    '2022-01-18_11-15-40',
                    '2022-01-18_11-15-40',
                    '2022-01-18_11-15-40',
                    '2022-01-18_11-15-40',
                    '2022-01-18_11-15-40',
                    '2022-01-18_11-15-40',
                    '2022-01-18_11-20-21',
                    '2022-01-18_11-20-21',
                    '2022-01-18_11-20-21',
                    '2022-01-18_11-20-21',
                    '2022-01-18_11-20-21',
                    '2022-01-18_11-20-21',
                    '2021-06-02-20-33-09',
                    '2021-06-02-20-33-09']
    all_wanted_f = [1656,
                    1716,
                    1746,
                    714,
                    743,
                    919,
                    438,
                    462,
                    576,
                    608,
                    619,
                    1110,
                    1153,
                    1355,
                    324,
                    354,
                    1135,
                    1159,
                    1549,
                    1580,
                    74,
                    648]

    logs_names = np.array(logs_names[:len(logs)])

    return logs, logs_names, all_wanted_s, all_wanted_f


def Myhal1_v1_logs():
    """
    Here we use the data Myhal1 v1
    """

    # Using the dates of the logs, you can easily gather consecutive ones. All logs should be of the same dataset.
    start = 'Log_2022-03-18_17-44-43'
    end = 'Log_2022-03-23_21-06-41'

    # Path to the results logs
    res_path = 'results'

    # Gathering names
    logs = np.sort([join(res_path, log) for log in listdir(res_path) if start <= log <= end])

    # Optinally add some specific folder that is not between start and end
    #logs = np.insert(logs, 0, 'results/Log_2021-05-27_17-20-02')
    logs = logs.astype('<U50')

    # Give names to the logs (for legends). These logs were all done with e500 and rot augment
    logs_names = ['50/50_4s/50',
                  'Real_4s/50',
                  '50/50_4s/40',
                  'Real_4s/40',
                  'etc']

    # Copy here the indices you selected with gui
    # all_wanted_s = []
    # all_wanted_f = []
    all_wanted_s = ['2022-03-09_15-58-56',
                    '2022-03-09_15-58-56',
                    '2022-03-09_15-58-56',
                    '2022-03-09_15-58-56',
                    '2022-03-09_15-58-56',
                    '2022-03-09_15-58-56',
                    '2022-03-09_15-58-56',
                    '2022-03-09_15-58-56',
                    '2022-03-09_16-03-21',
                    '2022-03-09_16-03-21',
                    '2022-03-09_16-03-21',
                    '2022-03-09_16-03-21',
                    '2022-03-09_16-03-21',
                    '2022-03-09_16-03-21',
                    '2022-03-09_16-03-21',
                    '2022-03-09_16-03-21',
                    '2022-03-09_16-03-21',
                    '2022-03-09_16-03-21',
                    '2022-03-09_16-03-21',
                    '2022-03-09_16-03-21',
                    '2022-03-09_16-03-21',
                    '2022-03-09_16-03-21',
                    '2022-03-09_16-03-21',
                    '2022-03-09_16-03-21',
                    '2022-03-09_16-03-21',
                    '2022-03-09_16-03-21',
                    '2022-03-09_16-03-21',
                    '2022-03-09_16-03-21',
                    '2022-03-09_16-03-21',
                    '2022-03-09_16-03-21',
                    '2022-03-09_16-03-21']
    all_wanted_f = [735,
                    357,
                    365,
                    844,
                    883,
                    1077,
                    1819,
                    1926,
                    26,
                    480,
                    527,
                    551,
                    647,
                    859,
                    889,
                    958,
                    1024,
                    1050,
                    1110,
                    1185,
                    1243,
                    1288,
                    1365,
                    1405,
                    1444,
                    1472,
                    1491,
                    1525,
                    1583,
                    1609,
                    1658]


   

    logs_names = np.array(logs_names[:len(logs)])

    return logs, logs_names, all_wanted_s, all_wanted_f


def Myhal1_v2_logs():
    """
    Here we use the data Myhal1 v2 (data until 2022-03-22)
    """

    # Using the dates of the logs, you can easily gather consecutive ones. All logs should be of the same dataset.
    start = 'Log_2022-03-23_21-06-42'
    end = 'Log_2022-05-02_14-32-20'

    # Path to the results logs
    res_path = 'results'

    # Gathering names
    logs = np.sort([join(res_path, log) for log in listdir(res_path) if start <= log <= end])

    # Optinally add some specific folder that is not between start and end
    #logs = np.insert(logs, 0, 'results/Log_2021-05-27_17-20-02')
    logs = logs.astype('<U50')

    # Give names to the logs (for legends). These logs were all done with e500 and rot augment
    logs_names = ['70/30_4s/40',
                  'Real_4s/40',
                  'etc']

    # Copy here the indices you selected with gui
    # all_wanted_s = []
    # all_wanted_f = []
    all_wanted_s = ['2022-03-09_15-58-56',
                    '2022-03-09_15-58-56',
                    '2022-03-09_15-58-56',
                    '2022-03-09_15-58-56',
                    '2022-03-09_16-03-21',
                    '2022-03-09_16-03-21',
                    '2022-03-09_16-03-21',
                    '2022-03-09_16-03-21',
                    '2022-03-09_16-03-21',
                    '2022-03-09_16-03-21',
                    '2022-03-09_16-03-21',
                    '2022-03-09_16-03-21',
                    '2022-03-09_16-03-21',
                    '2022-03-09_16-03-21',
                    '2022-03-09_16-03-21',
                    '2022-03-22_14-12-20',
                    '2022-03-22_14-12-20',
                    '2022-03-22_14-12-20',
                    '2022-03-22_14-12-20',
                    '2022-03-22_16-08-09',
                    '2022-03-22_16-08-09',
                    '2022-03-22_16-08-09',
                    '2022-03-22_16-08-09']
    all_wanted_f = [360,
                    380,
                    842,
                    885,
                    34,
                    523,
                    551,
                    664,
                    868,
                    917,
                    1144,
                    1367,
                    1390,
                    1459,
                    1478,
                    532,
                    668,
                    948,
                    1204,
                    470,
                    620,
                    999,
                    1550]


    logs_names = np.array(logs_names[:len(logs)])

    return logs, logs_names, all_wanted_s, all_wanted_f


# ----------------------------------------------------------------------------------------------------------------------
#
#           Main call
#       \***************/
#


if __name__ == '__main__':

    ##########
    # Clean-up
    ##########
    #
    # Optional. Do it to save space but you will lose some data:
    #   > all checkpoints except last
    #   > all future_visu
    #   > all val_preds
    #   > (option) all test temp data
    #

    cleaning = False
    if cleaning:
        res_path = 'results'
        max_clean_date = 'Log_2022-01-26_18-47-27'
        cleanup(res_path, max_clean_date, remove_tmp_test=False)
    

    ######################################
    # Step 1: Choose what you want to plot
    ######################################

    plotting = 'gifs'  # Comparison of last checkpoints of each logs as gif images

    # plotting = 'PR'  # Comparison of the performances with good metrics

    # plotting = 'conv'  # Convergence of the training sessions (plotting training loss and validation results)


    ##################################################
    # Step 2: Choose which results you want to compare
    ##################################################

    # Function returning the names of the log folders that we want to plot
    logs, logs_names, all_wanted_s, all_wanted_f = Myhal1_v2_logs()


    # Check that all logs are of the same dataset. Different object can be compared
    plot_dataset = None
    config = None
    all_val_days = None
    val_days = None
    log_val_days = []
    for log in logs:
        config = Config()
        config.load(log)
        this_dataset = config.dataset
        val_path = join(log, 'val_preds')
        this_val_days = np.unique([f[:19] for f in listdir(val_path) if f.endswith('pots.ply')])
        log_val_days += [this_val_days]
        if plot_dataset:
            if plot_dataset != this_dataset:
                raise ValueError('All logs must share the same dataset to be compared')
            if len(this_val_days) != len(val_days) or np.any(this_val_days != val_days):
                print('Warning: logs do not share the same validation folders. Comparison is not valid.')
                all_val_days = np.unique(np.hstack((all_val_days, this_val_days)))
        else:
            plot_dataset = this_dataset
            val_days = this_val_days
            all_val_days = this_val_days

    # Get simulatio nvalidation clouds first
    sim_path = '../Data/Simulation'
    sim_val_days = [val_day for val_day in all_val_days
                    if exists(join(sim_path, 'simulated_runs', val_day))]
    all_val_days = [val_day for val_day in all_val_days
                    if not exists(join(sim_path, 'simulated_runs', val_day))]

    # Get dataset path
    dataset_candidates = [join('../Data', path) for path in listdir('../Data')
                          if exists(join('../Data', path, 'runs', all_val_days[0]))]
    if len(dataset_candidates) > 1:
        raise ValueError('Error: Run ' + all_val_days[0] + ' was found in multiple datasets')
    if len(dataset_candidates) < 1:
        raise ValueError('Error: Run ' + all_val_days[0] + ' was not found in any dataset')
    dataset_path = dataset_candidates[0]

    if not np.all([exists(join(dataset_path, 'runs', val_day)) for val_day in all_val_days]):
        raise ValueError('Error: Not all validation folders were found in the dataset path ' + dataset_path)

    print('\n')
    
    print('Real validation')
    print('***************\n')
    print('Path:', dataset_path, '\n')

    print_logs_val_table(logs_names, all_val_days, log_val_days)

    print('\n')
    
    print('\nSimulation validation')
    print('*********************\n')
    print('Path:', sim_path, '\n')

    print_logs_val_table(logs_names, sim_val_days, log_val_days)

    print('\n')
        
    # Function returning the a few specific frames to show as gif
    if 'gifs' in plotting:
    
        print('\nGif Selection')
        print('*************\n')
        wanted_inds = wanted_gifs(logs[0], all_val_days, sim_val_days, dataset_path, sim_path, all_wanted_s=all_wanted_s, all_wanted_f=all_wanted_f)

        print('\n')
    
    print('\nPlotting function')
    print('*****************\n')

    ################
    # Plot functions
    ################
        
    if plotting == 'gifs':
        # Comparison of last checkpoints of each logs
        comparison_gifs(logs, logs_names, all_val_days, sim_val_days, dataset_path, sim_path, wanted_inds=wanted_inds, wanted_chkp=[-1])
        #comparison_gifs(logs[[2]])

    elif plotting == 'gifs-redo':
        # Comparison of last checkpoints of each logs
        comparison_gifs(logs, logs_names, wanted_inds=wanted_inds, dataset_path=dataset_path, sim_path=sim_path, redo=True)
        #comparison_gifs(logs[[2]])

    elif plotting == 'PR':
        # Comparison of the performances with good metrics
        comparison_metrics(logs, logs_names, all_val_days, sim_val_days,
                           dataset_path=dataset_path,
                           sim_path=sim_path,
                           wanted_chkps=[])
        #                  wanted_chkps=[])
        #comparison_metrics(logs[[1, 8]], logs_names[[1, 8]])

    else:

        # Plot the training loss and accuracy
        compare_trainings(logs, logs_names, smooth_epochs=3.0)

        # Plot the validation
        if config.dataset_task == 'collision_prediction':
            if config.dataset.startswith('MyhalCollision'):
                compare_convergences_collision2D(logs, logs_names, smooth_n=30)
        else:
            raise ValueError('Unsupported dataset : ' + plot_dataset)
