#
#
#      0==============================0
#      |    Deep Collision Checker    |
#      0==============================0
#
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      Class handling the test of any model
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
import torch.nn as nn
import os
os.environ.update(OMP_NUM_THREADS='1',
                  OPENBLAS_NUM_THREADS='1',
                  NUMEXPR_NUM_THREADS='1',
                  MKL_NUM_THREADS='1',)
import numpy as np
from os import makedirs, listdir
from os.path import exists, join
import time
import json
from sklearn.neighbors import KDTree

# PLY reader
from utils.ply import read_ply, write_ply

# Metrics
from utils.metrics import IoU_from_confusions, fast_confusion
from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt
from utils.mayavi_visu import show_ModelNet_examples
from models.blocks import LRFBlock

#from utils.visualizer import show_ModelNet_models

# ----------------------------------------------------------------------------------------------------------------------
#
#           Tester Class
#       \******************/
#


class ModelTester:

    # Initialization methods
    # ------------------------------------------------------------------------------------------------------------------

    def __init__(self, net, chkp_path=None, on_gpu=True):

        ############
        # Parameters
        ############

        # Choose to train on CPU or GPU
        if on_gpu and torch.cuda.is_available():
            self.device = torch.device("cuda:0")
            net.to(self.device)
        else:
            self.device = torch.device("cpu")

        ##########################
        # Load previous checkpoint
        ##########################

        if on_gpu and torch.cuda.is_available():
            checkpoint = torch.load(chkp_path)
        else:
            checkpoint = torch.load(chkp_path, map_location={'cuda:0': 'cpu'})
        net.load_state_dict(checkpoint['model_state_dict'])
        self.epoch = checkpoint['epoch']
        net.eval()
        print("\nModel and training state restored from " + chkp_path)

        return

    # Test main methods
    # ------------------------------------------------------------------------------------------------------------------

    def classification_test(self, net, test_loader, config, num_votes=100, debug=False):

        ############
        # Initialize
        ############

        # Choose test smoothing parameter (0 for no smothing, 0.99 for big smoothing)
        softmax = torch.nn.Softmax(1)

        # Number of classes including ignored labels
        nc_tot = test_loader.dataset.num_classes

        # Number of classes predicted by the model
        nc_model = config.num_classes

        # Initiate global prediction over test clouds
        self.test_probs = np.zeros((test_loader.dataset.num_models, nc_model))
        self.test_counts = np.zeros((test_loader.dataset.num_models, nc_model))

        t = [time.time()]
        mean_dt = np.zeros(1)
        last_display = time.time()
        while np.min(self.test_counts) < num_votes:

            # Run model on all test examples
            # ******************************

            # Initiate result containers
            probs = []
            targets = []
            obj_inds = []
            count = 0

            # Start validation loop
            for batch in test_loader:

                # New time
                t = t[-1:]
                t += [time.time()]

                if 'cuda' in self.device.type:
                    batch.to(self.device)

                # Forward pass
                outputs = net(batch, config)

                # Get probs and labels
                probs += [softmax(outputs).cpu().detach().numpy()]
                targets += [batch.labels.cpu().numpy()]
                obj_inds += [batch.model_inds.cpu().numpy()]

                if 'cuda' in self.device.type:
                    torch.cuda.synchronize(self.device)

                # Average timing
                t += [time.time()]
                mean_dt = 0.95 * mean_dt + 0.05 * (np.array(t[1:]) - np.array(t[:-1]))

                # Display
                if (t[-1] - last_display) > 1.0:
                    last_display = t[-1]
                    message = 'Test vote {:.0f} : {:.1f}% (timings : {:4.2f} {:4.2f})'
                    print(message.format(np.min(self.test_counts),
                                         100 * len(obj_inds) / config.validation_size,
                                         1000 * (mean_dt[0]),
                                         1000 * (mean_dt[1])))
            # Stack all validation predictions
            probs = np.vstack(probs)
            targets = np.hstack(targets)
            obj_inds = np.hstack(obj_inds)

            if np.any(test_loader.dataset.input_labels[obj_inds] != targets):
                raise ValueError('wrong object indices')

            # Compute incremental average (predictions are always ordered)
            self.test_counts[obj_inds] += 1
            self.test_probs[obj_inds] += (probs - self.test_probs[obj_inds]) / (self.test_counts[obj_inds])


            # Save/Display temporary results
            # ******************************

            test_labels = np.array(test_loader.dataset.label_values)

            # Compute classification results
            C1 = fast_confusion(test_loader.dataset.input_labels,
                                np.argmax(self.test_probs, axis=1),
                                test_labels)

            ACC = 100 * np.sum(np.diag(C1)) / (np.sum(C1) + 1e-6)
            print('Test Accuracy = {:.1f}%'.format(ACC))

            #
            # if debug:
            #     s = '      '
            #     for ic, cc in enumerate(C1):
            #         s += '{:3d} '.format(ic)
            #     s += '\n      '
            #     for ic, cc in enumerate(C1):
            #         s += '  | '.format(ic)
            #     s += '\n'
            #     for ic, cc in enumerate(C1):
            #         s += '{:3d} - '.format(ic)
            #         for c in cc:
            #             s += '{:3d} '.format(c)
            #         s += '\n'
            #     print(s)
            #
            #     stop = False
            #     while not stop:
            #         print('\nChose actual class and predicted class indices.')
            #         print('Two number between 0 and 39 (separated by a space). Enter anything else to stop.')
            #         in_str = input('Enter indices: ')
            #
            #         if type(in_str) == type('str'):
            #             in_list = [int(s) for s in in_str.split()]
            #             if len(in_list) == 2:
            #                 # Get the points from this confusion cell
            #                 bool1 = test_loader.dataset.input_labels == in_list[0]
            #                 bool2 = np.argmax(self.test_probs, axis=1) == in_list[1]
            #                 cell_inds = np.where(np.logical_and(bool1, bool2))[0]
            #                 cell_points = [test_loader.dataset.input_points[c_ind] for c_ind in cell_inds]
            #                 if cell_inds.shape[0]> 0:
            #                     show_ModelNet_models(cell_points)
            #                 else:
            #                     print('No model in this cell')
            #             else:
            #                 stop = True
            #         else:
            #             stop = True

        return

    def cloud_segmentation_test(self, net, test_loader, config, num_votes=100, debug=False):
        """
        Test method for cloud segmentation models
        """

        ############
        # Initialize
        ############

        # Choose test smoothing parameter (0 for no smothing, 0.99 for big smoothing)
        test_smooth = 0.95
        test_radius_ratio = 0.7
        softmax = torch.nn.Softmax(1)

        # Number of classes including ignored labels
        nc_tot = test_loader.dataset.num_classes

        # Number of classes predicted by the model
        nc_model = config.num_classes

        # Initiate global prediction over test clouds
        self.test_probs = [np.zeros((l.shape[0], nc_model)) for l in test_loader.dataset.input_labels]

        # Test saving path
        if config.saving:
            test_path = join('test', config.saving_path.split('/')[-1])
            if not exists(test_path):
                makedirs(test_path)
            if not exists(join(test_path, 'predictions')):
                makedirs(join(test_path, 'predictions'))
            if not exists(join(test_path, 'probs')):
                makedirs(join(test_path, 'probs'))
            if not exists(join(test_path, 'potentials')):
                makedirs(join(test_path, 'potentials'))
        else:
            test_path = None

        # If on validation directly compute score
        if test_loader.dataset.set == 'validation':
            val_proportions = np.zeros(nc_model, dtype=np.float32)
            i = 0
            for label_value in test_loader.dataset.label_values:
                if label_value not in test_loader.dataset.ignored_labels:
                    val_proportions[i] = np.sum([np.sum(labels == label_value)
                                                 for labels in test_loader.dataset.validation_labels])
                    i += 1
        else:
            val_proportions = None

        #####################
        # Network predictions
        #####################

        test_epoch = 0
        last_min = -0.5

        t = [time.time()]
        last_display = time.time()
        mean_dt = np.zeros(1)

        # Start test loop
        while True:
            print('Initialize workers')
            for i, batch in enumerate(test_loader):

                # New time
                t = t[-1:]
                t += [time.time()]

                if i == 0:
                    print('Done in {:.1f}s'.format(t[1] - t[0]))

                if 'cuda' in self.device.type:
                    batch.to(self.device)

                # Forward pass
                outputs = net(batch, config)

                t += [time.time()]

                # Get probs and labels
                stacked_probs = softmax(outputs).cpu().detach().numpy()
                s_points = batch.points[0].cpu().numpy()
                lengths = batch.lengths[0].cpu().numpy()
                in_inds = batch.input_inds.cpu().numpy()
                cloud_inds = batch.cloud_inds.cpu().numpy()
                torch.cuda.synchronize(self.device)

                # Get predictions and labels per instance
                # ***************************************

                i0 = 0
                for b_i, length in enumerate(lengths):

                    # Get prediction
                    points = s_points[i0:i0 + length]
                    probs = stacked_probs[i0:i0 + length]
                    inds = in_inds[i0:i0 + length]
                    c_i = cloud_inds[b_i]

                    if 0 < test_radius_ratio < 1:
                        mask = np.sum(points ** 2, axis=1) < (test_radius_ratio * config.in_radius) ** 2
                        inds = inds[mask]
                        probs = probs[mask]

                    # Update current probs in whole cloud
                    self.test_probs[c_i][inds] = test_smooth * self.test_probs[c_i][inds] + (1 - test_smooth) * probs
                    i0 += length

                # Average timing
                t += [time.time()]
                if i < 2:
                    mean_dt = np.array(t[1:]) - np.array(t[:-1])
                else:
                    mean_dt = 0.9 * mean_dt + 0.1 * (np.array(t[1:]) - np.array(t[:-1]))

                # Display
                if (t[-1] - last_display) > 1.0:
                    last_display = t[-1]
                    message = 'e{:03d}-i{:04d} => {:.1f}% (timings : {:4.2f} {:4.2f} {:4.2f})'
                    print(message.format(test_epoch, i,
                                         100 * i / config.validation_size,
                                         1000 * (mean_dt[0]),
                                         1000 * (mean_dt[1]),
                                         1000 * (mean_dt[2])))

            # Update minimum of potentials
            new_min = torch.min(test_loader.dataset.min_potentials)
            print('Test epoch {:d}, end. Min potential = {:.1f}'.format(test_epoch, new_min))
            #print([np.mean(pots) for pots in test_loader.dataset.potentials])

            # Save predicted cloud
            if last_min + 1 < new_min:

                # Update last_min
                last_min += 1

                # Show vote results (On subcloud so it is not the good values here)
                if test_loader.dataset.set == 'validation':
                    print('\nConfusion on sub clouds')
                    Confs = []
                    for i, file_path in enumerate(test_loader.dataset.files):

                        # Insert false columns for ignored labels
                        probs = np.array(self.test_probs[i], copy=True)
                        for l_ind, label_value in enumerate(test_loader.dataset.label_values):
                            if label_value in test_loader.dataset.ignored_labels:
                                probs = np.insert(probs, l_ind, 0, axis=1)

                        # Predicted labels
                        preds = test_loader.dataset.label_values[np.argmax(probs, axis=1)].astype(np.int32)

                        # Targets
                        targets = test_loader.dataset.input_labels[i]

                        # Confs
                        Confs += [fast_confusion(targets, preds, test_loader.dataset.label_values)]

                    # Regroup confusions
                    C = np.sum(np.stack(Confs), axis=0).astype(np.float32)

                    # Remove ignored labels from confusions
                    for l_ind, label_value in reversed(list(enumerate(test_loader.dataset.label_values))):
                        if label_value in test_loader.dataset.ignored_labels:
                            C = np.delete(C, l_ind, axis=0)
                            C = np.delete(C, l_ind, axis=1)

                    # Rescale with the right number of point per class
                    C *= np.expand_dims(val_proportions / (np.sum(C, axis=1) + 1e-6), 1)

                    # Compute IoUs
                    IoUs = IoU_from_confusions(C)
                    mIoU = np.mean(IoUs)
                    s = '{:5.2f} | '.format(100 * mIoU)
                    for IoU in IoUs:
                        s += '{:5.2f} '.format(100 * IoU)
                    print(s + '\n')

                # Save real IoU once in a while
                if int(np.ceil(new_min)) % 10 == 0:

                    # Project predictions
                    print('\nReproject Vote #{:d}'.format(int(np.floor(new_min))))
                    t1 = time.time()
                    proj_probs = []
                    for i, file_path in enumerate(test_loader.dataset.files):

                        print(i, file_path, test_loader.dataset.test_proj[i].shape, self.test_probs[i].shape)

                        print(test_loader.dataset.test_proj[i].dtype, np.max(test_loader.dataset.test_proj[i]))
                        print(test_loader.dataset.test_proj[i][:5])

                        # Reproject probs on the evaluations points
                        probs = self.test_probs[i][test_loader.dataset.test_proj[i], :]
                        proj_probs += [probs]

                    t2 = time.time()
                    print('Done in {:.1f} s\n'.format(t2 - t1))

                    # Show vote results
                    if test_loader.dataset.set == 'validation':
                        print('Confusion on full clouds')
                        t1 = time.time()
                        Confs = []
                        for i, file_path in enumerate(test_loader.dataset.files):

                            # Insert false columns for ignored labels
                            for l_ind, label_value in enumerate(test_loader.dataset.label_values):
                                if label_value in test_loader.dataset.ignored_labels:
                                    proj_probs[i] = np.insert(proj_probs[i], l_ind, 0, axis=1)

                            # Get the predicted labels
                            preds = test_loader.dataset.label_values[np.argmax(proj_probs[i], axis=1)].astype(np.int32)

                            # Confusion
                            targets = test_loader.dataset.validation_labels[i]
                            Confs += [fast_confusion(targets, preds, test_loader.dataset.label_values)]

                        t2 = time.time()
                        print('Done in {:.1f} s\n'.format(t2 - t1))

                        # Regroup confusions
                        C = np.sum(np.stack(Confs), axis=0)

                        # Remove ignored labels from confusions
                        for l_ind, label_value in reversed(list(enumerate(test_loader.dataset.label_values))):
                            if label_value in test_loader.dataset.ignored_labels:
                                C = np.delete(C, l_ind, axis=0)
                                C = np.delete(C, l_ind, axis=1)

                        IoUs = IoU_from_confusions(C)
                        mIoU = np.mean(IoUs)
                        s = '{:5.2f} | '.format(100 * mIoU)
                        for IoU in IoUs:
                            s += '{:5.2f} '.format(100 * IoU)
                        print('-' * len(s))
                        print(s)
                        print('-' * len(s) + '\n')

                    # Save predictions
                    print('Saving clouds')
                    t1 = time.time()
                    for i, file_path in enumerate(test_loader.dataset.files):

                        # Get file
                        points = test_loader.dataset.load_evaluation_points(file_path)

                        # Get the predicted labels
                        preds = test_loader.dataset.label_values[np.argmax(proj_probs[i], axis=1)].astype(np.int32)

                        # Save plys
                        cloud_name = file_path.split('/')[-1]
                        test_name = join(test_path, 'predictions', cloud_name)
                        write_ply(test_name,
                                  [points, preds],
                                  ['x', 'y', 'z', 'preds'])
                        test_name2 = join(test_path, 'probs', cloud_name)
                        prob_names = ['_'.join(test_loader.dataset.label_to_names[label].split())
                                      for label in test_loader.dataset.label_values]
                        write_ply(test_name2,
                                  [points, proj_probs[i]],
                                  ['x', 'y', 'z'] + prob_names)

                        # Save potentials
                        pot_points = np.array(test_loader.dataset.pot_trees[i].data, copy=False)
                        pot_name = join(test_path, 'potentials', cloud_name)
                        pots = test_loader.dataset.potentials[i].numpy().astype(np.float32)
                        write_ply(pot_name,
                                  [pot_points.astype(np.float32), pots],
                                  ['x', 'y', 'z', 'pots'])

                        # Save ascii preds
                        if test_loader.dataset.set == 'test':
                            if test_loader.dataset.name.startswith('Semantic3D'):
                                ascii_name = join(test_path, 'predictions', test_loader.dataset.ascii_files[cloud_name])
                            else:
                                ascii_name = join(test_path, 'predictions', cloud_name[:-4] + '.txt')
                            np.savetxt(ascii_name, preds, fmt='%d')

                    t2 = time.time()
                    print('Done in {:.1f} s\n'.format(t2 - t1))

            test_epoch += 1

            # Break when reaching number of desired votes
            if last_min > num_votes:
                break

        return

    def slam_segmentation_test(self, net, test_loader, config, num_votes=100, debug=True):
        """
        Test method for slam segmentation models
        """

        ############
        # Initialize
        ############

        # Choose validation smoothing parameter (0 for no smothing, 0.99 for big smoothing)
        test_smooth = 0.5
        last_min = -0.5
        softmax = torch.nn.Softmax(1)

        # Number of classes including ignored labels
        nc_tot = test_loader.dataset.num_classes
        nc_model = net.C

        # Test saving path
        test_path = None
        report_path = None
        if config.saving:
            test_path = join('test', config.saving_path.split('/')[-1])
            if not exists(test_path):
                makedirs(test_path)
            report_path = join(test_path, 'reports')
            if not exists(report_path):
                makedirs(report_path)

        if test_loader.dataset.set == 'validation':
            for folder in ['val_predictions', 'val_probs']:
                if not exists(join(test_path, folder)):
                    makedirs(join(test_path, folder))
        else:
            for folder in ['predictions', 'probs']:
                if not exists(join(test_path, folder)):
                    makedirs(join(test_path, folder))

        # Init validation container
        all_f_preds = []
        all_f_labels = []
        if test_loader.dataset.set == 'validation':
            for i, seq_frames in enumerate(test_loader.dataset.frames):
                all_f_preds.append([np.zeros((0,), dtype=np.int32) for _ in seq_frames])
                all_f_labels.append([np.zeros((0,), dtype=np.int32) for _ in seq_frames])

        #####################
        # Network predictions
        #####################

        predictions = []
        targets = []
        test_epoch = 0

        t = [time.time()]
        last_display = time.time()
        mean_dt = np.zeros(1)

        # Start test loop
        while True:
            print('Initialize workers')
            for i, batch in enumerate(test_loader):

                # New time
                t = t[-1:]
                t += [time.time()]

                if i == 0:
                    print('Done in {:.1f}s'.format(t[1] - t[0]))

                if 'cuda' in self.device.type:
                    batch.to(self.device)

                # Forward pass
                outputs = net(batch, config)

                # Get probs and labels
                stk_probs = softmax(outputs).cpu().detach().numpy()
                lengths = batch.lengths[0].cpu().numpy()
                f_inds = batch.frame_inds.cpu().numpy()
                r_inds_list = batch.reproj_inds
                r_mask_list = batch.reproj_masks
                labels_list = batch.val_labels
                torch.cuda.synchronize(self.device)

                t += [time.time()]

                # Get predictions and labels per instance
                # ***************************************

                i0 = 0
                for b_i, length in enumerate(lengths):

                    # Get prediction
                    probs = stk_probs[i0:i0 + length]
                    proj_inds = r_inds_list[b_i]
                    proj_mask = r_mask_list[b_i]
                    frame_labels = labels_list[b_i]
                    s_ind = f_inds[b_i, 0]
                    f_ind = f_inds[b_i, 1]

                    # Project predictions on the frame points
                    proj_probs = probs[proj_inds]

                    # Safe check if only one point:
                    if proj_probs.ndim < 2:
                        proj_probs = np.expand_dims(proj_probs, 0)

                    # Save probs in a binary file (uint8 format for lighter weight)
                    seq_name = test_loader.dataset.sequences[s_ind]
                    if test_loader.dataset.set == 'validation':
                        folder = 'val_probs'
                        pred_folder = 'val_predictions'
                    else:
                        folder = 'probs'
                        pred_folder = 'predictions'
                    filename = '{:s}_{:07d}.npy'.format(seq_name, f_ind)
                    filepath = join(test_path, folder, filename)
                    if exists(filepath):
                        frame_probs_uint8 = np.load(filepath)
                    else:
                        frame_probs_uint8 = np.zeros((proj_mask.shape[0], nc_model), dtype=np.uint8)
                    frame_probs = frame_probs_uint8[proj_mask, :].astype(np.float32) / 255
                    frame_probs = test_smooth * frame_probs + (1 - test_smooth) * proj_probs
                    frame_probs_uint8[proj_mask, :] = (frame_probs * 255).astype(np.uint8)
                    np.save(filepath, frame_probs_uint8)

                    # Save some prediction in ply format for visual
                    if test_loader.dataset.set == 'validation':

                        # Insert false columns for ignored labels
                        frame_probs_uint8_bis = frame_probs_uint8.copy()
                        for l_ind, label_value in enumerate(test_loader.dataset.label_values):
                            if label_value in test_loader.dataset.ignored_labels:
                                frame_probs_uint8_bis = np.insert(frame_probs_uint8_bis, l_ind, 0, axis=1)

                        # Predicted labels
                        frame_preds = test_loader.dataset.label_values[np.argmax(frame_probs_uint8_bis,
                                                                                 axis=1)].astype(np.int32)

                        # Save some of the frame pots
                        if f_ind % 20 == 0:
                            seq_path = join(test_loader.dataset.path, 'sequences', test_loader.dataset.sequences[s_ind])
                            velo_file = join(seq_path, 'velodyne', test_loader.dataset.frames[s_ind][f_ind] + '.bin')
                            frame_points = np.fromfile(velo_file, dtype=np.float32)
                            frame_points = frame_points.reshape((-1, 4))
                            predpath = join(test_path, pred_folder, filename[:-4] + '.ply')
                            #pots = test_loader.dataset.f_potentials[s_ind][f_ind]
                            pots = np.zeros((0,))
                            if pots.shape[0] > 0:
                                write_ply(predpath,
                                          [frame_points[:, :3], frame_labels, frame_preds, pots],
                                          ['x', 'y', 'z', 'gt', 'pre', 'pots'])
                            else:
                                write_ply(predpath,
                                          [frame_points[:, :3], frame_labels, frame_preds],
                                          ['x', 'y', 'z', 'gt', 'pre'])

                            # Also Save lbl probabilities
                            probpath = join(test_path, folder, filename[:-4] + '_probs.ply')
                            lbl_names = [test_loader.dataset.label_to_names[l]
                                         for l in test_loader.dataset.label_values
                                         if l not in test_loader.dataset.ignored_labels]
                            write_ply(probpath,
                                      [frame_points[:, :3], frame_probs_uint8],
                                      ['x', 'y', 'z'] + lbl_names)

                        # keep frame preds in memory
                        all_f_preds[s_ind][f_ind] = frame_preds
                        all_f_labels[s_ind][f_ind] = frame_labels

                    else:

                        # Save some of the frame preds
                        if f_inds[b_i, 1] % 100 == 0:

                            # Insert false columns for ignored labels
                            for l_ind, label_value in enumerate(test_loader.dataset.label_values):
                                if label_value in test_loader.dataset.ignored_labels:
                                    frame_probs_uint8 = np.insert(frame_probs_uint8, l_ind, 0, axis=1)

                            # Predicted labels
                            frame_preds = test_loader.dataset.label_values[np.argmax(frame_probs_uint8,
                                                                                     axis=1)].astype(np.int32)

                            # Load points
                            seq_path = join(test_loader.dataset.path, 'sequences', test_loader.dataset.sequences[s_ind])
                            velo_file = join(seq_path, 'velodyne', test_loader.dataset.frames[s_ind][f_ind] + '.bin')
                            frame_points = np.fromfile(velo_file, dtype=np.float32)
                            frame_points = frame_points.reshape((-1, 4))
                            predpath = join(test_path, pred_folder, filename[:-4] + '.ply')
                            #pots = test_loader.dataset.f_potentials[s_ind][f_ind]
                            pots = np.zeros((0,))
                            if pots.shape[0] > 0:
                                write_ply(predpath,
                                          [frame_points[:, :3], frame_preds, pots],
                                          ['x', 'y', 'z', 'pre', 'pots'])
                            else:
                                write_ply(predpath,
                                          [frame_points[:, :3], frame_preds],
                                          ['x', 'y', 'z', 'pre'])

                    # Stack all prediction for this epoch
                    i0 += length

                # Average timing
                t += [time.time()]
                mean_dt = 0.95 * mean_dt + 0.05 * (np.array(t[1:]) - np.array(t[:-1]))

                # Display
                if (t[-1] - last_display) > 1.0:
                    last_display = t[-1]
                    message = 'e{:03d}-i{:04d} => {:.1f}% (timings : {:4.2f} {:4.2f} {:4.2f}) / pots {:d} => {:.1f}%'
                    min_pot = int(torch.floor(torch.min(test_loader.dataset.potentials)))
                    pot_num = torch.sum(test_loader.dataset.potentials > min_pot + 0.5).type(torch.int32).item()
                    current_num = pot_num + (i + 1 - config.validation_size) * config.val_batch_num
                    print(message.format(test_epoch, i,
                                         100 * i / config.validation_size,
                                         1000 * (mean_dt[0]),
                                         1000 * (mean_dt[1]),
                                         1000 * (mean_dt[2]),
                                         min_pot,
                                         100.0 * current_num / len(test_loader.dataset.potentials)))


            # Update minimum od potentials
            new_min = torch.min(test_loader.dataset.potentials)
            print('Test epoch {:d}, end. Min potential = {:.1f}'.format(test_epoch, new_min))

            if last_min + 1 < new_min:

                # Update last_min
                last_min += 1

                if test_loader.dataset.set == 'validation' and int(np.ceil(last_min)) % 1 == 0:

                    #####################################
                    # Results on the whole validation set
                    #####################################

                    # Confusions for our subparts of validation set
                    Confs = np.zeros((len(predictions), nc_tot, nc_tot), dtype=np.int32)
                    for i, (preds, truth) in enumerate(zip(predictions, targets)):

                        # Confusions
                        Confs[i, :, :] = fast_confusion(truth, preds, test_loader.dataset.label_values).astype(np.int32)


                    # Show vote results
                    print('\nCompute confusion')

                    val_preds = []
                    val_labels = []
                    t1 = time.time()
                    for i, seq_frames in enumerate(test_loader.dataset.frames):
                        val_preds += [np.hstack(all_f_preds[i])]
                        val_labels += [np.hstack(all_f_labels[i])]
                    val_preds = np.hstack(val_preds)
                    val_labels = np.hstack(val_labels)
                    t2 = time.time()
                    C_tot = fast_confusion(val_labels, val_preds, test_loader.dataset.label_values)
                    t3 = time.time()
                    print(' Stacking time : {:.1f}s'.format(t2 - t1))
                    print('Confusion time : {:.1f}s'.format(t3 - t2))

                    s1 = '\n'
                    for cc in C_tot:
                        for c in cc:
                            s1 += '{:7.0f} '.format(c)
                        s1 += '\n'
                    if debug:
                        print(s1)

                    # Remove ignored labels from confusions
                    for l_ind, label_value in reversed(list(enumerate(test_loader.dataset.label_values))):
                        if label_value in test_loader.dataset.ignored_labels:
                            C_tot = np.delete(C_tot, l_ind, axis=0)
                            C_tot = np.delete(C_tot, l_ind, axis=1)

                    # Objects IoU
                    val_IoUs = IoU_from_confusions(C_tot)

                    # Compute IoUs
                    mIoU = np.mean(val_IoUs)
                    s2 = '{:5.2f} | '.format(100 * mIoU)
                    for IoU in val_IoUs:
                        s2 += '{:5.2f} '.format(100 * IoU)
                    print(s2 + '\n')

                    # Save a report
                    report_file = join(report_path, 'report_{:04d}.txt'.format(int(np.floor(last_min))))
                    str = 'Report of the confusion and metrics\n'
                    str += '***********************************\n\n\n'
                    str += 'Confusion matrix:\n\n'
                    str += s1
                    str += '\nIoU values:\n\n'
                    str += s2
                    str += '\n\n'
                    with open(report_file, 'w') as f:
                        f.write(str)

            test_epoch += 1

            # Break when reaching number of desired votes
            if last_min > num_votes:
                break

        return

    def check_equivariance(self, net, test_loader, config, num_votes=100, debug=False):

        # Choose test smoothing parameter (0 for no smothing, 0.99 for big smoothing)
        softmax = torch.nn.Softmax(1)

        # Start validation loop
        with torch.no_grad():
            for batch in test_loader:

                if 'cuda' in self.device.type:
                    batch.to(self.device)

                check_equivariant_probs = False
                if check_equivariant_probs:

                    # Forward pass
                    outputs_0 = net(batch, config, save_block_features=True)
                    features = net.intermediate_features

                    # Rotate and forward pass again
                    rotated_batch, rots = test_loader.dataset.rotate_batch(batch)
                    outputs_1 = net(rotated_batch, config, save_block_features=True)
                    rotated_features = net.intermediate_features

                    # Second inference on real data
                    print(torch.max(torch.abs(outputs_0 - outputs_1)))

                    p0 = softmax(outputs_0).cpu().numpy()
                    p1 = softmax(outputs_1).cpu().numpy()

                    print(p0.shape)

                    # unstack points
                    i0 = 0
                    points = []
                    lengths = batch.lengths[0].cpu().numpy()
                    s_points = batch.points[0].cpu().numpy()
                    for b_i, length in enumerate(lengths):
                        points.append(s_points[i0:i0 + length])
                        i0 += length

                    for i in range(p0.shape[0]):



                        plt.figure()
                        plt.plot(p0[i, :], 'o')
                        plt.plot(p1[i, :], '+')
                        plt.show(block=False)


                        order = np.argsort(p0[i, :])[::-1]

                        print('*********************************')
                        for ind in order[:3]:
                            print('{:.1f}% => {:s}'.format(100 * p0[i, ind], test_loader.dataset.label_to_names[ind]))
                        print('*********************************')

                        show_ModelNet_examples([points[i]])

                        time.sleep(1.0)

                check_lrf_orientations = True
                if check_lrf_orientations:

                    # Forward pass
                    print('\n\nForward pass ...')
                    outputs_0 = net(batch, config, debug_lrf=True)

                    np.set_printoptions(precision=1, suppress=True)
                    print('\n\n------------------------------')
                    for m in net.modules():
                        if isinstance(m, LRFBlock):

                            # Reshape predicted local reference frames
                            lrf = m.pred_rots.cpu().numpy()

                            print(lrf.shape)
                            print(lrf[0, 0, 0, :, :])
                            print(np.dot(lrf[0, 0, 0, :, :], lrf[0, 0, 0, :, :].T))
                            print('------------------------------')



                a = 1/0

    def part_segmentation_test(self, net, test_loader, config, num_votes=100, num_saves=5, debug=False):

        ##################
        # Pre-computations
        ##################

        # Choose test smoothing parameter (0 for no smothing, 0.99 for big smoothing)
        test_smooth = 0.95
        softmax = torch.nn.Softmax(1)

        print('Preparing test structures')
        t1 = time.time()

        # Collect original test file names
        original_path = join(test_loader.dataset.path, 'test_ply')
        test_names = [f[:-4] for f in listdir(original_path) if f[-4:] == '.ply']
        test_names = np.sort(test_names)

        original_labels = []
        original_points = []
        projection_inds = []
        for i, cloud_name in enumerate(test_names):

            # Read data in ply file
            data = read_ply(join(original_path, cloud_name + '.ply'))
            points = np.vstack((data['x'], -data['y'], data['z'])).T
            original_labels += [data['label'] - 1]
            original_points += [points]

            # Create tree structure to compute neighbors
            tree = KDTree(test_loader.dataset.input_points[i])
            projection_inds += [np.squeeze(tree.query(points, return_distance=False))]

        t2 = time.time()
        print('Done in {:.1f} s\n'.format(t2 - t1))

        ##########
        # Initiate
        ##########

        # Test saving path
        if config.saving:
            test_path = join('test', config.saving_path.split('/')[-1])
            if not exists(test_path):
                makedirs(test_path)
        else:
            test_path = None

        #####################
        # Network predictions
        #####################

        # Initiate result containers
        self.test_probs = [np.zeros((1, 1), dtype=np.float32) for _ in test_names]
        self.test_counts = np.zeros(test_names.shape, dtype=np.int32)

        # Number of epochs
        test_epoch = 0
        last_min = -0.5

        t = [time.time()]
        mean_dt = np.zeros(1)
        last_display = time.time()
        while np.min(self.test_counts) < num_votes:

            # Start validation loop
            print('Initialize workers')
            for i, batch in enumerate(test_loader):

                # New time
                t = t[-1:]
                t += [time.time()]

                if i == 0:
                    print('Done in {:.1f}s'.format(t[1] - t[0]))

                if 'cuda' in self.device.type:
                    batch.to(self.device)

                # Forward pass
                outputs = net(batch, config)

                t += [time.time()]

                # Get probs and labels
                stacked_probs = softmax(outputs).cpu().detach().numpy()
                s_points = batch.points[0].cpu().numpy()
                lengths = batch.lengths[0].cpu().numpy()
                obj_labels = batch.obj_labels.cpu().numpy()
                obj_inds = batch.model_inds.cpu().numpy()
                torch.cuda.synchronize(self.device)

                # Get predictions and labels per instance
                # ***************************************

                i0 = 0
                for b_i, length in enumerate(lengths):

                    # Get prediction
                    o_lbl = obj_labels[b_i]
                    o_i = obj_inds[b_i]
                    probs = stacked_probs[i0:i0 + length, :test_loader.dataset.num_parts[o_lbl]]

                    # Update current probs in whole cloud
                    self.test_probs[o_i] = test_smooth * self.test_probs[o_i] + (1 - test_smooth) * probs
                    self.test_counts[o_i] += 1
                    i0 += length

                # Average timing
                t += [time.time()]
                if i < 2:
                    mean_dt = np.array(t[1:]) - np.array(t[:-1])
                else:
                    mean_dt = 0.9 * mean_dt + 0.1 * (np.array(t[1:]) - np.array(t[:-1]))

                # Display
                if (t[-1] - last_display) > 1.0:
                    last_display = t[-1]
                    message = 'e{:03d}-i{:04d} => {:.1f}% (timings : {:4.2f} {:4.2f} {:4.2f})'
                    print(message.format(test_epoch, i,
                                         100 * i / config.validation_size,
                                         1000 * (mean_dt[0]),
                                         1000 * (mean_dt[1]),
                                         1000 * (mean_dt[2])))

            # Update minimum of potentials
            new_min = np.min(self.test_counts)
            print('Test epoch {:d}, end. Min potential = {:.1f}'.format(test_epoch, new_min))

            # Save predicted cloud
            if last_min + 1 < new_min:

                # Update last_min
                last_min += 1

                # Show vote results
                # *****************

                example_path = join(test_path, '_Examples')
                if not exists(example_path):
                    makedirs(example_path)

                SubConfs = []
                Confs = []
                for obj_i, avg_probs in enumerate(self.test_probs):

                    # Compute confusion matrices
                    proj_probs = avg_probs[projection_inds[obj_i]]
                    parts = [j for j in range(avg_probs.shape[1])]
                    SubConfs += [fast_confusion(test_loader.dataset.input_pts_labels[obj_i],
                                             np.argmax(avg_probs, axis=1),
                                             np.array(parts, dtype=np.int32))]
                    Confs += [fast_confusion(original_labels[obj_i],
                                             np.argmax(proj_probs, axis=1),
                                             np.array(parts, dtype=np.int32))]

                    # Save some of the models
                    if obj_i % 3 == 1:
                        filename = join(example_path, test_names[obj_i])
                        preds = np.argmax(proj_probs, axis=1).astype(np.int32)
                        write_ply(filename,
                                  [original_points[obj_i], original_labels[obj_i], preds],
                                  ['x', 'y', 'z', 'gt', 'pre'])

                # Regroup confusions per object class
                Confs = np.array(Confs, dtype=object)
                obj_mIoUs = []
                for l in test_loader.dataset.label_values:

                    # Get confusions for this object
                    obj_inds = np.where(test_loader.dataset.input_obj_labels == l)[0]
                    obj_confs = np.stack(Confs[obj_inds])

                    # Get IoU
                    obj_IoUs = IoU_from_confusions(obj_confs)
                    obj_mIoUs += [np.mean(obj_IoUs, axis=-1)]

                    # Get X best and worst prediction
                    order = np.argsort(obj_mIoUs[-1])
                    obj_inds = np.where(test_loader.dataset.input_obj_labels == l)[0]
                    worst_inds = obj_inds[order[:num_saves]]
                    best_inds = obj_inds[order[:-num_saves - 1:-1]]
                    worst_IoUs = obj_IoUs[order[:num_saves]]
                    best_IoUs = obj_IoUs[order[:-num_saves - 1:-1]]

                    # Save the names in a file
                    obj_path = join(test_path, test_loader.dataset.label_to_names[l])
                    if not exists(obj_path):
                        makedirs(obj_path)
                    worst_file = join(obj_path, 'worst_inds.txt')
                    best_file = join(obj_path, 'best_inds.txt')
                    with open(worst_file, "w") as text_file:
                        for w_i, w_IoUs in zip(worst_inds, worst_IoUs):
                            text_file.write('{:d} {:s} :'.format(w_i, test_names[w_i]))
                            for IoU in w_IoUs:
                                text_file.write(' {:.1f}'.format(100 * IoU))
                            text_file.write('\n')

                    with open(best_file, "w") as text_file:
                        for b_i, b_IoUs in zip(best_inds, best_IoUs):
                            text_file.write('{:d} {:s} :'.format(b_i, test_names[b_i]))
                            for IoU in b_IoUs:
                                text_file.write(' {:.1f}'.format(100 * IoU))
                            text_file.write('\n')

                    # Save the clouds
                    for i, w_i in enumerate(worst_inds):
                        filename = join(obj_path, 'worst_{:02d}.ply'.format(i + 1))
                        preds = np.argmax(self.test_probs[w_i][projection_inds[w_i]], axis=1).astype(np.int32)
                        write_ply(filename,
                                  [original_points[w_i], original_labels[w_i], preds],
                                  ['x', 'y', 'z', 'gt', 'pre'])

                    for i, b_i in enumerate(best_inds):
                        filename = join(obj_path, 'best_{:02d}.ply'.format(i + 1))
                        preds = np.argmax(self.test_probs[b_i][projection_inds[b_i]], axis=1).astype(np.int32)
                        write_ply(filename,
                                  [original_points[b_i], original_labels[b_i], preds],
                                  ['x', 'y', 'z', 'gt', 'pre'])

                # Display results
                # ***************

                objs_average = [np.mean(mIoUs) for mIoUs in obj_mIoUs]
                instance_average = np.mean(np.hstack(obj_mIoUs))
                class_average = np.mean(objs_average)

                print('Objs | Inst | Air  Bag  Cap  Car  Cha  Ear  Gui  Kni  Lam  Lap  Mot  Mug  Pis  Roc  Ska  Tab')
                print('-----|------|--------------------------------------------------------------------------------')

                s = '{:4.1f} | {:4.1f} | '.format(100 * class_average, 100 * instance_average)
                for AmIoU in objs_average:
                    s += '{:4.1f} '.format(100 * AmIoU)
                print(s + '\n')

    def slam_MyhalSim_test(self, net, test_loader, config, num_votes=100, debug=True):
        """
        Test method for slam segmentation models
        """

        ############
        # Initialize
        ############

        # Choose validation smoothing parameter (0 for no smothing, 0.99 for big smoothing)
        test_smooth = 0.5
        last_min = -0.5
        softmax = torch.nn.Softmax(1)

        # Number of classes including ignored labels
        nc_tot = test_loader.dataset.num_classes
        nc_model = net.C

        GT_categories = np.array(['ground',
                                  'chair',
                                  'movingP',
                                  'stillP',
                                  'table',
                                  'wall',
                                  'door'])
        GT_label_values = np.array([i for i, _ in enumerate(GT_categories)])

        nc_GT = len(GT_categories)


        # Test saving path
        test_path = None
        report_path = None
        if config.saving:
            test_path = join('test', config.saving_path.split('/')[-1])
            if not exists(test_path):
                makedirs(test_path)
            report_path = join(test_path, 'reports')
            if not exists(report_path):
                makedirs(report_path)

        if test_loader.dataset.set == 'validation':
            for folder in ['val_predictions', 'val_probs']:
                if not exists(join(test_path, folder)):
                    makedirs(join(test_path, folder))
        else:
            for folder in ['predictions', 'probs']:
                if not exists(join(test_path, folder)):
                    makedirs(join(test_path, folder))

        # Init validation container
        all_f_preds = []
        all_f_labels = []

        for i, seq_frames in enumerate(test_loader.dataset.frames):
            all_f_preds.append([np.zeros((0,), dtype=np.int32) for _ in seq_frames])
            all_f_labels.append([np.zeros((0,), dtype=np.int32) for _ in seq_frames])

        #####################
        # Network predictions
        #####################

        predictions = []
        targets = []
        test_epoch = 0

        t = [time.time()]
        last_display = time.time()
        mean_dt = np.zeros(1)

        # Start test loop
        while True:
            print('Initialize workers')
            for i, batch in enumerate(test_loader):

                # New time
                t = t[-1:]
                t += [time.time()]

                if i == 0:
                    print('Done in {:.1f}s'.format(t[1] - t[0]))

                if 'cuda' in self.device.type:
                    batch.to(self.device)

                # Forward pass
                outputs = net(batch, config)

                # Get probs and labels
                stk_probs = softmax(outputs).cpu().detach().numpy()
                lengths = batch.lengths[0].cpu().numpy()
                f_inds = batch.frame_inds.cpu().numpy()
                r_inds_list = batch.reproj_inds
                r_mask_list = batch.reproj_masks
                labels_list = batch.val_labels
                torch.cuda.synchronize(self.device)

                t += [time.time()]

                # Get predictions and labels per instance
                # ***************************************

                i0 = 0
                for b_i, length in enumerate(lengths):

                    # Get prediction
                    probs = stk_probs[i0:i0 + length]
                    proj_inds = r_inds_list[b_i]
                    proj_mask = r_mask_list[b_i]
                    frame_labels = labels_list[b_i]
                    s_ind = f_inds[b_i, 0]
                    f_ind = f_inds[b_i, 1]

                    # Project predictions on the frame points
                    proj_probs = probs[proj_inds]

                    # Safe check if only one point:
                    if proj_probs.ndim < 2:
                        proj_probs = np.expand_dims(proj_probs, 0)

                    # Save probs in a binary file (uint8 format for lighter weight)
                    seq_name = test_loader.dataset.sequences[s_ind]
                    if test_loader.dataset.set == 'validation':
                        folder = 'val_probs'
                        pred_folder = 'val_predictions'
                    else:
                        folder = 'probs'
                        pred_folder = 'predictions'
                    filename = '{:s}_{:07d}.npy'.format(seq_name, f_ind)
                    filepath = join(test_path, folder, filename)
                    if exists(filepath):
                        frame_probs_uint8 = np.load(filepath)
                    else:
                        frame_probs_uint8 = np.zeros((proj_mask.shape[0], nc_model), dtype=np.uint8)
                    frame_probs = frame_probs_uint8[proj_mask, :].astype(np.float32) / 255
                    frame_probs = test_smooth * frame_probs + (1 - test_smooth) * proj_probs
                    frame_probs_uint8[proj_mask, :] = (frame_probs * 255).astype(np.uint8)
                    np.save(filepath, frame_probs_uint8)

                    # Save some prediction in ply format for visual

                    # Insert false columns for ignored labels
                    frame_probs_uint8_bis = frame_probs_uint8.copy()
                    for l_ind, label_value in enumerate(test_loader.dataset.label_values):
                        if label_value in test_loader.dataset.ignored_labels:
                            frame_probs_uint8_bis = np.insert(frame_probs_uint8_bis, l_ind, 0, axis=1)

                    # Predicted labels
                    frame_preds = test_loader.dataset.label_values[np.argmax(frame_probs_uint8_bis,
                                                                             axis=1)].astype(np.int32)

                    # Save some of the frame pots
                    if f_ind % 20 == 0:

                        frame_points = test_loader.dataset.load_points(s_ind, f_ind)
                        predpath = join(test_path, pred_folder, filename[:-4] + '.ply')
                        # pots = test_loader.dataset.f_potentials[s_ind][f_ind]
                        pots = np.zeros((0,))
                        if pots.shape[0] > 0:
                            write_ply(predpath,
                                      [frame_points[:, :3], frame_labels, frame_preds, pots],
                                      ['x', 'y', 'z', 'gt', 'pre', 'pots'])
                        else:
                            write_ply(predpath,
                                      [frame_points[:, :3], frame_labels, frame_preds],
                                      ['x', 'y', 'z', 'gt', 'pre'])

                        # Also Save lbl probabilities
                        probpath = join(test_path, folder, filename[:-4] + '_probs.ply')
                        lbl_names = [test_loader.dataset.label_to_names[l]
                                     for l in test_loader.dataset.label_values
                                     if l not in test_loader.dataset.ignored_labels]
                        write_ply(probpath,
                                  [frame_points[:, :3], frame_probs_uint8],
                                  ['x', 'y', 'z'] + lbl_names)

                    # keep frame preds in memory
                    all_f_preds[s_ind][f_ind] = frame_preds
                    all_f_labels[s_ind][f_ind] = frame_labels

                    # Stack all prediction for this epoch
                    i0 += length

                # Average timing
                t += [time.time()]
                mean_dt = 0.95 * mean_dt + 0.05 * (np.array(t[1:]) - np.array(t[:-1]))

                # Display
                if (t[-1] - last_display) > 1.0:
                    last_display = t[-1]
                    message = 'e{:03d}-i{:04d} => {:.1f}% (timings : {:4.2f} {:4.2f} {:4.2f}) / pots {:d} => {:.1f}%'
                    min_pot = int(torch.floor(torch.min(test_loader.dataset.potentials)))
                    pot_num = torch.sum(test_loader.dataset.potentials > min_pot + 0.5).type(torch.int32).item()
                    current_num = pot_num + (i + 1 - config.validation_size) * config.val_batch_num
                    print(message.format(test_epoch, i,
                                         100 * i / config.validation_size,
                                         1000 * (mean_dt[0]),
                                         1000 * (mean_dt[1]),
                                         1000 * (mean_dt[2]),
                                         min_pot,
                                         100.0 * current_num / len(test_loader.dataset.potentials)))

            # Update minimum of potentials
            new_min = torch.min(test_loader.dataset.potentials)
            print('Test epoch {:d}, end. Min potential = {:.1f}'.format(test_epoch, new_min))

            if last_min + 1 < new_min:

                # Update last_min
                last_min += 1

                if int(np.ceil(last_min)) % 1 == 0:

                    #####################################
                    # Results on the whole validation set
                    #####################################

                    # Show vote results
                    print('\nCompute confusion')

                    C_tot = []
                    for i, seq_frames in enumerate(test_loader.dataset.frames):

                        t1 = time.time()
                        val_preds = np.hstack(all_f_preds[i])
                        val_labels = np.hstack(all_f_labels[i])
                        C_tot += [fast_confusion(val_labels, val_preds, GT_label_values)]
                        t2 = time.time()
                        print('   > {:s}: {:.1f}s'.format(test_loader.dataset.sequences[i], t2 - t1))
                        print(val_labels.shape, val_preds.shape)

                    C_tot = np.stack(C_tot, axis=0)
                    C_tot = np.sum(C_tot, axis=0)

                    # Make it nicer:
                    C_tot = np.delete(C_tot, 0, axis=1)
                    C_tot = np.delete(C_tot, -1, axis=1)
                    C_tot = np.delete(C_tot, -1, axis=1)

                    s = ' {:>10s}'.format('-')
                    for i2, _ in enumerate(C_tot[0, :]):
                        s += ' {:^10s}'.format(test_loader.dataset.label_to_names[i2])
                    s += '\n'
                    for i1, cc in enumerate(C_tot):
                        s += ' {:>10s}'.format(GT_categories[i1])
                        for i2, c in enumerate(cc):
                            s += ' {:^10d}'.format(c)
                        s += '\n'

                    print('*------------------------------------------------*')
                    print('Confusion\n')
                    print(s)
                    print('\n*------------------------------------------------*')

                    # Save some frame preds combined together with pose
                    print('Saving frames')
                    for s_ind, seq_frames in enumerate(test_loader.dataset.frames):

                        new_path = join(test_loader.dataset.original_path,
                                        'simulated_runs',
                                        test_loader.dataset.sequences[s_ind],
                                        'classif2_frames')

                        if not exists(new_path):
                            makedirs(new_path)

                        for f_ind, f_name in enumerate(seq_frames):

                            # load points
                            frame_points = test_loader.dataset.load_points(s_ind, f_ind)

                            # Save preds

                            print(frame_points.shape, all_f_preds[s_ind][f_ind].shape)

                            new_file = join(new_path, f_name + '.ply')
                            write_ply(new_file,
                                      [frame_points, all_f_preds[s_ind][f_ind]],
                                      ['x', 'y', 'z', 'cat'])
                    print('Done\n')

                    # Save a report
                    report_file = join(report_path, 'report_{:04d}.txt'.format(int(np.floor(last_min))))
                    str = 'Report of the confusion and metrics\n'
                    str += '***********************************\n\n\n'
                    str += 'Confusion matrix:\n\n'
                    str += s
                    str += '\n\n'
                    with open(report_file, 'w') as f:
                        f.write(str)

            test_epoch += 1

            # Break when reaching number of desired votes
            if last_min > num_votes:
                break

        return
