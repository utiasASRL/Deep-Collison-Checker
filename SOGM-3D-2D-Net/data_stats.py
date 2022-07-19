#
#
#      0==============================0
#      |    Deep Collision Checker    |
#      0==============================0
#
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      Callable script to start a training on MultiCollision dataset
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      Hugues THOMAS - 06/03/2020
#


# ----------------------------------------------------------------------------------------------------------------------
#
#           Imports and global variables
#       \**********************************/
#

# Common libs
import sys
import time
import signal
import os
os.environ.update(OMP_NUM_THREADS='1',
                  OPENBLAS_NUM_THREADS='1',
                  NUMEXPR_NUM_THREADS='1',
                  MKL_NUM_THREADS='1',)
import numpy as np


from os.path import exists, join
from os import makedirs

from MyhalCollision_sessions import UTIn3D_H_sessions, UTIn3D_A_sessions, UTIn3D_A_sessions_v2
from datasets.MultiCollision import MultiCollisionDataset, MultiCollisionSampler, MultiCollisionCollate

from train_MultiCollision import MultiCollisionConfig


# ----------------------------------------------------------------------------------------------------------------------
#
#           Main Call
#       \***************/
#

if __name__ == '__main__':


    for sessions in [UTIn3D_A_sessions_v2, UTIn3D_H_sessions]:

        print('\n\n************************************')

        dataset_path, map_day, refine_sessions, train_days, train_comments = sessions()

        val_inds = np.array([i for i, c in enumerate(train_comments) if 'val' in c.split('>')[0]])

        # Validation sessions
        train_inds = [i for i in range(len(train_days)) if i not in val_inds]

        all_frames = []
        for seq in train_days:
            seq_path = join(dataset_path, 'runs', seq, 'velodyne_frames')
            frames = np.array([vf[:-4] for vf in os.listdir(seq_path) if vf.endswith('.ply')])
            order = np.argsort([float(ff) for ff in frames])
            frames = frames[order]
            all_frames.append(frames)

        for s_ind, seq in enumerate(train_days):
            print(seq, 'val' in train_comments[s_ind].split('>')[0], all_frames[s_ind][0], all_frames[s_ind][-1], float(all_frames[s_ind][-1]) - float(all_frames[s_ind][0]), len(all_frames[s_ind]))


    for sessions in [UTIn3D_A_sessions_v2, UTIn3D_H_sessions]:

        dataset_path, map_day, refine_sessions, train_days, train_comments = UTIn3D_A_sessions_v2()
        dataset_path, map_day, refine_sessions, train_days, train_comments = UTIn3D_A_sessions_v2()
        val_inds = np.array([i for i, c in enumerate(train_comments) if 'val' in c.split('>')[0]])
        train_inds = [i for i in range(len(train_days)) if i not in val_inds]

    # Get sessions from the annotation script
    dataset_path, map_day, refine_sessions, train_days, train_comments = UTIn3D_A_sessions_v2()
    dataset_path2, map_day2, refine_sessions2, train_days2, train_comments2 = UTIn3D_H_sessions()

    # Get training and validation sets
    val_inds = np.array([i for i, c in enumerate(train_comments) if 'val' in c.split('>')[0]])
    val_inds2 = np.array([i for i, c in enumerate(train_comments2) if 'val' in c.split('>')[0]])

    # Validation sessions
    train_inds = [i for i in range(len(train_days)) if i not in val_inds]
    train_inds2 = [i for i in range(len(train_days2)) if i not in val_inds2]
    
    # Initialize configuration class
    config = MultiCollisionConfig()

    data_i = 2
    if data_i == 1:
        train_days_lists = [train_days]
    else:
        train_days_lists = [train_days2]
        dataset_path = dataset_path2

    training_dataset = MultiCollisionDataset(config,
                                             train_days_lists,
                                             chosen_set='training',
                                             dataset_paths=[dataset_path],
                                             simulated=[False],
                                             balance_classes=True)

    training_sampler = MultiCollisionSampler(training_dataset, manual_training_frames=False)

    im_lim = training_dataset.config.radius_2D / np.sqrt(2)

    from datasets.MultiCollision import ndimage, plt, Circle, Slider


    # Measure the % of frames with dynamic points
    # We keep the "class_mask_opened" in the end
    if True:
        for i_l, ll in enumerate(training_dataset.label_values):
            if ll == 4:
                
                print('\nTraining frame selection:\n')

                # Variable containg the selected inds for this class (will replace training_dataset.class_frames[i_l])
                selected_mask = np.zeros_like(training_dataset.all_inds[:, 0], dtype=bool)

                # Get 2D data
                all_pts, all_colors, all_labels = training_sampler.get_2D_data(im_lim)

                print('\nSelecting blobs of frames containing dynamic points:')

                all_seq_counts = []
                all_pts_counts = []

                for s_ind, seq in enumerate(training_dataset.sequences):

                    t0 = time.time()

                    # Get the wanted indices
                    class_mask = np.vstack(all_labels[s_ind]).T
                    class_mask = class_mask[i_l:i_l + 1] > 10

                    # Remove isolated inds with opening
                    open_struct = np.ones((1, 31))
                    class_mask_opened = ndimage.binary_opening(class_mask, structure=open_struct)
                    
                    # Remove the one where the person is disappearing or reappearing
                    erode_struct = np.ones((1, 31))
                    erode_struct[:, :13] = 0
                    class_mask_eroded = ndimage.binary_erosion(class_mask_opened, structure=erode_struct)

                    # Update selected inds for all sequences
                    seq_mask = training_dataset.all_inds[:, 0] == s_ind
                    selected_mask[seq_mask] = np.squeeze(class_mask_eroded)

                    all_seq_counts.append([np.sum(class_mask.astype(np.int32)),
                                           np.sum(class_mask_opened.astype(np.int32)),
                                           np.sum(class_mask_eroded.astype(np.int32)),
                                           class_mask.shape[1]])

                    # Get % of points
                    test_count = np.vstack(all_labels[s_ind])  # [Nframes, C]
                    test_count = np.sum(test_count, axis=0)  # [C]
                    all_pts_counts.append(test_count)

                    t1 = time.time()
                    print('Sequence {:s} done in {:.1f}s'.format(seq, t1 - t0))

                    debug = False
                    if debug:
                        # Figure
                        figA, axA = plt.subplots(1, 1, figsize=(10, 7))
                        plt.subplots_adjust(left=0.1, bottom=0.2)

                        # Plot first frame of seq
                        plotsA = [axA.scatter(all_pts[s_ind][0][:, 0],
                                              all_pts[s_ind][0][:, 1],
                                              s=2.0,
                                              c=all_colors[s_ind][0])]

                        # Show a circle of the loop closure area
                        axA.add_patch(Circle((0, 0), radius=0.2,
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
                        axtime = plt.axes([0.1, 0.1, 0.8, 0.015], facecolor=axcolor)
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

                        # Ax with the presence of dynamic points
                        class_mask = np.zeros_like(training_dataset.all_inds[:, 0], dtype=bool)
                        class_mask[training_dataset.class_frames[i_l]] = True
                        seq_mask = training_dataset.all_inds[:, 0] == s_ind
                        seq_class_frames = class_mask[seq_mask]
                        seq_class_frames = np.expand_dims(seq_class_frames, 0)
                        axdyn = plt.axes([0.1, 0.08, 0.8, 0.015])
                        axdyn.imshow(seq_class_frames, cmap='GnBu', aspect='auto')
                        axdyn.set_axis_off()

                        # Ax with the presence of dynamic points at least 10
                        dyn_img = np.vstack(all_labels[s_ind]).T
                        dyn_img = dyn_img[-1:]
                        dyn_img[dyn_img > 10] = 10
                        dyn_img[dyn_img > 0] += 10
                        axdyn = plt.axes([0.1, 0.06, 0.8, 0.015])
                        axdyn.imshow(dyn_img, cmap='OrRd', aspect='auto')
                        axdyn.set_axis_off()

                        
                        # Ax with opened
                        axdyn = plt.axes([0.1, 0.04, 0.8, 0.015])
                        axdyn.imshow(class_mask_opened, cmap='OrRd', aspect='auto')
                        axdyn.set_axis_off()
                        
                        # Ax with eroded
                        axdyn = plt.axes([0.1, 0.02, 0.8, 0.015])
                        axdyn.imshow(class_mask_eroded, cmap='OrRd', aspect='auto')
                        axdyn.set_axis_off()

                        plt.show()


                all_seq_counts = np.array(all_seq_counts, dtype=np.int32)
                all_percents = all_seq_counts.astype(np.float32)
                all_percents = all_percents / all_percents[:, -1:]

                all_pts_counts = np.vstack(all_pts_counts)

                for s_ind, seq in enumerate(training_dataset.sequences):
                    print(seq, all_percents[s_ind][1], all_pts_counts[s_ind])



    # Measure the total nubmer of points
    if False:
        # Advanced display
        N = training_dataset.all_inds.shape[0]
        progress_n = 50
        fmt_str = '[{:<' + str(progress_n) + '}] {:5.1f}%'
        
        print('\nGetting dataset NB of points')

        # Get 2D points colors and labels
        all_nb_pts = []
        tot_i = 0
        for s_ind, seq in enumerate(training_dataset.sequences):
            sum_frames = 0
            for f_ind, frame in enumerate(training_dataset.frames[s_ind]):
                pts = training_dataset.load_points(s_ind, f_ind)
                sum_frames += pts.shape[0]
                tot_i += 1
            all_nb_pts.append(sum_frames)
            print('', end='\r')
            print(fmt_str.format('#' * (((tot_i + 1) * progress_n) // N), 100 * (tot_i + 1) / N), end='', flush=True)
            
        # Show a nice 100% progress bar
        print('', end='\r')
        print(fmt_str.format('#' * progress_n, 100), flush=True)
        print('\n')

        for s_ind, seq in enumerate(training_dataset.sequences):
            print(seq, all_nb_pts[s_ind])
                




