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
from os.path import isfile, join, exists
from os import listdir, remove, getcwd, makedirs
import time
import pickle


# ----------------------------------------------------------------------------------------------------------------------
#
#           Utility functions
#       \***********************/
#





# ----------------------------------------------------------------------------------------------------------------------
#
#           Main call
#       \***************/
#


if __name__ == '__main__':
    
    # Parameters
    roots = ['../Data/UTI3D_A', '../Data/UTI3D_H']
    folders = ['annotated_frames',
               'annotation',
               'calibration',
               'collisions',
               'runs',
               'slam_offline']

    gdrive_parents = ['1fCffwd_Z9v6886LzO9RmkAMGUdaqAX7t', '1-XRsO3V5yh6iSZgznRORKP7RoKbWSi2a']


    for root, gdrive_parent in zip(roots, gdrive_parents):

        print('\n')
        print('-------------------------------------------------------------------------')
        print('\nPreparing' + root)
        print('*********' + '*' * len(root))


        for folder in folders:

            folder_path = join(root, folder)
            print('\n    ' + folder_path)

            # First zip folder
            print('        > zipping')
            shutil.make_archive(folder_path, 'zip', folder_path)

            # Upload the zip file on google drive
            print('        > uploading')
            zip_file = folder_path + '.zip'
            os.system('gdrive upload -p {:s} {:s}'.format(gdrive_parent, zip_file))

            # Now remove the file to save space
            print('        > removing')
            if exists(zip_file):
                remove(zip_file)
    
























