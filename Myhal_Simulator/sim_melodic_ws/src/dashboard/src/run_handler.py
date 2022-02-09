#!/usr/bin/env python

from utilities import math_utilities as mu
from utilities import plot_utilities as pu
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from tabulate import tabulate
import scipy.interpolate as interpolate
from itertools import cycle
import subprocess
import cPickle as pickle
import time
import json
import os
import numpy as np
import copy
import logging
import psutil
import shutil
import yaml
import plyfile as ply
#from moviepy.editor import VideoFileClip, concatenate_videoclips

plt.style.use('seaborn-bright')

class bc:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

    @classmethod
    def color(cls, text, color):
        return color + text + bc.ENDC

class Run:

    def __init__(self, name, meta, data = None):
        self.name = name
        self.data = data
        self.meta = meta

    def get_data(self,field):
        return self.data[field]
    
    def keys(self):
        return self.data.keys()

class Plot:
    ''' The parent class of all plot types '''

    def __init__(self, aggregate = False):
        '''Optional argument
        aggregate (bool): if True, the data for all runs in a given series will be averaged, default = False'''
        self.data = {'x_data':[],'y_data':[], 'series_name': [], 'color' : [], 'line': []}
        self.aggregate = aggregate
        self.series_list = []
        self.ax = None

    def add_ax(self, ax):
        self.ax = ax
        self.label()
        
    def init_axis(self):
        self.collect_data()
        for i in range(len(self.data['x_data'])):
            self.ax.plot(self.data['x_data'][i], self.data['y_data'][i], self.data['line'][i], label = self.data['series_name'][i], color = self.data['color'][i])
        self.ax.legend()

    def __str__(self):
        res = self.ax.get_title() + ", (x label: " + self.ax.get_xlabel() + ", y label: " + self.ax.get_ylabel() + ")"
        return res

    def add_series(series):
        self.series_list.append(series)

class TEBoxPlot(Plot):
    ''' A box plot for translation error: x axis = series, y axis = translation error '''

    def label(self):
        self.ax.set_title("Translation Error")
        self.ax.set(xlabel='Series', ylabel = 'Translation Error (m)')

    def init_axis(self):
        self.collect_data()
        bplot = self.ax.boxplot(self.data['y_data'], labels = self.data['x_data'], patch_artist = True, showfliers=False)
        for patch, color in zip(bplot['boxes'], self.data['color']):
            patch.set_facecolor(color)

    def collect_data(self):
        self.data = {'x_data':[],'y_data':[], 'series_name': [], 'color' : [], 'line': []}
        for series in self.series_list:
            name = series.name
            self.data['x_data'].append(name)
            self.data['color'].append(series.colors[0])
            s_dist = []

            for run in series.runs:
                gt_traj = run.get_data('gt_traj')
                loc_traj = run.get_data('amcl_traj') if ('amcl_traj' in run.keys()) else run.get_data('gmapping_traj')
                error = pu.translation_error(loc_traj,gt_traj)
                s_dist += list(error)
                
            self.data['y_data'].append(s_dist)

    def info(self):
        res = ""
        self.data.clear()
        self.data = {'x_data':[],'y_data':[], 'series_name': [], 'color' : [], 'line': []}
        self.collect_data()

        for i in range(len(self.data['x_data'])):
            series = self.data['x_data'][i]
            y = self.data['y_data'][i]
            avg_error = sum(y)/len(y)

            res+= "Series: " + bc.color(series,bc.HEADER) + " has an average translation error of {:.3f} m\n".format(avg_error)

        return res

class YEBoxPlot(Plot):
    ''' A box plot for yaw error: x axis = series, y axis = yaw error '''

    def label(self):
        self.ax.set_title("Yaw Error")
        self.ax.set(xlabel='Series', ylabel = 'Yaw Error (rad)')

    def init_axis(self):
        self.collect_data()
        bplot = self.ax.boxplot(self.data['y_data'], labels = self.data['x_data'], patch_artist = True, showfliers=False)
        for patch, color in zip(bplot['boxes'], self.data['color']):
            patch.set_facecolor(color)

    def collect_data(self):
        self.data = {'x_data':[],'y_data':[], 'series_name': [], 'color' : [], 'line': []}
        for series in self.series_list:
            name = series.name
            self.data['x_data'].append(name)
            self.data['color'].append(series.colors[0])
            s_dist = []

            for run in series.runs:
                gt_traj = run.get_data('gt_traj')
                loc_traj = run.get_data('amcl_traj') if ('amcl_traj' in run.keys()) else run.get_data('gmapping_traj')
                error = pu.yaw_error(loc_traj,gt_traj)
                s_dist += list(error)
                
            self.data['y_data'].append(s_dist)

    def info(self):
        res = ""
        self.data.clear()
        self.data = {'x_data':[],'y_data':[], 'series_name': [], 'color' : [], 'line': []}
        self.collect_data()

        for i in range(len(self.data['x_data'])):
            series = self.data['x_data'][i]
            y = self.data['y_data'][i]
            avg_error = sum(y)/len(y)

            res+= "Series: " + bc.color(series,bc.HEADER) + " has an average yaw error of {:.3f} m\n".format(avg_error)

        return res

class TranslationError(Plot):
    ''' A plot of the difference between the predicted position and actual position of the robot at a given point in time, w.r.t the distance travelled '''
    
    def label(self):
        self.ax.set_title("Translation Error")
        self.ax.set(xlabel='Distance Travelled (m)', ylabel = 'Translation Error (m)')

    def collect_data(self):
        self.data = {'x_data':[],'y_data':[], 'series_name': [], 'color' : [], 'line': []}
        for series in self.series_list:
            name = series.name

            max_dist = 0
            temp_data = {'x_data':[],'y_data':[], 'series_name': [], 'color' : [], 'line': []}

            for run in series.runs:
                gt_traj = run.get_data('gt_traj')
                loc_traj = run.get_data('amcl_traj') if ('amcl_traj' in run.keys()) else run.get_data('gmapping_traj')
                dists = pu.list_distances(gt_traj)
                temp_data['x_data'].append(dists[0])
                max_dist = max(max_dist, dists[1])
                temp_data['y_data'].append(pu.translation_error(loc_traj,gt_traj))
                temp_data['series_name'].append(name)
                temp_data['color'].append(series.colors[0])
                temp_data['line'].append('-')

            if self.aggregate:
                
                x_bins = list(np.linspace(0,max_dist, max_dist/0.1))
                y_store = [0]*len(x_bins)

                for i in range(len(temp_data['x_data'])):
                    dists = temp_data['x_data'][i]
                    min_x = dists[0]
                    max_x = dists[-1]
                    diffs = temp_data['y_data'][i]
                    f = interpolate.interp1d(dists,diffs);

                    j = 0
                    for x in x_bins:
                        if (x <= max_x and x >= min_x):
                            if (y_store[j] != 0):
                                y_store[j][0]+=f(x)
                                y_store[j][1]+=1
                            else:
                                y_store[j] = [f(x), 1] # [0] stores diff sums, [1] stores number of additions
                        j+=1

                temp_data['x_data'] = [x_bins]
                y_data = [0]*len(x_bins)
                for i in range(len(y_store)):
                    y_data[i] = float(y_store[i][0])/(float(y_store[i][1]))

                temp_data['y_data'] = [y_data]

                temp_data['series_name'] = [name]
                temp_data['color'] = [series.colors[0]]
                temp_data['line']= ['-']

            for key in self.data:
                self.data[key] += temp_data[key]
        
    def info(self):
        res = ""
        self.data.clear()
        self.data = {'x_data':[],'y_data':[], 'series_name': [], 'color' : [], 'line': []}
        temp = self.aggregate
        self.aggregate = True
        self.collect_data()
        self.aggregate = temp

        for i in range(len(self.data['series_name'])):
            x = self.data['x_data'][i]
            y = self.data['y_data'][i]
            avg_error = 0
            for j in range(len(self.data['x_data'][i])-1):

               avg_error += ((y[j]+y[j+1])/2.0)*(x[j+1]-x[j])
            
            avg_error = avg_error/(x[-1]-x[0])

            res+= "Series: " + bc.color(self.data['series_name'][i],bc.HEADER) + " has an average translation error of {:.3f} m\n".format(avg_error)

        return res

class YawError(Plot):
    ''' A plot of the difference between the predicted yaw and the actual yaw of the robot at a given point in time w.r.t the distance travelled'''

    def label(self):
        self.ax.set_title("Yaw Error")
        self.ax.set(xlabel='Distance Travelled (m)', ylabel = 'Yaw Error (m)')

    def collect_data(self):
        self.data = {'x_data':[],'y_data':[], 'series_name': [], 'color' : [], 'line': []}
        for series in self.series_list:
            name = series.name

            max_dist = 0
            temp_data = {'x_data':[],'y_data':[], 'series_name': [], 'color' : [], 'line': []}

            for run in series.runs:
                gt_traj = run.get_data('gt_traj')
                loc_traj = run.get_data('amcl_traj') if ('amcl_traj' in run.keys()) else run.get_data('gmapping_traj')
                dists = pu.list_distances(gt_traj)
                temp_data['x_data'].append(dists[0])
                max_dist = max(max_dist, dists[1])
                temp_data['y_data'].append(pu.yaw_error(loc_traj,gt_traj))
                temp_data['series_name'].append(name)
                temp_data['color'].append(series.colors[0])
                temp_data['line'].append('-')

            if self.aggregate:
                
                x_bins = list(np.linspace(0,max_dist, max_dist/0.1))
                y_store = [0]*len(x_bins)

                for i in range(len(temp_data['x_data'])):
                    dists = temp_data['x_data'][i]
                    min_x = dists[0]
                    max_x = dists[-1]
                    diffs = temp_data['y_data'][i]
                    f = interpolate.interp1d(dists,diffs);

                    j = 0
                    for x in x_bins:
                        if (x <= max_x and x >= min_x):
                            if (y_store[j] != 0):
                                y_store[j][0]+=f(x)
                                y_store[j][1]+=1
                            else:
                                y_store[j] = [f(x), 1] # [0] stores diff sums, [1] stores number of additions
                        j+=1

                temp_data['x_data'] = [x_bins]
                y_data = [0]*len(x_bins)
                for i in range(len(y_store)):
                    y_data[i] = float(y_store[i][0])/(float(y_store[i][1]))

                temp_data['y_data'] = [y_data]

                temp_data['series_name'] = [name]
                temp_data['color'] = [series.colors[0]]
                temp_data['line']= ['-']

            for key in self.data:
               self.data[key] += temp_data[key]
               
    def info(self):
        res = ""
        self.data.clear()
        self.data = {'x_data':[],'y_data':[], 'series_name': [], 'color' : [], 'line': []}
        temp = self.aggregate
        self.aggregate = True
        self.collect_data()
        self.aggregate = temp

        for i in range(len(self.data['series_name'])):
            x = self.data['x_data'][i]
            y = self.data['y_data'][i]
            avg_error = 0
            for j in range(len(self.data['x_data'][i])-1):

               avg_error += ((y[j]+y[j+1])/2.0)*(x[j+1]-x[j])
            
            avg_error = avg_error/(x[-1]-x[0])

            res+= "Series: " + bc.color(self.data['series_name'][i],bc.HEADER) + " has an average yaw error of {:.5f} rad\n".format(avg_error)
            
        return res


class TrajectoryPlot(Plot):
    '''A plot of the birds-eye view of the robots trajectory'''

    def __init__(self, aggregate=False, only_gt=True):
        Plot.__init__(self, aggregate)
        self.only_gt = only_gt 


    def label(self):
        self.ax.set_title('Trajectory Plot')
        self.ax.set(xlabel='x position (m)', ylabel = 'y position (m)')

    def init_axis(self):
        self.collect_data()
        for i in range(len(self.data['x_data'])):
            self.ax.plot(self.data['x_data'][i], self.data['y_data'][i], self.data['line'][i], label = self.data['series_name'][i], color = self.data['color'][i])
        self.ax.axis('equal')
        self.ax.legend()

    def collect_data(self):
        self.data = {'x_data':[],'y_data':[], 'series_name': [], 'color' : [], 'line': []}
        for series in self.series_list:
            for run in series.runs:
                gt_traj = run.get_data('gt_traj')
                self.data['x_data'].append(gt_traj['pos_x'])
                self.data['y_data'].append(gt_traj['pos_y'])
                self.data['series_name'].append(series.name+" ground truth")
                self.data['color'].append(series.colors[0])
                self.data['line'].append('-')

                if not self.only_gt:
                    loc_traj = run.get_data('amcl_traj') if ('amcl_traj' in run.keys()) else run.get_data('gmapping_traj')
                    self.data['x_data'].append(loc_traj['pos_x'])
                    self.data['y_data'].append(loc_traj['pos_y'])
                    self.data['series_name'].append(series.name+" localization")
                    self.data['color'].append(series.colors[1])
                    self.data['line'].append('--')


    def info(self):
        res = "Trajectory plot: no printable info available"
        return res

class PathDifference(Plot):
    ''' A plot of the percent difference between the robots path and the optimal path to complete the tour. This plot is only meaningfull if only successfull trials are included '''  

    def label(self):
        self.ax.set_title("Path Difference")
        self.ax.set(xlabel='Series', ylabel = 'Average percent difference from optimal path length (%)')

    def init_axis(self):
        self.collect_data()
        #self.ax.bar(self.data['x_data'], self.data['y_data'], color = self.data['color'], yerr = self.data['yerr'])
        bplot = self.ax.boxplot(self.data['y_data'], labels = self.data['x_data'], patch_artist = True, showfliers=False)
        for patch, color in zip(bplot['boxes'], self.data['color']):
            patch.set_facecolor(color)
        
    def collect_data(self):
        self.data = {'x_data':[],'y_data':[], 'series_name': [], 'color' : [], 'line': [], 'yerr': []}
        for series in self.series_list:
            self.data['x_data'].append(series.name)
            self.data['color'].append(series.colors[0])
            # find average path difference for the runs of the current series
            diffs = []
            for run in series.runs:
                data = run.data
                meta = run.meta
                optimal_dist = pu.list_distances(data['optimal_traj'])[1]
                gt_dist = pu.list_distances(data['gt_traj'])[1]
                if ((gt_dist-optimal_dist) < 0):
                    continue
                diffs.append(((gt_dist-optimal_dist)/optimal_dist)*100)

            #self.data['y_data'].append(np.average(diffs))
            self.data['y_data'].append(diffs)
            #self.data['yerr'].append(np.std(diffs))
    
    def info(self):
        if (len(self.data['x_data']) == 0):
            self.collect_data()
        
        res = ""
        for i in range(len(self.data['x_data'])):
            series = self.data['x_data'][i]
            diff = self.data['y_data'][i]
            res += "Series: " + bc.color(str(series), bc.HEADER) + " has a path that deviates {:.3f} % from the optimal path\n".format(diff)

        return res

class SuccessRate(Plot):
    ''' A plot of the success rate of a given series (a successful run is one where the robot completes the whole tour '''

    def label(self):
       self.ax.set_title("Success Rate")
       self.ax.set(xlabel='Series', ylabel = 'Success Rate (%)')


    def init_axis(self):
       self.collect_data()
       self.ax.bar(self.data['x_data'], self.data['y_data'])
       
    def collect_data(self):
       self.data = {'x_data':[],'y_data':[], 'series_name': [], 'color' : [], 'line': []}
       for series in self.series_list:
           self.data['x_data'].append(series.name)
           
           rate = 0
           for run in series.runs:
               data = run.data
               meta = run.meta
               if (meta['success_status'] == 'true'):
                   rate += 1
                   
           rate = float(rate)/len(series.runs)
           self.data['y_data'].append(rate*100)

    def info(self):
        if (len(self.data['x_data']) == 0):
            self.collect_data()

        res = ""
        for i in range(len(self.data['x_data'])):
            series = self.data['x_data'][i]
            succ = self.data['y_data'][i]
            res += "Series: " + bc.color(str(series), bc.HEADER) + " has a success rate of {:.3f} %\n".format(succ)

        return res

class RunHandler:

    def __init__(self):
        '''Handles the storage, searching, and modification of runs in the run_data_2.json file'''
        # initialize variables 
        self.username = os.environ['USER']
        self.filepath = '/home/' + self.username +'/Myhal_Simulation/'

        try:
            self.json_f = open(self.filepath+'run_data_2.json', 'r+')
            self.table = json.load(self.json_f)
        except:
            self.json_f = open(self.filepath+'run_data_2.json', 'w')
            self.table = {
                'tour_names': {},
                'filter_status': {'true': [], 'false':[]},
                'localization_technique': {},
                'success_status':{'true':[], 'false':[]},
                'scenarios':{},
                'class_method':{},
                'localization_test':{'true':[],'false':[]},
                'load_world':{},
                'times':[]}


        # add new runs 
        logging.info('Loading run metadata') 
        t1 = time.time()
        self.dirs = os.listdir(self.filepath + 'simulated_runs/')
        self.run_map = {}
        self.run_inds = []

        for name in self.dirs:
            self.read_run(name)

        # clean old runs
        for name in self.table['times']:
            if (name not in self.dirs):
                self.delete_run(name)

        # rewrite to run_data_2.json
        self.update_json()
        logging.info("Loaded and updated metadata in {:.2f} s".format(time.time() - t1))

    def update_json(self):
        self.json_f.seek(0)
        self.json_f.truncate()
        logging.info('Writing to run_data_2.json')
        json.dump(self.table, self.json_f, indent = 4, sort_keys=True)
        self.json_f.close()
        self.json_f = open(self.filepath+'run_data_2.json', 'r+')

    def user_remove_file(self, name):
        ''' lets the user remove a run from the simulated runs folder '''
        if (name in self.dirs):
            self.dirs.remove(name)
            if (name in self.table['times']):
                self.delete_run(name) 
                self.update_json()
            shutil.rmtree(self.filepath + 'simulated_runs/' + name)
            logging.info(name + ' successfully removed')
        else:
            logging.info('Directory ' + name + ' does not exist')

    def delete_run(self, name):
        if (name not in self.table['times']):
            logging.debug('Cannot delete ' + name + ' from run_data_2.json')
            return
        self.table['times'].remove(name)
        for key, value in self.table.items():
            if (key == 'times'):
                continue
            for k,v in value.items():
                if name in v:
                    v.remove(name)
        if (name in self.run_map):
            run = self.run_map.pop(name)
            del run
            logging.debug('Removed pickled data for ' + name)
        if (name in self.run_inds):
            self.run_inds.remove(name)
            logging.debug('Removed ' + name + ' from run_inds')

        logging.info('Deleted ' + name + ' from run_data_2.json')

    def read_run(self, name):
        ''' adds the run of name run to self.run_map if it has valid metadata. If it is not already present, adds the run to self.table '''

        def add_or_append(field, meta_d):
            if (type(meta_d[field]) == list):
                for att in meta_d[field]:
                    if (att in self.table[field] and name not in self.table[field][att]):
                        self.table[field][att].append(name)
                    elif (att not in self.table[field]):
                        self.table[field][att] = [name]
                return
                           
            if (meta_d[field] in self.table[field]):
                self.table[field][meta_d[field]].append(name)
            else:
                self.table[field][meta_d[field]] = [name]
            
        try:
            meta_f = open(self.filepath + 'simulated_runs/'+name + '/logs-' +name + '/meta.json')
            meta_d = json.load(meta_f)
            meta_f.close()
        except:
            logging.debug(name + " has a malformed or missing meta.json file")
            return
        
        # Read target reached
        with open(self.filepath + 'simulated_runs/'+name + '/logs-' +name + '/log.txt', 'rb') as txt_f:
            content = txt_f.readlines()
        successes = [int(line.startswith('Reached target')) for line in content 
                     if line.startswith('Reached target') or line.startswith('Failed to')]

        if len(successes) > 0:
            meta_d['targets_reached'] = float(np.sum(successes)) / len(successes)
            meta_d['success_status'] = bool(successes[-1])
        else:
            meta_d['targets_reached'] = -0.01
            meta_d['success_status'] = False



        # Read proportion of actors/tables
        line_i0 = -1
        for line_i, line in enumerate(content):
            if line.startswith('Scenario: ') and not line.startswith('Scenario: empty'):
                line_i0 = line_i
        
        if line_i0 > 0:
            meta_d['tables_p'] = float(content[line_i0 + 1].split()[-1])
            meta_d['actors_p'] = float(content[line_i0 + 2].split()[-1])
        else:
            meta_d['tables_p'] = 0
            meta_d['actors_p'] = 0

        self.run_map[name] = Run(name, meta_d)
        self.run_inds.append(name)
        self.run_inds.sort(reverse = True)

        if (name in self.table['times']):
            return

        for field in self.table:
            if (field == 'times'):
                continue
            add_or_append(field,meta_d)

        self.table['times'].append(name)
        logging.info('Added ' + name + ' to run_data_2.json')

    def gather_run(self, name):
        ''' deserializes the pickle data for the given run and stores it in self.run_map, returns true if the run exists and has its data loaded'''

        if (name not in self.run_map):
            return False

        if (self.run_map[name].data != None):
            return True

        runpath =self.filepath + 'simulated_runs/'+name + '/logs-' +name + '/' 
        logging.debug('Adding data for ' + name)
        try:
            data_f = open(runpath + 'processed_data.pickle')
            data_d = pickle.load(data_f)
            data_f.close()
            RAM = psutil.virtual_memory().percent 
            logging.debug('RAM taken: ' + str(RAM))
            if (RAM > 95):
                logging.critical('TOO MUCH RAM TAKEN TO LOAD DATA')
                exit()
        except:
            logging.warning('Could not deserialize processed_data.pickle for ' + name + ', removing from run_data_2.json')
            self.delete_run(name)
            return False

        # Save a ply of the traj if not already theregt_traj = run.get_data('gt_traj')
        loc_ply_file = self.filepath + 'simulated_runs/' + name + '/loc_pose.ply'
        if not os.path.exists(loc_ply_file):
            loc_traj = data_d['amcl_traj'] if ('amcl_traj' in data_d.keys()) else data_d['gmapping_traj']
            print(loc_traj.shape)
            print(loc_traj.dtype)
            el = ply.PlyElement.describe(loc_traj, "trajectory")
            ply.PlyData([el]).write(loc_ply_file)

        self.run_map[name].data = data_d
        return True


    def search(self, tour_names = None, filter_status = None, localization_technique = None, success_status = None, scenarios = None, earliest_date = None, latest_date = None, localization_test = None, class_method = None, load_world = None, date = None):
        '''returns a list of run corrisponding the given search, loads the data for those runs and adds it to self.run_map'''

        loc_l = locals()
        # create a set of all available runs (by name)
        results = set(self.run_map.keys())

        # first search by time field:
        if (date):
            results.intersection_update(set([date]))

        if (earliest_date):
            ed_int = mu.date_to_int(earliest_date)
            for rem in list(results):
                rem_int = mu.date_to_int(rem)
                if (rem_int < ed_int):
                    results.remove(rem)

        if (latest_date):
            lt_int = mu.date_to_int(latest_date)
            for rem in list(results):
                rem_int = mu.date_to_int(rem)
                if (rem_int > lt_int):
                    results.remove(rem)

        #handle all other fields with a single function 
        for par, arg in loc_l.items():
            if (par not in self.table or not arg):
                continue
            if (arg not in self.table[par]):
                return set()

            c_set = set(self.table[par][arg])
            results.intersection_update(c_set)

        res = []   
        for name in results:
            self.gather_run(name)
            res.append(self.run_map[name])

        return res
            
    

class Series:

    cl = mcolors.BASE_COLORS.keys()
    cl.remove('w')
    np.random.shuffle(cl)
    c_cycle = cycle(cl)

    def __init__(self, name, runs, colors = None):
        ''' an object for storing runs of a common set of characeristics which are going to be plotted'''

        self.runs = runs # stores a list of Run objects
        if (not colors or len(colors) < 2):
            self.select_colors()
        else:
            self.colors = colors
        self.name = name
        
    def select_colors(self):
       self.colors = [Series.c_cycle.next(), Series.c_cycle.next()] 

class Display:

    def __init__(self, rows, cols):
        self.rows = rows 
        self.cols = cols 
        self.plots = []
        self.series_map = {}

    def dim(self):
        return (self.rows,self.cols)

    def size(self):
        return self.rows * self.cols

    def add_series(self,series):
        self.series_map[series.name] = series

    def add_plot(self, plot):
        if (len(self.plots) >= self.size()):
            self.rows+=1
            logging.info('Adding row to display to accept plot')
        self.plots.append(plot)


    def display(self, path = None):
        self.init_plots()
        self.fig.set_size_inches((30,15), forward = False)
        if (path is not None):
            try:
                plt.savefig(path)
            except:
                logging.error('Could not save plot')
        plt.show()

    def init_plots(self):
        self.fig, axs = plt.subplots(self.rows,self.cols)
        axs = np.array(axs)
        i = 0
        for ax in axs.reshape(-1):
            if (i >= len(self.plots)):
                break
            self.plots[i].add_ax(ax)
            self.plots[i].series_list = self.series_map.values()
            self.plots[i].init_axis()
            i+=1

class Dashboard:
    ''' An interface between the user and Myhal Simulation data '''

    def __init__(self,verbosity = logging.INFO, rows = 1, cols =1):
        '''Optional arguments
        verbosity: specified the level of logging to STOUT, logging.DEBUG, logging.INFO, logging.WARNING ..., default = logging.INFO
        rows: the number of rows on the plot display, default = 1
        cols: the number of columns on the plot display, default = 1
        '''
    
        logging.basicConfig(level=verbosity, format = '%(levelname)s - %(message)s')
        self.handler = RunHandler()
        self.display = Display(rows,cols)

    def list_runs_helper(self, l):

        header = [bc.color('Index', bc.BOLD),
                  bc.color('Name', bc.BOLD),
                  bc.color('Filtering', bc.BOLD),
                  bc.color('Classification Method', bc.BOLD),
                  bc.color('Tour Name', bc.BOLD),
                  bc.color('Success', bc.BOLD),
                  bc.color('Targets', bc.BOLD),
                  bc.color('Localization Method', bc.BOLD),
                  bc.color('Scenarios', bc.BOLD),
                  bc.color('Actors_p', bc.BOLD)]

        fields = ['filter_status',
                  'class_method',
                  'tour_names',
                  'success_status',
                  'targets_reached',
                  'localization_technique',
                  'scenarios',
                  'actors_p']
        res = []
        c = 0
        for name in l:
            run = self.handler.run_map[name]
            run_l = [c, bc.color(name, bc.WARNING)]
            for f in fields:
                if (f == 'scenarios'):
                    scens = run.meta[f]
                    i = 0
                    while (i < len(scens)):
                        if (scens.count(scens[i]) > 1):
                            scens.pop(i)
                            continue
                        scens[i] = scens[i].encode('utf-8')
                        i+=1    
                    # compress the scenario name for display
                    particules = scens[0].split('_')[:2]
                    run_l.append('_'.join([part[:4] for part in particules]))
                elif (f == 'targets_reached'):
                    run_l.append('{:7.1f}%'.format(100*run.meta[f]))
                else:
                    run_l.append(run.meta[f])


            c+=1
            res.append(run_l)

        print tabulate(res,headers=header)

    def list_runs(self, num = 10):
        ''' displays meta data for the num most recent runs '''
        l = self.handler.run_inds[:num]
        self.list_runs_helper(l)
            
    def add_series(self, name, colors = None, tour_names = None, filter_status = None, localization_technique = None, success_status = None, scenarios = None, earliest_date = None, latest_date = None, localization_test = None, class_method = None, load_world = None, date = None, min_ind = None, max_ind = None, ind = None):
        ''' adds a series to the dashboard, where all runs in the series adhere to all specified parameters'''

        if (latest_date is None):
            latest_date = self.ind_to_date(min_ind)
        if (earliest_date is None):
            earliest_date = self.ind_to_date(max_ind)
        if (date is None):
            date = self.ind_to_date(ind)
        
        series = Series(name, self.handler.search(tour_names, filter_status, localization_technique, success_status, scenarios, earliest_date ,latest_date, localization_test, class_method, load_world, date), colors)

        self.display.add_series(series)

    def add_plot(self, plot):
        ''' add a plot to the dashboard '''
        self.display.add_plot(plot)

    def remove_series(self, name):
        ''' remove named series from dashboard '''
        if (name in self.display.series_map):
            self.display.series_map.pop(name)
        else:
            logging.info('Series ' + name + ' not found in display')

    def clear_series(self):
        ''' remove all series from the dashboard '''
        self.display.series_map.clear()

    def clear_plots(self):
        ''' remove all plots from the dashboard '''
        self.display.plots = []

    def clear(self):
        ''' remove all plots and series ''' 
        self.clear_plots()
        self.clear_series()

    def remove_plot(self, plot_class):
        ''' remove the given plot class from the dashboard '''
        i = 0
        while (i < len(self.display.plots)):
            if (isinstance(self.display.plots[i],plot_class)):
                self.display.plots.pop(i)
                continue
            i+=1

    def resize(self, rows, cols):
        ''' resizes the display of this dashboard '''
        self.display.rows= rows
        self.display.cols = cols

    def show(self, path = None, font_size = 14):
        ''' show current display as a plot, save the plot to path if given '''
        matplotlib.rcParams.update({'font.size': font_size})
        self.display.display(path)

    def ind_to_date(self, ind):
        if (ind is not None and (ind >=0 and ind < len(self.handler.run_inds))):
            date = self.handler.run_inds[ind]
        else:
            date = None
        return date

    def nori_to_date(self, name_or_ind):
        if (type(name_or_ind) == int):
            name = self.ind_to_date(name_or_ind)
        else:
            name = name_or_ind
        return name

    def rviz_run(self, name_or_ind, rate = 1):
        ''' play the bag file of the named run with a pre-configured rviz file '''
        name = self.nori_to_date(name_or_ind)
        if (name is None or not os.path.isdir(self.handler.filepath + '/simulated_runs/' + name)):
            logging.info('Invalid name or index')
            return

        ls = os.listdir(self.handler.filepath + '/simulated_runs/' + name)
        username = self.handler.username
        if ('raw_data.bag' in ls):
            script = "/home/"+username+"/catkin_ws/scripts/visualize_bag.sh -l " + name + " -r " + str(rate)
        else:
            script = "/home/"+username+"/catkin_ws/scripts/visualize_bag.sh -l " + name + " -r " + str(rate) + " -n localization_test.bag"

        subprocess.call(script, shell = True)

    def run_info(self, name_or_ind):
        ''' print the meta data of the named run '''
        name = self.nori_to_date(name_or_ind)
        if (name is None):
            logging.info('Invalid name or index')
            return

        header = [bc.color('Name',bc.BOLD), bc.color('Filtering Status',bc.BOLD), bc.color('Classification Method',bc.BOLD), bc.color('Tour Name',bc.BOLD), bc.color('Success Status',bc.BOLD), bc.color('Localization Method',bc.BOLD), bc.color('Scenarios',bc.BOLD) , bc.color('Localization Test',bc.BOLD)]
        fields = ['filter_status','class_method','tour_names', 'success_status', 'localization_technique', 'scenarios', 'localization_test']
        if (name not in self.handler.run_map):
            logging.info('Run ' + name + ' not found')
            return
        
        run = self.handler.run_map[name]
        run_l = [bc.color(name,bc.WARNING)]
        for f in fields:
            if (f == 'scenarios'):
                scens = run.meta[f]
                i = 0
                while (i < len(scens)):
                    if (scens.count(scens[i]) > 1):
                        scens.pop(i)
                        continue
                    scens[i] = scens[i].encode('utf-8')
                    i+=1    
                run_l.append(',\n'.join(scens))
                continue

            run_l.append(run.meta[f])

        print tabulate([run_l], headers=header)

    def list_dirs(self):
        ''' list all directories in the experimental folder and whether or not they are a valid run '''
        header = [bc.color('Name',bc.BOLD), bc.color('Is Valid Run',bc.BOLD)]
        res = []
        for name in self.handler.dirs:
            if (name in self.handler.run_inds):
                res.insert(0, [bc.color(name, bc.WARNING), 'true'])
            else:
                res.append([name, 'false'])

        print tabulate(res, headers = header)

    def remove_dir(self, name = None, clear_old = False):
        ''' remove names directory from experiments folder if it exists '''
        if (clear_old and name is None):
            confirm = raw_input('Type DEL to confirm you want to delete all old runs\n')
            if (confirm != 'DEL'):
                return

            dirs = self.handler.dirs[:]
            for dir in dirs:
                if (not os.path.isdir(self.handler.filepath + 'simulated_runs/'  + dir) or dir in self.handler.run_map):
                    continue 
                self.handler.user_remove_file(dir) 
            return

        if name not in self.handler.dirs:
            logging.info(str(name) + ' does not exist and cannot be deleted')
            return
        confirm = raw_input('Type DEL to confirm you want to delete ' + name + '\n')
        if (confirm == 'DEL'):
            self.handler.user_remove_file(name)

    def remove_series_dirs(self, name):
        ''' remove all the directories of a given series '''   
        if (name not in self.display.series_map):
            logging.info('Series ' + name + ' does not exist in the dashboard')
        
        series = self.display.series_map[name]
        runs = series.runs
        for i in range(len(runs)):
            runs[i] = runs[i].name
        print 'Runs to be removed:'
        self.list_runs_helper(runs)
        confirm = raw_input('Input DEL to confirm series deletion\n')
        if (confirm == 'DEL'):
            for run in runs:
                self.handler.user_remove_file(run)
            self.remove_series(name)
        logging.info('Series ' + name + ' sucessfully deleted')

    def plot_run(self, name_or_ind, plot_type, colors = None, filepath = None):
        ''' plot the named run with the given plot type (not on the display), return the plot'''
        name = self.nori_to_date(name_or_ind)
        if (name is None):
            logging.info('Invalid name or index')
            return
        if (isinstance(plot_type, Plot)):
            T = plot_type
        else:
            T = plot_type()
        if (name not in self.handler.run_map):
            logging.info('Run ' + name + ' not found')
        temp_display = Display(1,1)
        temp_display.add_plot(T)
        temp_display.add_series(Series(name, self.handler.search(date = name), colors))

        temp_display.display(filepath)
        return temp_display.plots[0]

    def plot_info(self, plot_class):
        ''' return textual information about the desired plot type '''
        found = False
        for plot in self.display.plots:
            plot.series_list = self.display.series_map.values()
            if (isinstance(plot, plot_class)):
                print bc.color(plot.__class__.__name__ + ": ",bc.BOLD)
                found = True
                print plot.info()

        if (not found):
            logging.info('Plot type not found in display')

    def watch(self, name_or_ind, save_path = ""):
        name = self.nori_to_date(name_or_ind)
        if (name is None or not os.path.isdir(self.handler.filepath + '/simulated_runs/' + name)):
            logging.info('Invalid name or index')
            return

        vid_path = self.handler.filepath + 'simulated_runs/' + name + '/logs-' + name + '/videos/'
        if (os.path.isdir(vid_path)):
            vids = os.listdir(vid_path)
        else:
            logging.info('Video Files Not Found')
            return
        
        print "Available videos for " + name + ":"
        for i in range(len(vids)):
            vid = vids[i]
            print "[" + str(i) + "] " + vid

        ind = input("Input the index you would like to view/save:\n")
        if (type(ind) != int or ind < 0 or ind >= len(vids)):
            print "Invalid index"
            return

        if (save_path != ""):
            print "Saving " + vids[ind] + " to " + save_path
            shutil.copyfile(vid_path + vids[ind], save_path)

        print "Playing " + vids[ind]
        
        subprocess.call("xdg-open " + vid_path + vids[ind], shell = True)
