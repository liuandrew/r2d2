import os
import traceback
import matplotlib.pyplot as plt
import sys
from pathlib import Path
import re
import scipy.interpolate
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
sys.path.insert(0, '..')

import numpy as np
import pandas as pd
import proplot as pplt


# Extraction function
def tflog2pandas(path: str) -> pd.DataFrame:
    """convert single tensorflow log file to pandas DataFrame
    Parameters
    ----------
    path : str
        path to tensorflow log file
    Returns
    -------
    pd.DataFrame
        converted dataframe
    """
    DEFAULT_SIZE_GUIDANCE = {
        "compressedHistograms": 1,
        "images": 1,
        "scalars": 0,  # 0 means load all
        "histograms": 1,
    }
    runlog_data = pd.DataFrame({"metric": [], "value": [], "step": []})
    try:
        event_acc = EventAccumulator(path, DEFAULT_SIZE_GUIDANCE)
        event_acc.Reload()
        tags = event_acc.Tags()["scalars"]
        for tag in tags:
            event_list = event_acc.Scalars(tag)
            values = list(map(lambda x: x.value, event_list))
            step = list(map(lambda x: x.step, event_list))
            r = {"metric": [tag] * len(step), "value": values, "step": step}
            r = pd.DataFrame(r)
            runlog_data = pd.concat([runlog_data, r])
    # Dirty catch of DataLossError
    except Exception:
        print("Event file possibly corrupt: {}".format(path))
        traceback.print_exc()
    return runlog_data



def print_runs(folder='../runs/', prin=True, descriptions=False, ret=False,
               exclude=1):
    '''
    print tensorboard runs directories
    
    prints out each unique experiment name and how many trials were associated
    next use load_exp_df with the name and trial number to load it into a df
    
    also plan to update this function as more runs are added with 
    descriptions of what each run is, if setting descriptions to True
        will print these out
        
    !!
    print, descriptions, and ret are all broken due to adding in directory
        inner_prints. Not hard to fix but no reason to atm
    !!
        
    new_run_descriptions:
        We split the naming of the run into sections based on underscores e.g.
        {nav}_{env_condition}_{model_condition}_{trial_num}
        [0]: 'nav': continuous navigation environment
        [1]: env_condition: 
            'c1': 1 Wall Color
            'c2': 2 Wall Colors Symmetric
            'c2.5': 2 Wall Colors Asymmetric
            'c4': 4 Wall Colors
            'pproxim': Poster on Proximal Wall (East wall)
            'pdistal': Poster on Distal Wall (North wall)
        [2]: model_condition
            'none': Basic model, goal only reward, nothing special
            'dist': Distance based reward shaping
            'auxeuclid0': No aux task, equivalent to 'none'
            'auxeuclid1': Null aux task, report constant 0 every timestep
            'auxeuclid2': Euclidean distance task, report Euclidean distance
                from start position
            'auxwall[0-3]': Wall reportin aux task, report angle needed to rotate
                to face the given wall
            'shared[0-2]': Same as 'none' but change network structure. Select how many
                layers are shared between actor and critic. Default is 0 so 0 is equivalent
                to 'none'
    exclude:
        1: don't print any grid nav or visible platform trials, or non nav trials

    '''
    space =  '    '
    branch = '│   '
    tee =    '├── '
    last =   '└── '
    
    run_descriptions = {
        'invisible_shared': '2D Nav, 4 Wall Colors, change how many layers in the NN' + \
                            'are shared after RNN. 0 is default',
        'nav_invisible_shared': 'Cont Nav, 4 Wall Colors, shared layers, same as invisible_shared',
        'nav_aux_wall': 'Cont Nav, 4 Wall Colors, Auxiliary task of reporting relative' + \
                            'angle to wall. 1 is proximal, 3 is distal',
        'nav_euclid_start': 'Cont Nav, 4 Wall Colors, Euclidean distance auxiliary task. ' + \
                            '0: no task. 1: null task (report 0), 2: euclidean distance task from start pos',
        'nav_invisible_color': 'Cont Nav, variable wall colors and reward structure. dist: reward shaping, ' + \
                               'none: no reward shaping. 2: Symmetrical, 2.5: Asymmetrical',
        'nav_visible': 'Cont Nav, 1 Wall Color, visible randomized platform with or w/o reward shaping baseline',
        'nav_aux_p[wall]_[aux_task]': 'Cont Nav, Poster distal or proximal, Auxiliary tasks of ' + \
                        'wall reporting or euclidean reporting (see nav_aux_wall and nav_euclid_start)',
    }
    

    
    path = Path(folder)
    if prin:
        print(path.name)
    ignore_dirs = []
    if exclude >= 1:
        ignore_dirs += ['invisible_poster', 'invisible_shared', 'invisible_wallcolors',
                        'nav_visible_reshaping', 'visible_reshaping', 'visible_wallcolors',
                        'acrobot', 'cartpole', 'mountaincar', 'pendulum']    
    
    def inner_print(path, depth):
        directories = []
        unique_experiments = {}
        original_experiment_names = {}
        for d in path.iterdir():
            if '__' in d.name:
                trial_name = d.name.split('__')[0]
                if not re.match('.*\d*', trial_name):
                    #not a trial, simply print
                    print(branch*depth+tee+d.name)
                exp_name = '_'.join(trial_name.split('_')[:-1])
                if exp_name in unique_experiments.keys():
                    unique_experiments[exp_name] += 1
                else:
                    unique_experiments[exp_name] = 1
                    original_experiment_names[exp_name] = d.name
            elif d.is_dir() and d.name not in ignore_dirs:
                directories.append(d)
        if prin:
            for key, value in unique_experiments.items():
                if value > 1:
                    print(branch*depth + tee+'EXP', key + ':', value)
                else:
                    print(branch*depth+tee+original_experiment_names[key])
        
        result_dict = unique_experiments.copy()
        for i, d in enumerate(directories):
            if prin:
                print(branch*depth + tee+d.name)
            sub_experiments = inner_print(d, depth+1)
            result_dict[d] = sub_experiments
        
        return result_dict
            
    directory_dict = inner_print(path, 0)    
    if ret:
        return directory_dict



def load_exp_df(exp_name=None, path=None, trial_num=0, folder='../runs/', save_csv=True):
    '''
    load experiment tensorboard run into a dataframe
    if save_csv is True, also convert into a csv for faster loading later
    
    if trial_num is None, directly try to load the given exp_name e.g., 
    load_exp_df(exp_name='nav_c4_shared0_t0__1661300055', 
            folder='../runs/nav_invisible_shared', trial_num=None)
        
    if path is given, directly use this path run folder to load e.g.,
    load_exp_df(path='../runs/nav_invisible_shared/nav_c4_shared0_t0__1661300055')

    path > exp_name in precedence if both parameters passed
    '''
    
    if path is None:
        if trial_num is not None:
            results = []
            if '/' in exp_name:
                sub_folders = '/'.join(exp_name.split('/')[:-1]) + '/'
                folder = folder + sub_folders
                exp_name = exp_name.split('/')[-1]
            dirs = os.listdir(folder)
            for d in dirs:
                if '__' in d:
                    trial_name = d.split('__')[0]
                    if re.match('.*\_\d*', trial_name):
                        name = '_'.join(trial_name.split('_')[:-1])
                        trial = trial_name.split('_')[-1]
                        if 't' in trial:
                            trial = trial.split('t')[-1]
                        trial = int(trial)

                        if exp_name == name and trial == trial_num:
                            results.append(folder + d)

            if len(results) > 1:
                print('Warning: more than one experiment with the name and trial num found')
            if len(results) == 0:
                print('No experiments found')
                return None

            path = results[0]
        else:
            path = folder + exp_name
    files = os.listdir(path)
    
    #look for a preconverted csv dataframe file
    for file in files:
        if '.csv' in file:
            df = pd.read_csv(path + '/' + file)
            return df
    
    #no preconverted csv found, convert the events file
    for file in files:
        if 'events.out.tfevents' in file:
            df = tflog2pandas(path + '/' + file)
            
            if save_csv:
                df.to_csv(path + '/' + 'tflog.csv')
            
            return df
        
    print('No tf events file correctly found')
    return results


def plot_exp_df(df, smoothing=0.1):
    '''
    Plot the experiments values from tensorboard df
    (get the df from load_exp_df)
    ''' 
    fig, ax = pplt.subplots(nrows=3, ncols=4, share=False)
    titles = list(df['metric'].unique())
    for i in range(12-len(titles)):
        titles.append('')
    ax.format(title=titles)
    
    for i, chart in enumerate(df['metric'].unique()):
        idx = df['metric'] == chart
        df.loc[idx, 'ewm'] = df.loc[idx, 'value'].ewm(alpha=smoothing).mean()
        d = df[df['metric'] == chart]
        
        ax[i].plot(d['step'], d['value'], c='C0', alpha=0.1)
        ax[i].plot(d['step'], d['ewm'], c='C0')


        
def average_runs(trial_name, metric='return', ax=None, ret=False, ewm=0.01,
                label=None, cloud_alpha=0.1, cloud_by='minmax', ignore_first=16, 
                color=None, medians=False, div_x_by_mil=False, ls=None, verbose=False,
                ignore_trial_idxs=[]):
    '''
    Get the average over a bunch of trials of the same name
    
    trial_name: Name of experiment, 
    metric: Name of metric to plot, some shortcuts can be passed like
        value_loss, policy_loss, return, length
    ax: Optionally pass ax to plot on
    ewm: whether to do an exponential average of metric
        if not wanted, pass False
    ret: if True, instead of plotting, return the xs, ys, min_x, max_x
    label: whether to plot with a label
    cloud_alpha: alpha to show cloud of trials (0 for invisible)
    cloud_by: choose how cloud should be calculated. Options:
        'minmax': plot the min and max performance
        'iqr': plot interquartile range
        'std': plot 1 standard deviation range
    ignore_first: ignore_first n elements to filter out noisy first data
    medians: plot medians rather than means
    div_x_by_mil: if True, divide x values by 1e6, so that we can label in the xlabel
        that it was x10^6 instead of as part of xticks
    ls: linestyle. e.g., '--' is close dash, (0, (5, 5)) is spaced dash
    ignore_trial_idxs: list of indexes to not include, for example if removing outliers
    '''
    shortcut_to_key = {
        'value_loss': 'losses/value_loss',
        'policy_loss': 'losses/policy_loss',
        'aux_loss': 'losses/auxiliary_loss',
        'return': 'charts/episodic_return',
        'length': 'charts/episodic_length'
    }
    if ax is None and ret is False:
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    
    exp_name = trial_name.split('/')[-1]
    base_folder = Path('../runs/')
    # exps = print_runs(prin=False, ret=True) #just borrow function to get number of experiments
    folder = '/'.join(trial_name.split('/')[:-1])
    if len(folder) == 0:
        folder = '.'
    folder = base_folder/folder
    if verbose: print(folder)        
    trials = list(folder.iterdir())
    trial_names = [item.name.split('__')[0] for item in folder.iterdir()]
    
    #Remove any trial names without numbers (not a trial folder)
    keep_trial_names = []
    keep_trials = []
    for i, name in enumerate(trial_names):
        if re.search('\d+', name):
            keep_trial_names.append(name)
            keep_trials.append(trials[i])
    trial_names = keep_trial_names
    trials = keep_trials
    
    trial_nums = [item.split('_')[-1] for item in trial_names]
    trial_names = ['_'.join(item.split('_')[:-1]) for item in trial_names]
    # print(trial_names)
    # print(trial_nums)
    trial_nums = [int(re.search('\d+', num)[0]) for num in trial_nums]
            
    if metric in shortcut_to_key:
        metric = shortcut_to_key[metric]

    # for i, trial in enumerate(trial_names):
    #     print(trials[i], trial == exp_name)
    
    if exp_name not in trial_names:
        print('No experiments with the given name found in runs folder')    
    else:
        # num_trials = exps[trial_name]
        trial_idxs = np.array(trial_names) == exp_name
        if verbose: print(np.sum(trial_idxs))
        exps = np.array(trials)[trial_idxs]
        
        if verbose: print(exps)
                
        # Averaging code same as from plot_cloud_from_dict in data_visualize.ipynb
        first_xs = []
        last_xs = []
        inters = []
        num_trials = 0        
        
        for i, exp in enumerate(exps):
            if i in ignore_trial_idxs:
                continue
            # Load df from run file
            # df = load_exp_df(trial_name, i)
            if verbose:
                print(str(exp))
            df = load_exp_df(path=str(exp))
            df = df[df['metric'] == metric]
            if len(df) < 1:
                raise Exception('No metric called {} found in {}'.format(
                    metric, exp))
            
            first_xs.append(df.iloc[0]['step'])
            last_xs.append(df.iloc[-1]['step'])
            if ewm:
                df['ewm'] = df['value'].ewm(alpha=ewm).mean()
                inter = scipy.interpolate.interp1d(df['step'], df['ewm'])
                inters.append(inter)
            else:
                inter = scipy.interpolate.interp1d(df['step'], df['value'])
            num_trials += 1
        min_x = np.max(first_xs)
        max_x = np.min(last_xs)
        xs = np.arange(min_x, max_x, 200)
        ys = np.zeros((num_trials, len(xs)))
        
        for j in range(num_trials):
            ys[j] = inters[j](xs)
        
        if ret is False:
            if medians:
                y_mid = np.median(ys, axis=0)[ignore_first:]
            else:
                y_mid = np.mean(ys, axis=0)[ignore_first:]
            
            
            if cloud_by == 'minmax':
                cloud_low = np.min(ys, axis=0)[ignore_first:]
                cloud_high = np.max(ys, axis=0)[ignore_first:]
            elif cloud_by == 'iqr':
                cloud_low = np.percentile(ys, 25, axis=0)[ignore_first:]
                cloud_high = np.percentile(ys, 75, axis=0)[ignore_first:]
            elif cloud_by == 'std':
                std = np.std(ys, axis=0)[ignore_first:]
                cloud_low = y_mid - std
                cloud_high = y_mid + std
            else:
                raise('No proper cloud_by option given. Should be "minmax", "iqr", or "std"')

            if div_x_by_mil:
                xs = xs / 1000000

            if ewm:
                if color is not None:
                    ax.fill_between(xs[ignore_first:], cloud_low, 
                                cloud_high, alpha=cloud_alpha, color=color)
                else:
                    ax.fill_between(xs[ignore_first:], cloud_low, 
                                cloud_high, alpha=cloud_alpha)
                    
            h = ax.plot(xs[ignore_first:], y_mid, label=label, color=color, ls=ls)
            return h
        else:
            return xs, ys, min_x, max_x
