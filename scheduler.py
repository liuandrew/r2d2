import pickle
from datetime import datetime
import os
import shutil

import argparse


CONFIG_FOLDER = 'experiment_configs/'
'''
File for running training experients from experiment_configs folder
Also has functions for handling storing and reading experiment config files
    and experiment log
If running this file directly from command line, will iterate through experiment
    config files and run the experiments
'''

def convert_config_to_command(file=None, config=None, python3=False, cont=False,
                              config_folder=CONFIG_FOLDER):
    '''
    when passed a file name in experiment_configs, load the config and turn it into
    a command line line to run the experiment
    '''
    if file != None:
        config = pickle.load(open(config_folder + file, 'rb'))
    if config == None:
        raise ValueError('No file or config given')
    if python3:
        # run_string = 'python3 main.py '
        run_string = 'python3 r2d2_algo/r2d2.py '
    else:
        # run_string = 'python main.py '
        run_string = 'python r2d2_algo/r2d2.py '
    for key in config:
        if config[key] is True:
            run_string = run_string + '--' + key.replace('_', '-') + ' '
        elif type(config[key]) == dict:
            run_string = run_string + '--' + key.replace('_', '-') + ' '
            add_str = ''
            for key2 in config[key]:
                add_str += key2 + '=' + str(config[key][key2]).replace(' ', '') + ' '
            run_string = run_string + add_str
        elif type(config[key]) == list:
            run_string = run_string + '--' + key.replace('_', '-') + ' ' + str(config[key]).replace(' ', '') + ' '
        else:
            run_string = run_string + '--' + key.replace('_', '-')  + ' ' + str(config[key]) + ' '
    if cont:
        run_string = run_string + '--cont '
    # #additionally add file name flag
    # if file != None:
    #     run_string = run_string + '--config-file-name ' + file + ' '
    # print(run_string)
    return run_string



# def reset_exp_log():
#     '''
#     reset the experiment log in case of write error
#     '''
#     exp_log = pd.DataFrame(columns=['file', 'begin', 'end', 'exp_name', 'save_name', 'num_env_steps',
#        'env_name', 'recurrent_policy', 'algo', 'num_mini_batch',
#        'num_processes', 'success', 'env_kwargs', 'wandb_project_name',
#        'capture_video', 'track', 'recurrent'])
#     pickle.dump(exp_log, open('experiment_log', 'wb'))
#     return exp_log




def save_exp_log(exp_log):
    '''
    save an updated experiment log
    '''
    pickle.dump(exp_log, open('experiment_log', 'wb'))
    



def load_exp_log():
    '''
    load experiment log to globals
    '''
    try:
        exp_log = pickle.load(open('experiment_log', 'rb'))
    except:
        exp_log = reset_exp_log()
    return exp_log


    

def add_exp_row(file):
    '''
    Add a config to the experiment log
    '''
    exp_log = load_exp_log()
    config = pickle.load(open(CONFIG_FOLDER + file, 'rb'))
    index = len(exp_log)
    exp_log = exp_log.append(config, ignore_index=True)
    exp_log.loc[index, 'begin'] = datetime.now()
    exp_log.loc[index, 'file'] = file
    save_exp_log(exp_log)




def write_latest_exp_complete(file):
    '''
    Write the time at which the experiment is completed for a certain filename
    Note that if there are multiple entries in the experiment log with the 
    same filename, we will just pick the one with highest index to update
    '''
    exp_log = load_exp_log()

    idx = exp_log[exp_log['file'] == file].index.max()
    exp_log.loc[idx, 'end'] = datetime.now()
    exp_log.loc[idx, 'success'] = True
    save_exp_log(exp_log)




def run_experiment(file, python3=False, cont=False):
    '''
    Pass a config file to run an experiment
    Save the experiment to experiment log
    If the experiment is successfully complete ('end' column is filled)
        then archive the config file
    Otherwise delete the row that was added
    '''
    # if cont is False:
    #     add_exp_row(file)
    run_string = convert_config_to_command(file, python3, cont)
    os.system(run_string)

    # exp_log = load_exp_log()
    # idx = exp_log[exp_log['file'] == file].index.max()
    # if exp_log.loc[idx, 'success'] is True:
    #     #experiment completed successfully
    ext = str(int(datetime.now().timestamp()))
    shutil.move(CONFIG_FOLDER + file, CONFIG_FOLDER + 'archive/' + file + ext)
    # else:
    #     exp_log.loc[idx, 'success'] = False

    # save_exp_log(exp_log)
    



if __name__ == "__main__":
    '''
    If running scheduler, go through experiment_configs folder and run each of the configs
    '''
    
    def strtobool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')
    
    parser = argparse.ArgumentParser(description='RL')
    parser.add_argument('--python3', type=lambda x:bool(strtobool(x)), default=False, nargs='?', const=True,
        help='flag for whether command line should use python3 or python')

    parser.add_argument('--manual', type=str, default=None,
        help='whether to manually run a single experiment, and what file name it is')

    parser.add_argument('--cont', type=lambda x:bool(strtobool(x)), default=False, nargs='?', const=True,
        help='if toggled, attempt to load a model as named from save_path under the right folder to continue experiment')

    args = parser.parse_args()
    # print(args.python3)

    files = os.listdir(CONFIG_FOLDER)

    for file in files:
        # Only run files that include the same text as used in --manual argument when running scheduler.py
        if args.manual is not None:
            if args.manual not in file:
                continue
            
        if file not in ['.ipynb_checkpoints', 'archive', '.gitkeep']:
            print('running experiment: ', file)
            run_experiment(file, python3=args.python3, cont=args.cont)
            print('experiment complete')