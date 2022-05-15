import subprocess
import argparse
import json
import copy
import itertools
from multiprocessing import Pool
import time


def run_experiment(experiment):
    cmd = ['python', 'pretrain_maze.py']
    for argument in experiment:
        time.sleep(5)
        cmd.append(argument)
    
    return subprocess.check_output(cmd)


if __name__ == '__main__':
    default = ['maximum_timestep=50', 'use_wandb=true', 'agent.batch_size=64', 
    'agent.init_alpha=0.2', 'agent.feature_dim=128','agent.hidden_dim=128',
    'num_pretrain_frames=510000','oracle_dur=50000', 'num_pretrain_frames=2010000',
    'maze_type=square_bottleneck']
    seeds = ['seed=100', 'seed=101','seed=102', 'seed=103','seed=104', 'seed=105']
    num_devices = 4
    num_exp_per_device = 3
    pool_size = num_devices * num_exp_per_device

    experiments = []
    device = 0
    for seed in seeds:
        exp = copy.deepcopy(default)
        
        exp.append(seed)
        
        device_id = int(device % num_devices)
        exp.append(f'device_id={device_id}')
        experiments.append(exp)
        device += 1

    pool = Pool(pool_size)
    stdouts = pool.map(run_experiment, experiments, chunksize=1)
    pool.close()