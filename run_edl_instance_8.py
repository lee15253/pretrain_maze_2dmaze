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
    default = ['use_wandb=true', 'agent.batch_size=64', 
    'agent.init_alpha=0.2', 'agent.feature_dim=128','agent.hidden_dim=128',
    'num_pretrain_frames=510000','oracle_dur=50000', 'num_train_frames=21000000']
    seeds = ['seed=105', 'seed=101','seed=102', 'seed=103','seed=104']
    maze_type_maximum_timestep = [
        ('maze_type=square_bottleneck', 'maximum_timestep=50','agent.skill_dim=80', 'agent.vae_args.codebook_size=80', 'agent.max_skill_dim=80'),
        ('maze_type=square_large','maximum_timestep=200','agent.skill_dim=80', 'agent.vae_args.codebook_size=80', 'agent.max_skill_dim=80'),
        ('maze_type=square_upside','maximum_timestep=200','agent.skill_dim=95', 'agent.vae_args.codebook_size=95', 'agent.max_skill_dim=95')
     ]
    num_devices = 4
    num_exp_per_device = 4
    pool_size = num_devices * num_exp_per_device

    experiments = []
    device = 0
    for mtmt, seed in itertools.product(*[maze_type_maximum_timestep, seeds]):
        exp = copy.deepcopy(default)
        
        exp.append(mtmt[0])
        exp.append(mtmt[1])
        exp.append(mtmt[2])
        exp.append(mtmt[3])
        exp.append(mtmt[4])
        exp.append(seed)
        
        device_id = int(device % num_devices)
        exp.append(f'device_id={device_id}')
        experiments.append(exp)
        device += 1
        
    pool = Pool(pool_size)
    stdouts = pool.map(run_experiment, experiments, chunksize=1)
    pool.close()