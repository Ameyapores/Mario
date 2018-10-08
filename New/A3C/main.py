import os
import argparse
import gym
import numpy as np
import torch
import torch.cuda
import torch.multiprocessing as _mp

from setup_env import setup_env
from model import ActorCritic
from shared_adam import SharedAdam
from Train import train

SAVEPATH = os.getcwd() + '/save/mario_a3c_params.pkl'

parser = argparse.ArgumentParser(description='A3C')
parser.add_argument('--lr', type=float, default=0.0001,
                    help='learning rate (default: 0.0001)')
parser.add_argument('--gamma', type=float, default=0.9,
                    help='discount factor for rewards (default: 0.9)')
parser.add_argument('--tau', type=float, default=1.00,
                    help='parameter for GAE (default: 1.00)')
parser.add_argument('--entropy-coef', type=float, default=0.01,
                    help='entropy term coefficient (default: 0.01)')
parser.add_argument('--value-loss-coef', type=float, default=0.5,
                    help='value loss coefficient (default: 0.5)')
parser.add_argument('--max-grad-norm', type=float, default=250,
                    help='value loss coefficient (default: 50)')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed (default: 4)')
parser.add_argument('--num-processes', type=int, default=6,
                    help='how many training processes to use (default: 4)')
parser.add_argument('--num-steps', type=int, default=50,
                    help='number of forward steps in A3C (default: 50)')
parser.add_argument('--max-episode-length', type=int, default=10000,
                    help='maximum length of an episode (default: 1000000)')
parser.add_argument('--env-name', default='SuperMarioBros-1-1-v0',
                    help='environment to train on (default: SuperMarioBros-1-4-v0)')
parser.add_argument('--no-shared', default=False,
                    help='use an optimizer without shared momentum.')
parser.add_argument('--use-cuda',default=True,
                    help='run on gpu.')
parser.add_argument('--save-interval', type=int, default=100,
                    help='model save interval (default: 10)')
parser.add_argument('--save-path',default=SAVEPATH,
                    help='model save interval (default: {})'.format(SAVEPATH))
parser.add_argument('--non-sample', type=int,default=2,
                    help='number of non sampling processes (default: 2)')
parser.add_argument('--render', default=False, type=bool, help='renders the atari environment')
parser.add_argument('--test', default=False, type=bool, help='sets lr=0, chooses most likely actions')

mp = _mp.get_context('spawn')
print("Cuda: " + str(torch.cuda.is_available()))
if __name__ == '__main__':
    os.environ['OMP_NUM_THREADS'] = '1'

    args = parser.parse_args()
    env = setup_env(args.env_name)
    if args.render:  args.num_processes = 1 ; args.test = True # render mode -> test mode w one process
    if args.test:  args.lr = 0 # don't train in render mode

    shared_model = ActorCritic(1, env.action_space.n)
    if args.use_cuda:
        shared_model.cuda()
    shared_model.share_memory()

    if os.path.isfile(args.save_path):
        print('Loading A3C parametets ...')
        shared_model.load_state_dict(torch.load(args.save_path))

    optimizer = SharedAdam(shared_model.parameters(), lr=args.lr)
    optimizer.share_memory()

    print ("No of available cores : {}".format(mp.cpu_count())) 

    processes = []
    counter = mp.Value('i', 0)
    lock = mp.Lock()

    no_sample = args.non_sample   
    sample_val = args.num_processes - no_sample
    for rank in range(0, args.num_processes):
        if rank < sample_val:                           # select random action
            p = mp.Process(target=train, args=(rank, args, shared_model, counter, lock, optimizer))
        else:                                           # select best action
            p = mp.Process(target=train, args=(rank, args, shared_model, counter, lock, optimizer, False))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()