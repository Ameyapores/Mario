from __future__ import print_function
from scipy.misc import imresize
import torch, os, argparse, sys, time, glob
from setup_env import setup_env
import torch.nn as nn
import torch.nn.functional as F
from scipy.signal import lfilter
import torch.multiprocessing as mp
import numpy as np
from mario_icm import ActorCritic
os.environ['OMP_NUM_THREADS'] = '1'

def printlog(args, s, end='\n', mode='a'):
    print(s, end=end) ; f=open(args.save_dir+'log.txt',mode) ; f.write(s+'\n') ; f.close()

discount = lambda x, gamma: lfilter([1],[1,-gamma],x[::-1])[::-1] # discounted rewards one liner
prepro = lambda img: imresize(img[35:195].mean(2), (84,84)).astype(np.float32).reshape(1,84,84)/255.

class SharedAdam(torch.optim.Adam): # extend a pytorch optimizer so it shares grads across processes
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        super(SharedAdam, self).__init__(params, lr, betas, eps, weight_decay)
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['shared_steps'], state['step'] = torch.zeros(1).share_memory_(), 0
                state['exp_avg'] = p.data.new().resize_as_(p.data).zero_().share_memory_()
                state['exp_avg_sq'] = p.data.new().resize_as_(p.data).zero_().share_memory_()
                
        def step(self, closure=None):
            for group in self.param_groups:
                for p in group['params']:
                    if p.grad is None: continue
                    self.state[p]['shared_steps'] += 1
                    self.state[p]['step'] = self.state[p]['shared_steps'][0] - 1 # a "step += 1"  comes later
            super.step(closure)

def cost_func(args, values, logps, actions, rewards):
    np_values = values.view(-1).data.numpy()

    # generalized advantage estimation using \delta_t residuals (a policy gradient method)
    delta_t = np.asarray(rewards) + args.gamma * np_values[1:] - np_values[:-1]
    logpys = logps.gather(1, torch.tensor(actions).view(-1,1))
    gen_adv_est = discount(delta_t, args.gamma * args.tau)
    policy_loss = -(logpys.view(-1) * torch.FloatTensor(gen_adv_est.copy())).sum()
    
    # l2 loss over value estimator
    rewards[-1] += args.gamma * np_values[-1]
    discounted_r = discount(np.asarray(rewards), args.gamma)
    discounted_r = torch.tensor(discounted_r.copy(), dtype=torch.float32)
    value_loss = .5 * (discounted_r - values[:-1,0]).pow(2).sum()

    entropy_loss = -(-logps * torch.exp(logps)).sum() # encourage lower entropy
    return policy_loss + 0.5 * value_loss + 0.01 * entropy_loss
def train(shared_model, shared_optimizer, rank, args, info):
    env = setup_env(args.env) # make a local (unshared) environment
    #env.seed(args.seed + rank) ; torch.manual_seed(args.seed + rank) # seed everything
    model = ActorCritic(channels=1, memsize=args.hidden, num_actions=args.num_actions) # a local/unshared model
    state = torch.tensor(prepro(env.reset()))
    start_time = last_disp_time = time.time()
    episode_length, epr, eploss, done  = 0, 0, 0, True

    while info['frames'][0] <= 4e7 or args.test: # openai baselines uses 40M frames...we'll use 80M
        model.load_state_dict(shared_model.state_dict()) # sync with shared model

        hx = torch.zeros(1, 256) if done else hx.detach()  # rnn activation vector
        values, logps, actions, rewards = [], [], [], []

        for step in range(args.rnn_steps):
            episode_length += 1
            value, logit, hx = model((state.view(1,1,84,84), hx))
            logp = F.log_softmax(logit, dim=-1)

            action = torch.exp(logp).multinomial(num_samples=1).data[0]#logp.max(1)[1].data if args.test else
            state, reward, done, _ = env.step(action.numpy()[0])

            