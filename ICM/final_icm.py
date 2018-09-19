import math
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from model import ActorCritic
from torch.autograd import Variable
import time
from setup_env import setup_env
import argparse
import torch.multiprocessing as mp
from scipy.misc import imresize


def get_args():
    parser = argparse.ArgumentParser(description='A3C')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.0001)')
    parser.add_argument('--gamma', type=float, default=0.90, metavar='G',
                        help='discount factor for rewards (default: 0.99)')
    parser.add_argument('--tau', type=float, default=1.00, metavar='T',
                        help='parameter for GAE (default: 1.00)')
    parser.add_argument('--render', default=False, type=bool, help='renders the atari environment')
    parser.add_argument('--test', default=False, type=bool, help='sets lr=0, chooses most likely actions')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--num-processes', type=int, default=20, metavar='N',
                        help='how many training processes to use (default: 4)')
    parser.add_argument('--num-steps', type=int, default=20, metavar='NS',
                        help='number of forward steps in A3C (default: 20)')
    parser.add_argument('--max-episode-length', type=int, default=10000, metavar='M',
                        help='maximum length of an episode (default: 10000)')
    parser.add_argument('--env', default='SuperMarioBros-v0', metavar='ENV',
                        help='environment to train on (default: PongDeterministic-v3)')
    parser.add_argument('--no-shared', default=False, metavar='O',
                        help='use an optimizer without shared momentum.')
    parser.add_argument('--use-cuda',default=True,
                        help='run on gpu.')
    parser.add_argument('--horizon', default=0.99, type=float, help='horizon for running averages')
####################
    parser.add_argument('--eta', type=float, default=0.01, metavar='LR',
                        help='scaling factor for intrinsic reward')
    parser.add_argument('--beta', type=float, default=0.2, metavar='LR',
                        help='balance between inverse & forward')
    parser.add_argument('--lmbda', type=float, default=0.1, metavar='LR',
                        help='lambda : balance between A3C & icm')

    parser.add_argument('--outdir', default="../output", help='Output log directory')
    parser.add_argument('--record', action='store_true', help="Record the policy running video")
    return parser.parse_args()

def printlog(args, s, end='\n', mode='a'):
    print(s, end=end) ; f=open(args.save_dir+'log.txt',mode) ; f.write(s+'\n') ; f.close()
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
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                if group['weight_decay'] != 0:
                    grad = grad.add(group['weight_decay'], p.data)

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)

                denom = exp_avg_sq.sqrt().add_(group['eps'])

                bias_correction1 = 1 - beta1 ** state['step'].item()
                bias_correction2 = 1 - beta2 ** state['step'].item()
                step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1

                p.data.addcdiv_(-step_size, exp_avg, denom)

        return loss

def train(shared_model, shared_optimizer, rank, args, info, select_sample= True):
    FloatTensor = torch.cuda.FloatTensor if args.use_cuda else torch.FloatTensor

    env = setup_env(args.env)
    num_outputs= env.action_space.n
    model = ActorCritic(num_inputs=1, action_space= env.action_space)
    if args.use_cuda:
        model.cuda()
    state = torch.tensor(prepro(env.reset()))
    model.train()


    start_time = last_disp_time = time.time()
    episode_length, epr, eploss, done  = 0, 0, 0, True # bookkeeping

    while info['frames'][0] <= 40e7 or args.test:
        model.load_state_dict(shared_model.state_dict())
        if done:
            cx = Variable(torch.zeros(1, 512)).type(FloatTensor)
            hx = Variable(torch.zeros(1, 512)).type(FloatTensor)
        else:
            cx = Variable(cx.data).type(FloatTensor)
            hx = Variable(hx.data).type(FloatTensor)

        values, logps, rewards, entropies = [], [], [], []
        inverses, forwards, actions, vec_st1s = [], [], [], []

        for step in range(args.num_steps):
            value, logit, (hx, cx) = model(
                (Variable(state.unsqueeze(0)).type(FloatTensor), (hx, cx)),
                icm = False
            )
            s_t = state
            prob = F.softmax(logit, dim=-1)
            logp = F.log_softmax(logit, dim=-1)
            entropy = -(logp * prob).sum(-1, keepdim=True)
            entropies.append(entropy)

            action = prob.multinomial(num_samples=1).data
            action= action.to(torch.device("cpu"))
            
            logp = logp.gather(1, Variable(action))
            
            
            oh_action = torch.Tensor(1, num_outputs).type(FloatTensor)
            oh_action.zero_()
            oh_action.scatter_(1,action,1)
            oh_action = Variable(oh_action)
            a_t = oh_action
            
            actions.append(oh_action)

            state, reward, done, _ = env.step(action.numpy()[0][0])
            if args.render: env.render()
            state = torch.tensor(prepro(state))
            done = done or episode_length >= 1e4
            reward = max(min(reward, 1), -1)
            s_t1 = state

            vec_st1, inverse, forward = model(
                (
                    Variable(s_t.unsqueeze(0)).type(FloatTensor),
                    Variable(s_t1.unsqueeze(0)).type(FloatTensor),
                    a_t.type(FloatTensor)
                ),
                icm = True
            )            
            reward_intrinsic = args.eta * ((vec_st1 - forward).pow(2)).sum(1) / 2.
            
            reward_intrinsic = reward_intrinsic.to(torch.device("cpu"))
            reward_intrinsic = reward_intrinsic.data.numpy()[0]
            reward += reward_intrinsic
            epr += reward

            info['frames'].add_(1) ; num_frames = int(info['frames'].item())
            if num_frames % 2e6 == 0: # save every 2M frames
                printlog(args, '\n\t{:.0f}M frames: saved model\n'.format(num_frames/1e6))
                torch.save(shared_model.state_dict(), args.save_dir+'model.{:.0f}.tar'.format(num_frames/1e6))

            if done: # update shared data
                info['episodes'] += 1
                interp = 1 if info['episodes'][0] == 1 else 1 - args.horizon
                info['run_epr'].mul_(1-interp).add_(interp * epr)
                info['run_loss'].mul_(1-interp).add_(interp * eploss)

            if rank == 0 and time.time() - last_disp_time > 60: # print info ~ every minute
                elapsed = time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - start_time))
                printlog(args, 'time {}, episodes {:.0f}, frames {:.1f}M, mean epr {:.2f}, run loss {:.2f}'
                    .format(elapsed, info['episodes'].item(), num_frames/1e6,
                    info['run_epr'].item(), info['run_loss'].item()))
                last_disp_time = time.time()

            if done: # maybe print info.
                episode_length, epr, eploss = 0, 0, 0
                state = torch.tensor(prepro(env.reset()))
            values.append(value) ; logps.append(logp) ; actions.append(oh_action) ; rewards.append(reward) ; vec_st1s.append(vec_st1) ; inverses.append(inverse); forwards.append(forward)
        
        R = torch.zeros(1, 1).type(FloatTensor)
        if not done:
            value, _, _ = model(
                (Variable(state.unsqueeze(0)).type(FloatTensor), (hx, cx)),
                icm = False
            )
            R = value.data.type(FloatTensor)
        values.append(Variable(R).type(FloatTensor))
        
        policy_loss = 0
        value_loss = 0
        inverse_loss = 0
        forward_loss = 0
        gae = torch.zeros(1, 1).type(FloatTensor)
        R = Variable(R).type(FloatTensor)

        for i in reversed(range(len(rewards))):
            R = args.gamma * R + rewards[i]
            advantage = R - values[i]
            value_loss = value_loss + 0.5 * advantage.pow(2)

            delta_t = rewards[i] + args.gamma * \
                values[i + 1].data.type(FloatTensor) - values[i].data.type(FloatTensor)
            gae = gae * args.gamma * args.tau + delta_t

            policy_loss = policy_loss - \
                logps[i] * Variable(gae).type(FloatTensor) - 0.01 * entropies[i]

            cross_entropy = - (actions[i] * torch.log(inverses[i] + 1e-15)).sum(1)
            inverse_loss = inverse_loss + cross_entropy
            forward_err = forwards[i] - vec_st1s[i]
            forward_loss = forward_loss + 0.5 * (forward_err.pow(2)).sum(1)
            loss =(1-args.beta) * inverse_loss + args.beta * forward_loss + args.lmbda * (policy_loss + 0.5 * value_loss)
            eploss += loss.item()

        shared_optimizer.zero_grad()
        ((1-args.beta) * inverse_loss + args.beta * forward_loss).backward(retain_graph=True)
        (args.lmbda * (policy_loss + 0.5 * value_loss)).backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 40)
        for param, shared_param in zip(model.parameters(), shared_model.parameters()):
            if shared_param.grad is None: 
                return
            shared_param._grad = param.grad # sync gradients with shared model
        shared_optimizer.step()

if __name__ == "__main__":
    if sys.version_info[0] > 2:
        mp.set_start_method('spawn') # this must not be in global scope
    elif sys.platform == 'linux' or sys.platform == 'linux2':
        raise "Must be using Python 3 with linux!" # or else you get a deadlock in conv2d
    print("Cuda: " + str(torch.cuda.is_available()))

    args = get_args()
    args.save_dir = '{}/'.format(args.env.lower()) # keep the directory structure simple
    if args.render:  args.num_processes = 1 ; args.test = True # render mode -> test mode w one process
    if args.test:  args.lr = 0 # don't train in render mode
    args.num_actions = setup_env(args.env).action_space # get the action space of this game
    os.makedirs(args.save_dir) if not os.path.exists(args.save_dir) else None # make dir to save models etc.
    
    shared_model = ActorCritic(num_inputs=1, action_space =args.num_actions).share_memory()
    if args.use_cuda:
        shared_model.cuda()
    shared_optimizer = SharedAdam(shared_model.parameters(), lr=args.lr)

    info = {k: torch.DoubleTensor([0]).share_memory_() for k in ['run_epr', 'run_loss', 'episodes', 'frames']}
    info['frames'] += shared_model.try_load(args.save_dir) * 1e6
    if int(info['frames'].item()) == 0: printlog(args,'', end='', mode='w') # clear log file

    processes = []
    for rank in range(args.num_processes):
        p = mp.Process(target=train, args=(shared_model, shared_optimizer, rank, args, info))
        p.start() ; processes.append(p)
    for p in processes: p.join()
