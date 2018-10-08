from setup_env import setup_env
from model import ActorCritic
import os
import time
from collections import deque
import csv
from scipy.misc import imresize
import numpy as np
import cv2
from itertools import count

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable


def ensure_shared_grads(model, shared_model):
    for param, shared_param in zip(model.parameters(),
                                   shared_model.parameters()):
        if shared_param.grad is not None:
            return
        shared_param._grad = param.grad

prepro = lambda img: imresize(img[0:84].mean(2), (84,84)).astype(np.float32).reshape(1,84,84)/255.

def train(rank, args, shared_model, counter, lock, optimizer=None, select_sample=True):

    FloatTensor = torch.cuda.FloatTensor if args.use_cuda else torch.FloatTensor
    
    env = setup_env(args.env_name)
    #env.seed(args.seed + rank)

    model = ActorCritic(1, env.action_space.n)

    if args.use_cuda:
        model.cuda()

    if optimizer is None:
        optimizer = optim.Adam(shared_model.parameters(), lr=args.lr)

    model.train()

    state= prepro(env.reset())
    state = torch.from_numpy(state)

    savefile = os.getcwd() + '/save/mario_curves.csv'
    title = ['Time','No. Steps', 'Total Reward', 'Episode Length', 'Episode loss']
    with open(savefile, 'a', newline='') as sfile:
        writer = csv.writer(sfile)
        writer.writerow(title)  

    start_time = last_disp_time= time.time()  
    episode_length, reward_sum, average_loss, done = 0, 0, 0, True

    for num_iter in count(2e5) or args.test:
        model.load_state_dict(shared_model.state_dict())
        if done:
            cx = Variable(torch.zeros(1, 512)).type(FloatTensor)
            hx = Variable(torch.zeros(1, 512)).type(FloatTensor)
        else:
            cx = Variable(cx.data).type(FloatTensor)
            hx = Variable(hx.data).type(FloatTensor)

        values, log_probs, rewards, entropies = [], [], [], []
        
        for step in range(args.num_steps):
            episode_length += 1            
            state_inp = Variable(state.unsqueeze(0)).type(FloatTensor)
            inp1= (state_inp, (hx, cx))
            value, logit, (hx, cx) = model(inp1)
            prob = F.softmax(logit, dim=-1)
            log_prob = F.log_softmax(logit, dim=-1)
            entropy = -(log_prob * prob).sum(-1, keepdim=True)
            entropies.append(entropy)
            
            if select_sample:
                action = prob.multinomial(num_samples=1).data
            else:
                action = prob.max(-1, keepdim=True)[1].data

            log_prob = log_prob.gather(-1, Variable(action))
            action_out = action.cpu()

            state, reward, done, _ = env.step(action_out.numpy()[0][0])
            if args.render: env.render()
            done = done or episode_length >= args.max_episode_length
            reward_sum += reward
            reward = max(min(reward, 1), -1)

            with lock:
                counter.value += 1

            if num_iter % 1e6 ==0:
                print('\n\t{:.0f}M frames: saved model\n'.format(num_iter/1e6))
                torch.save(shared_model.state_dict(), args.save_path)

            if rank == 0 and time.time() - last_disp_time > 60: # print info ~ every minute
                print("Time {}, num steps {}, FPS {:.0f}, episode reward {}, episode length {}, episode loss {}".format(
                    time.strftime("%Hh %Mm %Ss",
                                  time.gmtime(time.time() - start_time)), 
                    counter.value, counter.value / (time.time() - start_time),
                    reward_sum, episode_length, average_loss))
                data = [time.time() - last_disp_time,
                        counter.value, reward_sum, episode_length, average_loss]
                with open(savefile, 'a', newline='') as sfile:
                    writer = csv.writer(sfile)
                    writer.writerows([data])  
                last_disp_time = time.time()          
            
            if done:
                episode_length, reward_sum, average_loss = 0, 0, 0
                state = torch.from_numpy(prepro(env.reset()))
                
            env.locked_levels = [False] + [True] * 31
            state = torch.from_numpy(prepro(state))
            values.append(value)
            log_probs.append(log_prob)
            rewards.append(reward)

        R = torch.zeros(1, 1)
        if not done:
            state_inp = Variable(state.unsqueeze(0)).type(FloatTensor)
            inp3= (state_inp, (hx, cx))
            value, _, _ = model(inp3)
            R = value.data

        values.append(Variable(R).type(FloatTensor))
        policy_loss, value_loss = 0, 0
        R = Variable(R).type(FloatTensor)
        gae = torch.zeros(1, 1).type(FloatTensor)

        for i in reversed(range(len(rewards))):
            R = args.gamma * R + rewards[i]
            advantage = R - values[i]
            value_loss = value_loss + 0.5 * advantage.pow(2)

            delta_t = rewards[i] + args.gamma * \
                values[i + 1].data - values[i].data
            gae = gae * args.gamma * args.tau + delta_t

            policy_loss = policy_loss - \
                log_probs[i] * Variable(gae).type(FloatTensor) - args.entropy_coef * entropies[i]

        total_loss = policy_loss + args.value_loss_coef * value_loss 
        loss1= total_loss.cpu()
        average_loss += loss1.detach().numpy()[0][0]
        optimizer.zero_grad()

        (total_loss).backward(retain_graph=True)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

        ensure_shared_grads(model, shared_model)
        optimizer.step()