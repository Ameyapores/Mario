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
from action import ACTIONS


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

    for num_iter in count() or args.test:
        if num_iter % 10000 ==0 and num_iter > 0:
                print('\n\t{:.0f}M frames: saved model\n'.format(num_iter/1e6))
                torch.save(shared_model.state_dict(), args.save_path)

        model.load_state_dict(shared_model.state_dict())
        if done:
            cx = Variable(torch.zeros(1, 512)).type(FloatTensor)
            hx = Variable(torch.zeros(1, 512)).type(FloatTensor)
        else:
            cx = Variable(cx.data).type(FloatTensor)
            hx = Variable(hx.data).type(FloatTensor)

        values, log_probs, rewards, entropies = [], [], [], []
        actions, inverses, forwards, vec_st1s = [], [], [], []
        
        for step in range(args.num_steps):
            episode_length += 1            
            state_inp = Variable(state.unsqueeze(0)).type(FloatTensor)
            inp1= (state_inp, (hx, cx))
            value, logit, (hx, cx) = model(inp1, False)
            s_t= state
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
            #print ('action', action_out)
            out = action_out[0][0]
            action_out1 = torch.from_numpy(ACTIONS[out])
            action_out1 = torch.unsqueeze(action_out1, 0)

            state, reward, done, _ = env.step(action_out.numpy()[0][0])
            if args.render: env.render()
            done = done or episode_length >= args.max_episode_length
            
            #print ('reward1:', reward)
            reward = max(min(reward, 1), -1)

            a_t =action_out1
            a_t= a_t.type(FloatTensor)
            #print ('a_t', a_t)

            actions.append(a_t)
            state = torch.tensor(prepro(state))
            s_t1 = state
            inp2= (Variable(s_t.unsqueeze(0)).type(FloatTensor), Variable(s_t1.unsqueeze(0)).type(FloatTensor), a_t)
            
            vec_st1, inverse, forward = model(inp2,True)
            inverse1= inverse.max(-1, keepdim=True)[1].data
            inverse1= inverse1.cpu()
            #print ('inverse', inverse1)
            inverse2= inverse1[0][0]
            act_hat =  torch.from_numpy(ACTIONS[inverse2])
            act_hat = torch.unsqueeze(act_hat, 0)
            act_hat=act_hat.type(FloatTensor)
            #print ('inverse:', act_hat)            
            reward_intrinsic = args.eta * ((vec_st1 - forward).pow(2)).sum(1) / 2.
            reward_intrinsic = reward_intrinsic.cpu()
            #print ('reward_int:', reward_intrinsic)
            reward += reward_intrinsic
            #print ('total_reward:', reward)
            reward_sum += reward.detach().numpy()
            #print ('reward_sum', reward_sum)
            with lock:
                counter.value += 1

            if rank == 0 and time.time() - last_disp_time > 60: # print info ~ every minute
                print("Time {}, num steps {}, FPS {:.0f}, episode reward {}, episode length {}, episode loss {}".format(
                    time.strftime("%Hh %Mm %Ss",
                                  time.gmtime(time.time() - start_time)), 
                    counter.value, counter.value / (time.time() - start_time),
                    reward_sum[0], episode_length, average_loss))
                data = [time.time() - last_disp_time,
                        counter.value, reward_sum[0], episode_length, average_loss]
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
            reward= reward.type(FloatTensor)
            rewards.append(reward)
            forwards.append(forward)
            vec_st1s.append(vec_st1)
            actions.append(a_t)
            inverses.append(act_hat)

        R = torch.zeros(1, 1)
        if not done:
            state_inp = Variable(state.unsqueeze(0)).type(FloatTensor)
            inp3= (state_inp, (hx, cx))
            value, _, _ = model(inp3, False)
            R = value.data

        values.append(Variable(R).type(FloatTensor))
        policy_loss, value_loss, inverse_loss, forward_loss = 0, 0, 0, 0
        R = Variable(R).type(FloatTensor)
        gae = torch.zeros(1, 1).type(FloatTensor)
        #print (len(forwards), len(vec_st1s), reversed(range(len(rewards))))
        for i in reversed(range(len(rewards))):
            R = args.gamma * R + rewards[i]
            advantage = R - values[i]
            value_loss = value_loss + 0.5 * advantage.pow(2)

            delta_t = rewards[i] + args.gamma * \
                values[i + 1].data - values[i].data
            gae = gae * args.gamma * args.tau + delta_t

            policy_loss = policy_loss - \
                log_probs[i] * Variable(gae).type(FloatTensor) - args.entropy_coef * entropies[i]

            forward_err = forwards[i] - vec_st1s[i]
            forward_loss = forward_loss + 0.5 * (forward_err.pow(2)).sum(1)
            cross_entropy = - (actions[i] * torch.log(inverses[i])).sum(1)
            inverse_loss = inverse_loss + cross_entropy
            inverse_loss = inverse_loss.type(FloatTensor)

        total_loss = policy_loss + args.value_loss_coef * value_loss + args.beta * forward_loss
        loss1= total_loss.cpu()
        average_loss += loss1.detach().numpy()[0][0]
        optimizer.zero_grad()

        (total_loss).backward(retain_graph=True)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

        ensure_shared_grads(model, shared_model)
        optimizer.step()