from setup_env import setup_env
from model import ActorCritic, ICM
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

def train(rank, args, shared_model1, shared_model2, counter, lock, optimizer=None, select_sample=True):
    FloatTensor = torch.cuda.FloatTensor if args.use_cuda else torch.FloatTensor
    DoubleTensor = torch.cuda.DoubleTensor if args.use_cuda else torch.DoubleTensor
    ByteTensor = torch.cuda.ByteTensor if args.use_cuda else torch.ByteTensor

    env = setup_env(args.env_name)
    model1 = ActorCritic(1, env.action_space.n)
    model2= ICM(1, env.action_space.n)

    if args.use_cuda:
        model1.cuda(), model2.cuda()

    if optimizer is None:
        #optimizer1 = optim.Adam(shared_model1.parameters(requires_grad=False), lr=args.lr)
        optimizer = optim.Adam(shared_model1.parameters(), lr=args.lr)
    
    model1.train()
    model2.train()

    state= prepro(env.reset())
    state = torch.from_numpy(state)

    savefile = os.getcwd() + '/save/mario_curves.csv'
    title = ['Time','No. Steps', 'Total Reward', 'Episode Length']
    with open(savefile, 'a', newline='') as sfile:
        writer = csv.writer(sfile)
        writer.writerow(title)  
    start_time = time.time()  

    done = True
    episode_length = 0
    for num_iter in count():
        if rank == 0:
            if num_iter % args.save_interval == 0 and num_iter > 0:
                #print ("Saving model at :" + args.save_path)            
                torch.save(shared_model2.state_dict(), args.save_path2)

        if num_iter % (args.save_interval * 2.5) == 0 and num_iter > 0 and rank == 1:    # Second saver in-case first processes crashes 
            #print ("Saving model for process 1 at :" + args.save_path)            
            torch.save(shared_model2.state_dict(), args.save_path2)

        model1.load_state_dict(shared_model1.state_dict())
        model2.load_state_dict(shared_model2.state_dict())

        if done:
            cx = Variable(torch.zeros(1, 512)).type(FloatTensor)
            hx = Variable(torch.zeros(1, 512)).type(FloatTensor)
        else:
            cx = Variable(cx.data).type(FloatTensor)
            hx = Variable(hx.data).type(FloatTensor)

        values = [], log_probs = [], rewards = [], entropies = [], actions= [], inverses= [], forwards=[], vec_st1s=[]
        actions = deque(maxlen=4000)
        ep_start_time = time.time()

        for step in range(args.num_steps):
            episode_length += 1     
            state_inp = Variable(state.unsqueeze(0)).type(FloatTensor)
            value, logit, (hx, cx) = model1((state_inp, (hx, cx)))
            s_t = state
            prob = F.softmax(logit, dim=-1)
            log_prob = F.log_softmax(logit, dim=-1)
            entropy = -(log_prob * prob).sum(-1, keepdim=True)
            entropies.append(entropy)

            action = prob.max(-1, keepdim=True)[1].data
            
            log_prob = log_prob.gather(-1, Variable(action))
            
            action_out = action.to(torch.device("cpu"))
            out = action[0][0]
            action_out1 = torch.from_numpy(ACTIONS[out])
            action_out1 = torch.unsqueeze(action_out1, 0)
            
            state, reward, done, _ = env.step(action_out.numpy()[0][0])
            if args.render: env.render()
            done = done or episode_length >= args.max_episode_length
            reward = max(min(reward, 1), -1)

            a_t =action_out1
            a_t= a_t.type(FloatTensor)
            #print ("action_out:", a_t)
            
            actions.append(a_t)
            actions.append(a_t)
            state = torch.tensor(prepro(state))
            s_t1 = state

            vec_st1, inverse, forward = model2(
                    Variable(s_t.unsqueeze(0)).type(FloatTensor),
                    Variable(s_t1.unsqueeze(0)).type(FloatTensor),
                    a_t.type(FloatTensor)
                )

            inverse1= inverse.max(-1, keepdim=True)[1].data
            out2= inverse1[0][0]
            inverse_out2 = torch.from_numpy(ACTIONS[out2])
            inverse_out2 = torch.unsqueeze(inverse_out2, 0)
            inverse_out2 = inverse_out2.type(FloatTensor)
            #print ("inverse_out", inverse_out2)
            #print (inverse)
            reward_intrinsic = args.eta * ((vec_st1 - forward).pow(2)).sum(1) / 2.
            reward_intrinsic = reward_intrinsic.to(torch.device("cpu"))
            reward_intrinsic = reward_intrinsic.data.numpy()[0]
            reward += reward_intrinsic

            with lock:
                counter.value += 1

            if actions.count(actions[0]) == actions.maxlen:
                done = True
            
            if done:
                episode_length = 0
                #env.change_level(0)
                state = torch.from_numpy(prepro(env.reset()))
                print("Time {}, num steps {}, FPS {:.0f}, episode reward {}, episode length {}".format(
                    time.strftime("%Hh %Mm %Ss",
                                  time.gmtime(time.time() - start_time)), 
                    counter.value, counter.value / (time.time() - start_time),
                    reward, episode_length))
            
                data = [time.time() - ep_start_time,
                        counter.value, reward, episode_length]
            
                with open(savefile, 'a', newline='') as sfile:
                    writer = csv.writer(sfile)
                    writer.writerows([data])
            
            env.locked_levels = [False] + [True] * 31
            state = torch.from_numpy(prepro(state))
            values.append(value)
            log_probs.append(log_prob)
            rewards.append(reward)
            
            inverses.append(inverse_out2)
            forwards.append(forward)
            vec_st1s.append(vec_st1)

            if done:
                break

            R = torch.zeros(1, 1)

            if not done:
                state_inp = Variable(state.unsqueeze(0)).type(FloatTensor)
                value, _, _ = model1((state_inp, (hx, cx)))
                R = value.data

            
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
                    values[i + 1].data - values[i].data
                gae = gae * args.gamma * args.tau + delta_t

                policy_loss = policy_loss - \
                    log_probs[i] * Variable(gae).type(FloatTensor) - args.entropy_coef * entropies[i]
                
                forward_err = forwards[i] - vec_st1s[i]
                forward_loss = forward_loss + 0.5 * (forward_err.pow(2)).sum(1)
                cross_entropy = - (actions[i] * torch.log(inverses[i] + 1e-15)).sum(1)
                inverse_loss = inverse_loss + cross_entropy
                inverse_loss = inverse_loss.type(FloatTensor)

                optimizer.zero_grad()
                ((1-args.beta) * inverse_loss + args.beta * forward_loss).backward(retain_graph=True)
                (policy_loss + args.value_loss_coef * value_loss).backward(retain_graph=True)

                torch.nn.utils.clip_grad_norm_(model2.parameters(), args.max_grad_norm)

                ensure_shared_grads(model2, shared_model2)
                optimizer.step()



