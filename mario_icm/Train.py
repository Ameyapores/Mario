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
#from action import ACTIONS
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
    #torch.manual_seed(args.seed + rank)
    #print("Process No : {} | Sampling : {}".format(rank, select_sample))

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

    done = True
    episode_length = 0
    for num_iter in count():

        if rank == 0:
            #env.render()

            if num_iter % args.save_interval == 0 and num_iter > 0:
                #print ("Saving model at :" + args.save_path)            
                torch.save(shared_model.state_dict(), args.save_path)

        if num_iter % (args.save_interval * 2.5) == 0 and num_iter > 0 and rank == 1:    # Second saver in-case first processes crashes 
            #print ("Saving model for process 1 at :" + args.save_path)            
            torch.save(shared_model.state_dict(), args.save_path)
        
        model.load_state_dict(shared_model.state_dict())
        if done:
            cx = Variable(torch.zeros(1, 512)).type(FloatTensor)
            hx = Variable(torch.zeros(1, 512)).type(FloatTensor)
        else:
            cx = Variable(cx.data).type(FloatTensor)
            hx = Variable(hx.data).type(FloatTensor)

        values, log_probs, rewards, entropies = [], [], [], []
        forwards, vec_st1s = [], []
        
        for step in range(args.num_steps):
            episode_length += 1            
            state_inp = Variable(state.unsqueeze(0)).type(FloatTensor)
            value, logit, (hx, cx) = model((state_inp, (hx, cx)), False)
            s_t=state
            prob = F.softmax(logit, dim=-1)
            log_prob = F.log_softmax(logit, dim=-1)
            entropy = -(log_prob * prob).sum(-1, keepdim=True)
            entropies.append(entropy)
            
            if select_sample:
                action = prob.multinomial(num_samples=1).data
            else:
                action = prob.max(-1, keepdim=True)[1].data

            log_prob = log_prob.gather(-1, Variable(action))
            action_out = action.to(torch.device("cpu"))
            oh_action = torch.Tensor(1, env.action_space.n).type(torch.cuda.LongTensor)
            oh_action.zero_()
            oh_action.scatter_(1,action,1)
            a_t = oh_action.type(FloatTensor)
            #actions.append(oh_action)


            '''
            out = action_out[0][0]
            action_out1 = torch.from_numpy(ACTIONS[out])
            action_out1 = torch.unsqueeze(action_out1, 0)'''

            state, reward, done, _ = env.step(action_out.numpy()[0][0])
            done = done or episode_length >= args.max_episode_length
            reward = max(min(reward, 1), -1)

            #a_t =action_out1.type(FloatTensor)
            #actions.append(a_t)
            state = torch.tensor(prepro(state))
            s_t1 = state
            inp2= (Variable(s_t.unsqueeze(0)).type(FloatTensor), Variable(s_t1.unsqueeze(0)).type(FloatTensor), a_t)

            vec_st1, inverse, forward = model(inp2, True)
            reward_intrinsic = args.eta * ((vec_st1 - forward).pow(2)).sum(1) / 2.
            reward_intrinsic = reward_intrinsic.to(torch.device("cpu"))
            reward += reward_intrinsic

            with lock:
                counter.value += 1

            if done:
                episode_length = 0
                #env.change_level(0)
                state = torch.from_numpy(prepro(env.reset()))
                #print ("Process {} has completed.".format(rank))
            
            env.locked_levels = [False] + [True] * 31
            state = torch.from_numpy(prepro(state))
            values.append(value)
            log_probs.append(log_prob)
            reward= reward.type(FloatTensor)
            rewards.append(reward)
            forwards.append(forward)
            vec_st1s.append(vec_st1)
            #actions.append(a_t)
            
            if done:
                break

        R = torch.zeros(1, 1)
        if not done:
            state_inp = Variable(state.unsqueeze(0)).type(FloatTensor)
            value, _, _ = model((state_inp, (hx, cx)), False)
            R = value.data

        values.append(Variable(R).type(FloatTensor))
        policy_loss = 0
        value_loss = 0
        forward_loss = 0
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

            forward_err = forwards[i] - vec_st1s[i]
            forward_loss = forward_loss + 0.5 * (forward_err.pow(2)).sum(1)

        #total_loss = policy_loss + args.value_loss_coef * value_loss +
        #print ("Process {} loss :".format(rank), total_loss.data)

        optimizer.zero_grad()

        (args.beta * forward_loss).backward(retain_graph=True)
        (args.lmbda * (policy_loss + 0.5 * value_loss)).backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

        ensure_shared_grads(model, shared_model)
        optimizer.step()
    #print ("Process {} closed.".format(rank))

def test(rank, args, shared_model, counter):
    #torch.manual_seed(args.seed + rank)

    FloatTensor = torch.cuda.FloatTensor if args.use_cuda else torch.FloatTensor

    env = setup_env(args.env_name)
    #env.seed(args.seed + rank)

    model = ActorCritic(1, env.action_space.n)
    if args.use_cuda:
        model.cuda()
    model.eval()

    state = prepro(env.reset())

    state = torch.from_numpy(state)
    reward_sum = 0
    done = True
    savefile = os.getcwd() + '/save/mario_curves.csv'

    title = ['Time','No. Steps', 'Total Reward', 'Episode Length']
    with open(savefile, 'a', newline='') as sfile:
        writer = csv.writer(sfile)
        writer.writerow(title)    

    start_time = time.time()

    actions = deque(maxlen=4000)
    episode_length = 0
    while True:
        episode_length += 1
        ep_start_time = time.time()
        if done:
            model.load_state_dict(shared_model.state_dict())
            cx = Variable(torch.zeros(1, 512), volatile=True).type(FloatTensor)
            hx = Variable(torch.zeros(1, 512), volatile=True).type(FloatTensor)

        else:
            cx = Variable(cx.data, volatile=True).type(FloatTensor)
            hx = Variable(hx.data, volatile=True).type(FloatTensor)

        state_inp = Variable(state.unsqueeze(0), volatile=True).type(FloatTensor)
        value, logit, (hx, cx) = model((state_inp, (hx, cx)), False)
        prob = F.softmax(logit, dim=-1)
        action = prob.max(-1, keepdim=True)[1].data
        action_out = action.to(torch.device("cpu"))

        state, reward, done, _ = env.step(action_out.numpy()[0][0])
        #env.render()
        done = done or episode_length >= args.max_episode_length
        reward_sum += reward

        actions.append(action[0][0])
        if actions.count(actions[0]) == actions.maxlen:
            done = True

        if done:
            print("Time {}, num steps {}, FPS {:.0f}, episode reward {}, episode length {}".format(
                time.strftime("%Hh %Mm %Ss",
                              time.gmtime(time.time() - start_time)), 
                counter.value, counter.value / (time.time() - start_time),
                reward_sum, episode_length))
            
            data = [time.time() - ep_start_time,
                    counter.value, reward_sum, episode_length]
            
            with open(savefile, 'a', newline='') as sfile:
                writer = csv.writer(sfile)
                writer.writerows([data])
            
            reward_sum = 0
            episode_length = 0
            actions.clear()
            time.sleep(60)
            env.locked_levels = [False] + [True] * 31
            #env.change_level(0)
            state = prepro(env.reset())
        state = torch.from_numpy(prepro(state))