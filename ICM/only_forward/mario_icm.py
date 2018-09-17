import torch
import torch.nn as nn
import torch.nn.functional as F
import glob
import numpy as np


class ActorCritic(torch.nn.Module):
    
    def __init__(self, num_inputs, action_space):
        super(ActorCritic, self).__init__()
        self.conv1 = nn.Conv2d(num_inputs, 32, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(32, 32, 3, stride=2, padding=1)

        self.gru = nn.GRUCell(1152, 256)

        num_outputs = action_space.n
        self.critic_linear = nn.Linear(256, 1)
        self.actor_linear = nn.Linear(256, num_outputs)

        ################################################################
        self.icm_conv1 = nn.Conv2d(num_inputs, 32, 3, stride=2, padding=1)
        self.icm_conv2 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.icm_conv3 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.icm_conv4 = nn.Conv2d(32, 32, 3, stride=2, padding=1)

        #self.icm_lstm = nn.LSTMCell(32 * 3 * 3, 256)

        self.inverse_linear1 = nn.Linear(2304, 256)
        self.inverse_linear2 = nn.Linear(256, num_outputs)

        self.forward_linear1 = nn.Linear(1152 + num_outputs, 256)
        self.forward_linear2 = nn.Linear(256, 1152)


    def forward(self, inputs, icm):

        if icm == False:
            """A3C"""
            inputs, a3c_hx = inputs

            x = F.elu(self.conv1(inputs))
            x = F.elu(self.conv2(x))
            x = F.elu(self.conv3(x))
            x = F.elu(self.conv4(x))

            #x = x.view(-1, 32 * 3 * 3)
            a3c_hx = self.gru(x.view(-1, 1152), (a3c_hx))
            x = a3c_hx

            critic = self.critic_linear(x)
            actor = self.actor_linear(x)
            return critic, actor, a3c_hx

        else:
            """icm"""
            s_t, s_t1, a_t = inputs
            '''
            s_t, (icm_hx, icm_cx) = s_t
            s_t1, (icm_hx1, icm_cx1) = s_t1
            '''
            vec_st = F.elu(self.icm_conv1(s_t))
            vec_st = F.elu(self.icm_conv2(vec_st))
            vec_st = F.elu(self.icm_conv3(vec_st))
            vec_st = F.elu(self.icm_conv4(vec_st))

            vec_st1 = F.elu(self.icm_conv1(s_t1))
            vec_st1 = F.elu(self.icm_conv2(vec_st1))
            vec_st1 = F.elu(self.icm_conv3(vec_st1))
            vec_st1 = F.elu(self.icm_conv4(vec_st1))

            vec_st = vec_st.view(-1, 1152)
            vec_st1 = vec_st1.view(-1, 1152)

            inverse_vec = torch.cat((vec_st, vec_st1), 1)
            forward_vec = torch.cat((vec_st, a_t), 1)

            inverse = self.inverse_linear1(inverse_vec)
            inverse = F.relu(inverse)
            inverse = self.inverse_linear2(inverse)
            inverse = F.softmax(inverse, dim=0)####

            forward = self.forward_linear1(forward_vec)
            forward = F.relu(forward)
            forward = self.forward_linear2(forward)

            return vec_st1, inverse, forward
            #return vec_st1, inverse, forward, (icm_hx, icm_cx), (icm_hx1, icm_cx1)
        
    def try_load(self, save_dir):
        paths = glob.glob(save_dir + '*.tar') ; step = 0
        if len(paths) > 0:
            ckpts = [int(s.split('.')[-2]) for s in paths]
            ix = np.argmax(ckpts) ; step = ckpts[ix]
            self.load_state_dict(torch.load(paths[ix]))
        print("\tno saved models") if step is 0 else print("\tloaded model: {}".format(paths[ix]))
        return step

