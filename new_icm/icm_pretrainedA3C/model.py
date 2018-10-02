import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def normalized_columns_initializer(weights, std=1.0):
    out = torch.randn(weights.size())
    out *= std / torch.sqrt(out.pow(2).sum(1, keepdim=True))
    return out

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = np.prod(weight_shape[1:4])
        fan_out = np.prod(weight_shape[2:4]) * weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = weight_shape[1]
        fan_out = weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)


class ActorCritic(torch.nn.Module):
    def __init__(self, num_inputs, num_actions):
        super(ActorCritic, self).__init__()
        self.conv1 = nn.Conv2d(num_inputs, 32, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(32, 32, 3, stride=2, padding=1)

        self.lstm = nn.LSTMCell(32 * 6 * 6, 512)

        self.critic_linear = nn.Linear(512, 1)
        self.actor_linear = nn.Linear(512, num_actions)

        self.apply(weights_init)
        self.actor_linear.weight.data = normalized_columns_initializer(
            self.actor_linear.weight.data, 0.01)
        self.actor_linear.bias.data.fill_(0)
        self.critic_linear.weight.data = normalized_columns_initializer(
            self.critic_linear.weight.data, 1.0)
        self.critic_linear.bias.data.fill_(0)

        self.lstm.bias_ih.data.fill_(0)
        self.lstm.bias_hh.data.fill_(0)

        self.train()

    def forward(self, inputs):
        inputs, (hx, cx) = inputs
        x = F.elu(self.conv1(inputs))
        x = F.elu(self.conv2(x))
        x = F.elu(self.conv3(x))
        x = F.elu(self.conv4(x))

        x = x.view(-1, 32 * 6 * 6)
        hx, cx = self.lstm(x, (hx, cx))
        x = hx

        return self.critic_linear(x), self.actor_linear(x), (hx, cx)

class ICM(torch.nn.Module):

    def __init__(self, num_inputs, num_actions):
        super(ICM, self).__init__()
        self.icm_conv1 = nn.Conv2d(num_inputs, 32, 3, stride=2, padding=1)
        self.icm_conv2 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.icm_conv3 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.icm_conv4 = nn.Conv2d(32, 32, 3, stride=2, padding=1)

        self.inverse_linear1 = nn.Linear(2304, 512)
        self.inverse_linear2 = nn.Linear(512, num_actions)

        self.forward_linear1 = nn.Linear(1152 + num_actions, 512)
        self.forward_linear2 = nn.Linear(512, 1152)

        self.apply(weights_init)
        self.inverse_linear1.weight.data = normalized_columns_initializer(
            self.inverse_linear1.weight.data, 0.01)
        self.inverse_linear1.bias.data.fill_(0)
        self.inverse_linear2.weight.data = normalized_columns_initializer(
            self.inverse_linear2.weight.data, 1.0)
        self.inverse_linear2.bias.data.fill_(0)
        
        self.forward_linear1.weight.data = normalized_columns_initializer(
            self.forward_linear1.weight.data, 0.01)
        self.forward_linear1.bias.data.fill_(0)
        self.forward_linear2.weight.data = normalized_columns_initializer(
            self.forward_linear2.weight.data, 1.0)
        self.forward_linear2.bias.data.fill_(0)
        
        self.train

    def forward(self, inputs, icm):
        s_t, s_t1, a_t = inputs
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
        inverse = F.softmax(inverse, dim=-1)####

        forward = self.forward_linear1(forward_vec)
        forward = F.relu(forward)
        forward = self.forward_linear2(forward)

        return vec_st1, inverse, forward