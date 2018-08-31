from setup_env import setup_env
import numpy as np
from collections import deque
import argparse
from model import build_network, build_icm_model, get_reward_intrinsic
import h5py
from replay_queue import ReplayQueue

parser = argparse.ArgumentParser(description='Training model')
parser.add_argument('--with_reward', dest='with_reward', action='store_true')
parser.add_argument('--queue_size', default=25600, help='Size of queue holding agent experience', dest='queue_size',
                    type=int)
args = parser.parse_args()

past_range=3
class ActingAgent(object):
    def __init__(self, num_action, mem_queue, n_step=8, discount=0.99):
        self.value_net, self.policy_net, self.load_net, _ = build_network((past_range, *env.observation_space.shape[:2]), num_action)
        self.icm = build_icm_model((env.observation_space.shape[:2]), (num_action,))

        self.value_net.compile(optimizer='rmsprop', loss='mse')
        self.policy_net.compile(optimizer='rmsprop', loss='mse')
        self.load_net.compile(optimizer='rmsprop', loss='mse', loss_weights=[0.5, 1.])  # dummy loss
        self.icm.compile(optimizer="rmsprop", loss=lambda y_true, y_pred: y_pred)

        self.num_action = num_action
        self.observations = np.zeros((past_range, *env.observation_space.shape[:2]))
        self.last_observations = np.zeros_like(self.observations)

        self.n_step_data = deque(maxlen=n_step)
        self.n_step = n_step
        self.discount = discount
        self.mem_queue=ReplayQueue(args.queue_size)

    def save_observation(self, observation):
        self.last_observations = self.observations[...]
        self.observations = np.roll(self.observations, -1, axis=0)
        #self.observations[-input_depth:, ...] = transform_screen(observation)

    def init_episode(self, observation):
        for _ in range(env.observation_space.shape[-1]):
            self.save_observation(observation)

    def reset(self):
        self.n_step_data.clear()

    def sars_data(self, action, reward, observation, terminal, mem_queue):
        self.save_observation(observation)
        reward = np.clip(reward, -1., 1.)
        # reward /= args.reward_scale

        self.n_step_data.appendleft([self.last_observations,
                                     action, reward])

        if terminal or len(self.n_step_data) >= self.n_step:
            r = 0.
            if not terminal:
                r = self.value_net.predict(self.observations[None, ...])[0]
            for i in range(len(self.n_step_data)):
                r = self.n_step_data[i][2] + self.discount * r
                mem_queue.push((self.n_step_data[i][0], self.n_step_data[i][1], r))
            self.reset()

    def choose_action(self, observation=None, eps=0.1):
        if np.random.rand(1) < eps:
            action = np.random.rand(self.num_action)
        elif observation is None:
            action = self.policy_net.predict(self.observations[None, ...])[0]
        else:
            action = self.policy_net.predict([observation])[0]
        action = np.round(action)
        return action.astype(np.int32), np.argmax(action)

import os
output_dir: str =None

output_dir = '{}/train'.format(output_dir)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

print('writing results to {}'.format(repr(output_dir)))
weights_file = '{}/weights.h5'.format(output_dir)
env=setup_env('SuperMarioBros-v0')

agent=ActingAgent(7, 5)

frames=0
best_score = 0
avg_score = deque([0], maxlen=25)
last_update = 0
eta = 1.0 

while True:
    done =False
    episode_reward = 0
    op_last = np.zeros(env.action_space.n)
    observation = env.reset()
    obs_last = observation
    agent.init_episode(observation)

    while not done:
        frames += 1
        action_list, action = agent.choose_action(eps = 1.0 / (frames / 10000.0 + 2.0))
        observation, reward, done, _ = env.step(action)
        env.render(mode='human')
        r_in = get_reward_intrinsic(agent.icm, [(obs_last[:83,:83]), (observation[:83,:83]), action_list])

        if args.with_reward:
            total_reward = reward + eta * r_in[0]
        else:
            total_reward = eta * r_in[0]
        episode_reward += total_reward
        best_score = max(best_score, episode_reward)
        agent.sars_data(action, total_reward, observation, done, mem_queue=25000)
        op_last = action
        obs_last = observation

env.close()
    



