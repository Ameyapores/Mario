from model import build_network, build_icm_model, get_reward_intrinsic
import numpy as np
from collections import deque
from setup_env import setup_env
import argparse
from keras.optimizers import RMSprop
import keras.backend as K

save_freq = 250000
learning_rate = 0.001
batch_size = 20
checkpoint = 0
steps = 1000000
swap_freq= 100

def policy_loss(advantage=0., beta=0.01):
    def loss(y_true, y_pred):
        return -K.sum(K.log(K.sum(y_true * y_pred, axis=-1) + K.epsilon()) * K.flatten(advantage)) + \
               beta * K.sum(y_pred * K.log(y_pred + K.epsilon()))
    return loss

def value_loss():
    def loss(y_true, y_pred):
        return 0.5 * K.sum(K.square(y_true - y_pred))
    return loss
beta=0.01
class LearningAgent(object):
    def __init__(self, action_space, batch_size=32, swap_freq=200):
        env= setup_env('SuperMarioBros-v0')
        _, _, self.train_net, advantage = build_network((env.observation_space.shape[-1], *env.observation_space.shape[:2]), (env.action_space.n))
        self.icm = build_icm_model((env.observation_space.shape[:2]), (env.action_space.n,))
        self.train_net.compile(optimizer=RMSprop(epsilon=0.1, rho=0.99),loss=[value_loss(), policy_loss(advantage, beta)])
        self.icm.compile(optimizer="rmsprop", loss=lambda y_true, y_pred: y_pred)

        self.pol_loss = deque(maxlen=25)
        self.val_loss = deque(maxlen=25)
        self.values = deque(maxlen=25)
        self.swap_freq = swap_freq
        self.swap_counter = self.swap_freq
        self.batch_size = batch_size
        self.unroll = np.arange(self.batch_size)
        self.targets = np.zeros((self.batch_size, env.action_space.n))
        self.counter = 0

    def store_results(self, loss, values, loss_icm):
        self.pol_loss.append(loss[2])
        self.val_loss.append(loss[1])
        self.values.append(np.mean(values))
        min_val, max_val, avg_val = min(self.values), max(self.values), np.mean(self.values)

    def learn(self, last_observations, actions, rewards, learning_rate=0.001):
        K.set_value(self.train_net.optimizer.lr, learning_rate)
        frames = len(last_observations)
        self.counter += frames
        values, policy = self.train_net.predict([last_observations, self.unroll])    

        self.targets.fill(0.)
        advantage = rewards - values.flatten()
        self.targets[self.unroll, :] = actions.astype(np.float32)    
        loss = self.train_net.train_on_batch([last_observations, advantage],
                                             [rewards, self.targets])
        loss_icm = self.icm.train_on_batch([last_observations[:, -2, ...],
                                            last_observations[:, -1, ...],
                                            actions,
                                            rewards.reshape((-1, 1))],
                                            np.zeros((self.batch_size,)))
        self.store_results(loss, values, loss_icm)
        self.swap_counter -= frames
        if self.swap_counter < 0:
            self.swap_counter += self.swap_freq
            return True
        return False

def train():
    env=setup_env('SuperMarioBros-v0')
    agent = LearningAgent(env.action_space, batch_size, swap_freq)
    if checkpoint > 0:
        agent.train_net.load_weights('model-%s-%d.h5' % ('SuperMarioBros-v0', checkpoint,))

    last_obs = np.zeros((batch_size,) + (env.observation_space.shape[-1], *env.observation_space.shape[:2]))
    actions = np.zeros((batch_size, env.action_space.n), dtype=np.int32)
    rewards = np.zeros(batch_size)

    idx = 0
    agent.counter = checkpoint
    save_counter = checkpoint % save_freq + save_freq
    while True:
        last_obs[idx, ...], actions[idx, ...], rewards[idx] = mem_queue.get()
        idx = (idx + 1) % batch_size
        if idx == 0:
            lr = max(1.0e-8, (steps - agent.counter) / steps * learning_rate)
            updated = agent.learn(last_obs, actions, rewards, learning_rate)

        save_counter -= 1
        if save_counter < 0:
            save_counter += save_freq
            agent.train_net.save_weights('model-%s-%d.h5' % ('SuperMarioBros-v0', agent.counter,), overwrite=True)
            agent.icm.save_weights('icm_model-%s-%d.h5' % ('SuperMarioBros-v0', agent.counter,), overwrite=True)