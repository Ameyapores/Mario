from setup_env import setup_env
import numpy as np
from collections import deque
import argparse
from model import build_network, build_icm_model, get_reward_intrinsic
import h5py

from multiprocessing import *
import keras.backend as K
from keras.optimizers import RMSprop

parser = argparse.ArgumentParser(description='Training model')
parser.add_argument('--with_reward', dest='with_reward', action='store_true')
parser.add_argument('--processes', default=4, help='Number of processes that generate experience for agent',
                    dest='processes', type=int)
parser.add_argument('--queue_size', default=256, help='Size of queue holding agent experience', dest='queue_size',
                    type=int)
parser.add_argument('--batch_size', default=20, help='Batch size to use during training', dest='batch_size', type=int)
args = parser.parse_args()

past_range=3

save_freq = 250000
learning_rate = 0.001
batch_size = 20
checkpoint = 0
steps = 1000000
swap_freq= 100

class ActingAgent(object):
    def __init__(self, num_action, n_step=8, discount=0.99):
        env= setup_env('SuperMarioBros-v0')
        self.value_net, self.policy_net, self.load_net, _ = build_network(env.observation_space.shape, num_action)
        self.icm = build_icm_model((env.observation_space.shape[:2]), (num_action,))

        self.value_net.compile(optimizer='rmsprop', loss='mse')
        self.policy_net.compile(optimizer='rmsprop', loss='mse')
        self.load_net.compile(optimizer='rmsprop', loss='mse', loss_weights=[0.5, 1.])  # dummy loss
        self.icm.compile(optimizer="rmsprop", loss=lambda y_true, y_pred: y_pred)

        self.num_action = num_action
        self.observations = np.zeros(env.observation_space.shape)
        self.last_observations = np.zeros_like(self.observations)

        self.n_step_data = deque(maxlen=n_step)
        self.n_step = n_step
        self.discount = discount
        #self.queue= ReplayQueue(Replay_memory_size)

    def save_observation(self, observation):
        self.last_observations = self.observations[...]
        self.observations = np.roll(self.observations, -1, axis=0)

    def init_episode(self, observation):
        for _ in range(past_range):
            self.save_observation(observation)

    def reset(self):
        self.n_step_data.clear()

    def sars_data(self, action, reward, observation, terminal, mem_queue):
        self.save_observation(observation)
        reward = np.clip(reward, -1., 1.)

        self.n_step_data.appendleft([self.last_observations, action, reward])

        if terminal or len(self.n_step_data) >= self.n_step:
            r = 0.
            if not terminal:
                r = self.value_net.predict(self.observations[None])[0]
            for i in range(len(self.n_step_data)):
                r = self.n_step_data[i][2] + self.discount * r
                mem_queue.put(self.n_step_data[i][0], self.n_step_data[i][1], r)
            self.reset()
        #return (self.queue)

    def choose_action(self, observation=None, eps=0.1):
        if np.random.rand(1) < eps:
            action = np.random.rand(self.num_action)
        elif observation is None:
            action = self.policy_net.predict(self.observations[None])[0]
        else:
            action = self.policy_net.predict([observation])[0]
        action = np.round(action)
        return action.astype(np.int32), np.argmax(action)

def generate_exp(mem_queue, weight_dict, no):

    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    pid = os.getpid()
    os.environ['THEANO_FLAGS'] = 'floatX=float32,device=gpu,nvcc.fastmath=True,lib.cnmem=0,' + \
                                 'compiledir=th_comp_act_' + str(no)
    print(' %5d> Process started' % (pid,))
    env=setup_env('SuperMarioBros-v0')
    agent=ActingAgent(env.action_space.n)
    frames=0
    if frames > 0:
        print(' %5d> Loaded weights from file' % (pid,))
        agent.load_net.load_weights('model-%s-%d.h5' % (args.game.split("/")[-1], frames))
        agent.icm.load_weights('icm_model-%s-%d.h5' % (args.game.split("/")[-1], frames))
    else:
        import time
        while 'weights' not in weight_dict:
            time.sleep(0.1)
        agent.load_net.set_weights(weight_dict['weights'])
        agent.icm.set_weights(weight_dict['weights_icm'])
        print(' %5d> Loaded weights from dict' % (pid,))
    
    batch_size = args.batch_size
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
            r_in = get_reward_intrinsic(agent.icm, [(obs_last[None]),(observation[None]), np.array([action_list])])

            if args.with_reward:
                total_reward = reward + eta * r_in[0]
            else:
                total_reward = eta * r_in[0]
            episode_reward += total_reward
            best_score = max(best_score, episode_reward)
            agent.sars_data(action, total_reward, observation, done, mem_queue)
            op_last = action
            obs_last = observation
            if frames % 2000 == 0:
                print(' %5d> Best: %4d; Avg: %6.2f; Max: %4d' % (
                        pid, best_score, np.mean(avg_score), np.max(avg_score)))

            if frames % batch_size == 0:
                update = weight_dict.get('update', 0)
                if update > last_update:
                    last_update = update
                    # print(' %5d> Getting weights from dict' % (pid,))
                    agent.load_net.set_weights(weight_dict['weights'])
                    agent.icm.set_weights(weight_dict['weights_icm'])           
            
        avg_score.append(episode_reward)
    env.close()
    #print(agent.sars_data) 
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

def train(mem_queue, weight_dict):
    import os
    pid = os.getpid()
    os.environ['THEANO_FLAGS'] = 'floatX=float32,device=gpu,nvcc.fastmath=False,lib.cnmem=0.3,' + \
                                 'compiledir=th_comp_learn,optimizer=fast_compile'
    print(' %5d> Learning process' % (pid,))

    env=setup_env('SuperMarioBros-v0')
    agent = LearningAgent(env.action_space, batch_size, swap_freq)
    if checkpoint > 0:
        print(' %5d> Loading weights from file' % (pid,))
        agent.train_net.load_weights('model-%s-%d.h5' % ('SuperMarioBros-v0', checkpoint,))
    
    print(' %5d> Setting weights in dict' % (pid,))
    weight_dict['update'] = 0
    weight_dict['weights'] = agent.train_net.get_weights()
    weight_dict['weights_icm'] = agent.icm.get_weights()

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
            if updated:
                # print(' %5d> Updating weights in dict' % (pid,))
                weight_dict['weights'] = agent.train_net.get_weights()
                weight_dict['weights_icm'] = agent.icm.get_weights()
                weight_dict['update'] += 1
        
        save_counter -= 1
        if save_counter < 0:
            save_counter += save_freq
            agent.train_net.save_weights('model-%s-%d.h5' % ('SuperMarioBros-v0', agent.counter,), overwrite=True)
            agent.icm.save_weights('icm_model-%s-%d.h5' % ('SuperMarioBros-v0', agent.counter,), overwrite=True)
    

#x= generate_exp()

def init_worker():
    import signal
    signal.signal(signal.SIGINT, signal.SIG_IGN)

def main():
    manager = Manager()
    weight_dict = manager.dict()
    mem_queue = manager.Queue(args.queue_size)

    pool = Pool(args.processes + 1, init_worker)

    try:
        for i in range(args.processes):
            pool.apply_async(generate_exp, (mem_queue, weight_dict, i))

        pool.apply_async(train, (mem_queue, weight_dict))

        pool.close()
        pool.join()

    except KeyboardInterrupt:
        pool.terminate()
        pool.join()

if __name__ == "__main__":
    main()
          
