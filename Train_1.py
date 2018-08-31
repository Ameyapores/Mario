import gym
import gym_super_mario_bros
from model import build_network, build_icm_model, get_reward_intrinsic
import numpy as np
from collections import deque
from setup_env import setup_env
import argparse
from keras.optimizers import RMSprop
import keras.backend as K

parser = argparse.ArgumentParser(description='Training model')
parser.add_argument('--game', default='SuperMarioBros-v0', help='OpenAI gym environment name', dest='game', type=str)
parser.add_argument('--lr', default=0.001, help='Learning rate', dest='learning_rate', type=float)
parser.add_argument('--steps', default=1000000, help='Number of frames to decay learning rate', dest='steps', type=int)
parser.add_argument('--batch_size', default=20, help='Batch size to use during training', dest='batch_size', type=int)
parser.add_argument('--swap_freq', default=100, help='Number of frames before swapping network weights',
                    dest='swap_freq', type=int)
parser.add_argument('--checkpoint', default=0, help='Frame to resume training', dest='checkpoint', type=int)
parser.add_argument('--save_freq', default=250000, help='Number of frames before saving weights', dest='save_freq',
                    type=int)
parser.add_argument('--queue_size', default=256, help='Size of queue holding agent experience', dest='queue_size',
                    type=int)
parser.add_argument('--n_step', default=5, help='Number of steps', dest='n_step', type=int)
parser.add_argument('--reward_scale', default=1., dest='reward_scale', type=float)
parser.add_argument('--beta', default=0.01, dest='beta', type=float)
parser.add_argument('--with_reward', dest='with_reward', action='store_true')
args = parser.parse_args()

class LearningAgent(object):
    def __init__(self, env: gym.Env, batch_size=32, swap_freq=200):
        
        super().__init__(env, render_mode)
        self.train_net, advantage = build_network((env.observation_space.shape[-1], *env.observation_space.shape[:2]), (env.action_space.n))
        self.icm = build_icm_model((env.observation_space.shape[:2]), (env.action_space.n,))

        self.train_net.compile(optimizer=RMSprop(epsilon=0.1, rho=0.99),loss=[value_loss(), policy_loss(advantage, args.beta)])
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
        print('\rFrames: %8d; Policy-Loss: %10.6f; Avg: %10.6f '
              '--- Value-Loss: %10.6f; Avg: %10.6f '
              '--- V-value; Min: %6.3f; Max: %6.3f; Avg: %6.3f'
              '--- ICM-Loss: %f' % (
                  self.counter,
                  loss[2], np.mean(self.pol_loss),
                  loss[1], np.mean(self.val_loss),
                  min_val, max_val, avg_val, loss_icm), end='')
        
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
      
def learn_proc(mem_queue):
    save_freq = args.save_freq
    learning_rate = args.learning_rate
    batch_size = args.batch_size
    checkpoint = args.checkpoint
    steps = args.steps

    env = setup_env(args.game)
    agent = LearningAgent(env.action_space, batch_size=args.batch_size, swap_freq=args.swap_freq)
    if checkpoint > 0:
        agent.train_net.load_weights('model-%s-%d.h5' % (args.game, checkpoint,))

    weight_dict['update'] = 0
    weight_dict['weights'] = agent.train_net.get_weights()
    weight_dict['weights_icm'] = agent.icm.get_weights()

    last_obs = np.zeros((batch_size,) + env.observation)
    actions = np.zeros((batch_size, env.action_space.num_discrete_space), dtype=np.int32)
    rewards = np.zeros(batch_size)

    idx = 0
    agent.counter = checkpoint
    save_counter = checkpoint % save_freq + save_freq
    while True:
        last_obs[idx, ...], actions[idx, ...], rewards[idx] = mem_queue.get()
        idx = (idx + 1) % batch_size
        if idx == 0:
            lr = max(1.0e-8, (steps - agent.counter) / steps * learning_rate)
            updated = agent.learn(last_obs, actions, rewards, learning_rate=lr)
            if updated:
                weight_dict['weights'] = agent.train_net.get_weights()
                weight_dict['weights_icm'] = agent.icm.get_weights()
                weight_dict['update'] += 1
        save_counter -= 1
        if save_counter < 0:
            save_counter += save_freq
            agent.train_net.save_weights('model-%s-%d.h5' % (args.game.split("/")[-1], agent.counter,), overwrite=True)
            agent.icm.save_weights('icm_model-%s-%d.h5' % (args.game.split("/")[-1], agent.counter,), overwrite=True)
                
def policy_loss(advantage=0., beta=0.01):
    def loss(y_true, y_pred):
        return -K.sum(K.log(K.sum(y_true * y_pred, axis=-1) + K.epsilon()) * K.flatten(advantage)) + \
               beta * K.sum(y_pred * K.log(y_pred + K.epsilon()))
    return loss

def value_loss():
    def loss(y_true, y_pred):
        return 0.5 * K.sum(K.square(y_true - y_pred))
    return loss

class ActingAgent(object):
    def __init__(self, num_action, n_step=8, discount=0.99):
        super().__init__(env, render_mode)
        self.value_net, self.policy_net, self.load_net, _ = build_network(env.observation_space,
                                                                          num_action)
        self.icm = build_icm_model(env.observation_space, (num_action,))

        self.value_net.compile(optimizer='rmsprop', loss='mse')
        self.policy_net.compile(optimizer='rmsprop', loss='mse')
        self.load_net.compile(optimizer='rmsprop', loss='mse', loss_weights=[0.5, 1.])  # dummy loss
        self.icm.compile(optimizer="rmsprop", loss=lambda y_true, y_pred: y_pred)

        self.num_action = num_action
        self.observations = np.zeros(env.observation_space)
        self.last_observations = np.zeros_like(self.observations)

        self.n_step_data = deque(maxlen=n_step)
        self.n_step = n_step
        self.discount = discount

    def init_episode(self, observation):
        for _ in range(past_range):
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
                mem_queue.put((self.n_step_data[i][0], self.n_step_data[i][1], r))
            self.reset()

    def choose_action(self, observation=None, eps=0.1):
        if np.random.rand(1) < eps:
            action = np.random.rand(self.num_action)
        elif observation is None:
            action = self.policy_net.predict(self.observations[None, ...])[0]
        else:
            action = self.policy_net.predict([observation])[0]
        action = np.round(action)
        return action.astype(np.int32)

    def save_observation(self, observation):
        self.last_observations = self.observations[...]
        self.observations = np.roll(self.observations, -input_depth, axis=0)
        #self.observations[-input_depth:, ...] = transform_screen(observation)

def generate_experience_proc(mem_queue, weight_dict):
    
    frames = 0
    batch_size = args.batch_size
    env = setup_env(args.game)
    agent = ActingAgent(env.action_space.num_discrete_space, n_step=args.n_step)
    if frames > 0:
        agent.load_net.load_weights('model-%s-%d.h5' % (args.game.split("/")[-1], frames))
        agent.icm.load_weights('icm_model-%s-%d.h5' % (args.game.split("/")[-1], frames))
    else:
        import time
        while 'weights' not in weight_dict:
           time.sleep(0.1)
        agent.load_net.set_weights(weight_dict['weights'])
        agent.icm.set_weights(weight_dict['weights_icm'])

    best_score = 0
    avg_score = deque([0], maxlen=25)
    last_update = 0
    eta = 1.0 # parameter for intrinsic reward
    while True:
        done = False
        episode_reward = 0
        op_last = np.zeros(env.action_space.n)
        observation = env.reset()
        obs_last = observation.copy()
        agent.init_episode(observation)

        while not done:
            frames += 1
            action = agent.choose_action(eps = 1.0 / (frames / 10000.0 + 2.0))
            observation, reward, done, _ = env.step(action)
            env.render()
            #if self.render_mode is not None:
            #    self.env.render(mode=self.render_mode)
            if args.with_reward:
                total_reward = reward + eta * r_in[0]
            else:
                total_reward = eta * r_in[0]
            episode_reward += total_reward
            best_score = max(best_score, episode_reward)
            agent.sars_data(action, total_reward, observation, done, mem_queue)
            op_last = action
            obs_last = observation.copy()
            if frames % 2000 == 0:
                print(' %5d> Best: %4d; Avg: %6.2f; Max: %4d' % (
                      pid, best_score, np.mean(avg_score), np.max(avg_score)))
            if frames % batch_size == 0:
                update = weight_dict.get('update', 0)
                if update > last_update:
                    last_update = update
                    agent.load_net.set_weights(weight_dict['weights'])
                    agent.icm.set_weights(weight_dict['weights_icm'])
            avg_score.append(episode_reward)
        env.close()
        
def main():
    mem_queue = args.queue_size
    try:
        learn_proc(mem_queue)
        generate_experience_proc(mem_queue, weight_dict)
        

    except KeyboardInterrupt:
        pool.terminate()
        pool.join()     
        
        
if __name__ == "__main__":
    main()
          
