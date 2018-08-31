
import gym
from nes_py.wrappers import BinarySpaceToDiscreteSpaceEnv, wrap as nes_py_wrap
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
import gym_super_mario_bros

def setup_env(env_id: str) -> gym.Env:
    env = gym_super_mario_bros.make(env_id)
    env = BinarySpaceToDiscreteSpaceEnv(env, SIMPLE_MOVEMENT)
    print(env.observation_space, env.action_space)
    env = nes_py_wrap(env)
    print((env.observation_space.shape))
    return env
    

# explicitly define the outward facing API of this module
__all__ = [setup_env.__name__]

x=setup_env('SuperMarioBros-v0')
