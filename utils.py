import numpy as np
from crowd_sim.envs import CrowdSim
#import .scenarios as scenarios
import gym_vecenv


def normalize_obs(obs, mean, std):
    if mean is not None:
        return np.divide((obs - mean), std)
    else:
        return obs


def make_env(seed,rank):
    def _thunk():
        env = make_multiagent_env()
        env.seed(seed + rank) # seed not implemented
        return env
    return _thunk


def make_multiagent_env():
    #scenario = scenarios.load(env_id+".py").Scenario(num_agents=num_agents, dist_threshold=dist_threshold,
    #                                                 arena_size=arena_size, identity_size=identity_size)
    #world = scenario.make_world()

    env = CrowdSim()
    return env


def make_parallel_envs(args):
    # make parallel environments
    envs = [make_env(args.seed, i) for i in range(args.num_processes)]
    if args.num_processes > 1:
        envs = gym_vecenv.SubprocVecEnv(envs)
    else:
        envs = gym_vecenv.DummyVecEnv(envs)

    envs = gym_vecenv.MultiAgentVecNormalize(envs, ob=False, ret=True)
    return envs


def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module
