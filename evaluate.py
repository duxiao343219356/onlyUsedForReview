import numpy as np
import torch
from arguments import get_args
from utils import normalize_obs
from learner import setup_master
import time
import math

def evaluate(args, seed, policies_list, ob_rms=None, render=False, env=None, master=None, render_attn=True):
    """
    RL evaluation: supports eval through training code as well as independently
    policies_list should be a list of policies of all the agents;
    len(policies_list) = num agents
    """
    if env is None or master is None: # if any one of them is None, generate both of them
        master, env = setup_master(args, return_env=True)

    if seed is None: # ensure env eval seed is different from training seed
        seed = np.random.randint(0,100000)
    print("Evaluation Seed: ",seed)
    env.seed(seed)

    if ob_rms is not None:
        obs_mean, obs_std = ob_rms
    else:
        obs_mean = None
        obs_std = None
    master.load_models(policies_list)
    master.set_eval_mode()

    num_eval_episodes = args.num_eval_episodes
    all_episode_rewards = np.full((num_eval_episodes, env.n), 0.0)
    per_step_rewards = np.full((num_eval_episodes, env.n), 0.0)

    # TODO: provide support for recurrent policies and mask
    recurrent_hidden_states = None
    mask = None

    # world.dists at the end of episode for simple_spread
    final_min_dists = []
    num_success = 0
    episode_length = 0
    EX_extra_time = 0
    extra_time = []
    length_time = []

    human_num_success = 0
    human_eisode_length = 0
    human_EX_extra_time = 0
    human_extra_time = []
    t1 = 0
    t2 = 0
    for t in range(num_eval_episodes):
        obs = env.reset()
        obs = normalize_obs(obs, obs_mean, obs_std)
        done = [False]*env.n
        episode_rewards = np.full(env.n, 0.0)
        episode_steps = 0
        # if render:
            # attn = None if not render_attn else master.team_attn
            # if attn is not None and len(attn.shape)==3:
            #     attn = attn.max(0)
            # env.render()
            
        while not np.all(done):
            actions = []
            with torch.no_grad():
                actions = master.eval_act(obs, recurrent_hidden_states, mask)
            actions = actions.reshape(env.n,1)
            episode_steps += 1
            #print('eval_actiontype:{},eval_actions:{}'.format(type(actions)),actions)
            obs, reward, done, info = env.step(actions)
            # print('obs:{}'.format(obs))
            obs = normalize_obs(obs, obs_mean, obs_std)
            episode_rewards += np.array(reward)
            # if render:
            #     attn = None if not render_attn else master.team_attn
            #     if attn is not None and len(attn.shape)==3:
            #         attn = attn.max(0)
            #     env.render(attn=attn)
            #     if args.record_video:
            #         time.sleep(0.08)
        if render and t==10:
          print('render')
          # env.render()
        # import pdb
        # pdb.set_trace()
        ####evaluate human
        # print("t1:{}{}".format(t1,t2))
        human_num_success += info['n'][0]['human_is_success']
        # print("human_is_success:{}".format(info['n'][0]['human_is_success']))
        if info['n'][0]['human_is_success'] == True and info['n'][0]['human_steps']/10 - info['n'][0]['init_human_time']>0:
          human_EX_extra_time = (human_EX_extra_time*t1 + info['n'][0]['human_steps']/10 - info['n'][0]['init_human_time'])/(t1+1)
          human_extra_time.append(info['n'][0]['human_steps']/10 - info['n'][0]['init_human_time']) 
          # print("human_steps:{}, init_human:{}".format(info['n'][0]['human_steps']/10,info['n'][0]['init_human_time']))

        per_step_rewards[t] = episode_rewards/episode_steps
        num_success += (info['n'][0]['is_success']==True)  
        # num_success += info['n'][0]['is_success']
        episode_length = (episode_length*t + info['n'][0]['steps'])/(t+1)
        if info['n'][0]['is_success'] == True and info['n'][0]['steps']/10 - info['n'][0]['init_max_time'] >0:
          EX_extra_time = (EX_extra_time*t2 + info['n'][0]['steps']/10 - info['n'][0]['init_max_time'])/(t2+1)
          #pow_time = (pow_time*t + math.pow(info['n'][0]['steps']/10 - info['n'][0]['init_max_time'], 2))/(t+1)
          extra_time.append(info['n'][0]['steps']/10 - info['n'][0]['init_max_time'])
        # length_time.append(info['n'][0]['init_max_time'])
        t1 = t1+1
        t2 = t2+1 
        # for simple spread env only
        if args.env_name == 'simple_spread':
            final_min_dists.append(env.min_dists)
        # if render:
        #     print("Ep {} | Success: {} \n Av per-step reward: {:.2f} | Ep Length {}".format(t,info['n'][0]['is_success'],
        #         per_step_rewards[t][0],info['n'][0]['steps']))
        all_episode_rewards[t, :] = episode_rewards # all_episode_rewards shape: num_eval_episodes x num agents

        if args.record_video:
            # print(attn)
            input('Press enter to continue: ')

    #human
    # print("ex:{},extra_time{}".format(human_EX_extra_time, human_extra_time))
    pow_time_human = 0
    for time in human_extra_time:
      pow_time_human = pow_time_human + math.pow(time - human_EX_extra_time, 2)
    huam_Std_extra_time = 0
    if len(human_extra_time) != 0:
      huam_Std_extra_time = math.sqrt(pow_time_human/len(human_extra_time))
    

    #rotbot
    pow_time = 0
    for time in extra_time:
      pow_time = pow_time + math.pow(time - EX_extra_time, 2)
    Std_extra_time = 0
    if len(extra_time) != 0:
      Std_extra_time = math.sqrt(pow_time/len(extra_time))
    info_time = []
    info_time.append(episode_length)
    info_time.append(EX_extra_time)
    info_time.append(Std_extra_time)
    info_time.append(human_num_success)
    info_time.append(human_EX_extra_time)
    info_time.append(huam_Std_extra_time)
    return all_episode_rewards, per_step_rewards, final_min_dists, num_success, info_time


if __name__ == '__main__':
    args = get_args()
    checkpoint = torch.load(args.load_dir, map_location=lambda storage, loc: storage)
    policies_list = checkpoint['models']
    ob_rms = checkpoint['ob_rms']
    all_episode_rewards, per_step_rewards, final_min_dists, num_success, episode_length = evaluate(args, args.seed, 
                    policies_list, ob_rms, args.render, render_attn=args.masking)
    print("Average Per Step Reward {}\nNum Success {}/{} | Av. Episode Length {:.2f} | EX_extra_time {:.2f} | Std_extra_time {:.2f})"
            .format(per_step_rewards.mean(0),num_success,args.num_eval_episodes,episode_length[0], episode_length[1], episode_length[2]))
    if final_min_dists:
        print("Final Min Dists {}".format(np.stack(final_min_dists).mean(0)))
