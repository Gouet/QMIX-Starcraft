import qmix
from smac.env import StarCraft2Env
import numpy as np
import torch

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

class Transform:
    def transform(self, tensor):
        raise NotImplementedError

    def infer_output_info(self, vshape_in, dtype_in):
        raise NotImplementedError

class OneHot(Transform):
    def __init__(self, out_dim):
        self.out_dim = out_dim

    def transform(self, tensor):
        y_onehot = tensor.new(*tensor.shape[:-1], self.out_dim).zero_()
        y_onehot.scatter_(-1, tensor.long(), 1)
        return y_onehot.float()

    def infer_output_info(self, vshape_in, dtype_in):
        return (self.out_dim,), th.float32

class Runner:
    def __init__(self, arglist, scenario, actors):
        self.envs = []
        for _ in range(actors):
            env = StarCraft2Env(map_name=scenario, replay_dir="./replay/")
            self.envs.append(env)
        env_info = self.envs[0].get_env_info()

        self.actors = actors
        self.scenario = scenario

        self.n_actions = env_info["n_actions"]
        self.n_agents = env_info["n_agents"]
        self.state_shape = env_info["state_shape"]
        self.obs_shape = env_info["obs_shape"] + self.n_agents + self.n_actions
        self.episode_limit = env_info['episode_limit']
        self.agent_id_one_hot = OneHot(self.n_agents)
        self.actions_one_hot = OneHot(self.n_actions)

        self.agent_id_one_hot_array = []
        for agent_id in range(self.n_agents):
            self.agent_id_one_hot_array.append(self.agent_id_one_hot.transform(torch.FloatTensor([agent_id])).cpu().detach().numpy())
        self.agent_id_one_hot_array = np.array(self.agent_id_one_hot_array)

        self.qmix_algo = qmix.QMix(arglist.train, self.n_agents, self.obs_shape, self.state_shape, self.n_actions, 0.0005)
        if arglist.train == False:
            self.qmix_algo.load_model('./saved/agents_' + str(arglist.load_episode_saved))
            print('Load model agent ', str(arglist.load_episode_saved))

        self.episode_global_step = 0
        self.episode = 0

        self.state_zeros = np.zeros(self.state_shape)
        self.obs_zeros = np.zeros((self.n_agents, self.obs_shape))
        self.actions_zeros = np.zeros([self.n_agents, 1])
        self.reward_zeros = 0
        self.agents_available_actions_zeros = np.zeros((self.n_agents, self.n_actions))
        pass

    def reset(self):
        self.qmix_algo.on_reset(1)
        self.terminated = []
        self.episode_reward = []
        self.episode_step = []
        self.obs_array = []
        self.state_array = []
        self.replay_buffers = []
        self.win_counted_array = []
        self.episodes = []
        episode_managed = self.episode
        for env in self.envs:
            env.reset()
            self.episodes.append(episode_managed)
            self.win_counted_array.append(False)
            self.terminated.append(False)
            self.episode_reward.append(0)
            self.episode_step.append(0)
            self.replay_buffers.append(qmix.ReplayBuffer(self.episode_limit))

            actions_one_hot_reset = torch.zeros_like(torch.empty(self.n_agents, self.n_actions))
            obs = np.array(env.get_obs())
            self.obs_array.append(np.concatenate([obs, actions_one_hot_reset, self.agent_id_one_hot_array], axis=-1))
            self.state_array.append(np.array(env.get_state()))
            episode_managed += 1
        pass

    def run(self):
        while not all(self.terminated):
            for i, env in enumerate(self.envs):
                if self.terminated[i] == True:
                    continue

                agents_available_actions = []
                for agent_id in range(self.n_agents):
                    agents_available_actions.append(env.get_avail_agent_actions(agent_id))

                actions = self.qmix_algo.act(torch.FloatTensor(self.obs_array[i]).to(device), torch.FloatTensor(agents_available_actions).to(device))

                reward, terminated, _ = env.step(actions)
                self.terminated[i] = terminated

                agents_available_actions2 = []
                for agent_id in range(self.n_agents):
                    agents_available_actions2.append(env.get_avail_agent_actions(agent_id))

                obs2 = np.array(env.get_obs())
                actions_one_hot_agents = []
                for action in actions:
                    actions_one_hot_agents.append(self.actions_one_hot.transform(torch.FloatTensor(action)).cpu().detach().numpy())
                actions_one_hot_agents = np.array(actions_one_hot_agents)

                obs2 = np.concatenate([obs2, actions_one_hot_agents, self.agent_id_one_hot_array], axis=-1)
                state2 = np.array(env.get_state())

                self.replay_buffers[i].add(self.state_array[i], state2, actions, [reward], [self.terminated[i]],
                self.obs_array[i], obs2, agents_available_actions, agents_available_actions2, 0)

                self.qmix_algo.decay_epsilon_greddy(self.episode_global_step)

                self.episode_reward[i] += reward
                self.episode_step[i] += 1

                self.obs_array[i] = obs2
                self.state_array[i] = state2

            self.episode_global_step += 1

        for idx, replay_buffer in enumerate(self.replay_buffers):
            for i in range(self.episode_step[idx], self.episode_limit):
                replay_buffer.add(self.state_zeros, self.state_zeros, self.actions_zeros, [self.reward_zeros],
                [True], self.obs_zeros, self.obs_zeros, self.agents_available_actions_zeros, self.agents_available_actions_zeros, 1)

        self.episode += self.actors

        for idx, env in enumerate(self.envs):
            self.win_counted_array[idx] = env.win_counted

        return self.replay_buffers

    def save(self):
        for env in self.envs:
            env.save_replay()

    def close(self):
        for env in self.envs:
            env.close()