# Vanilla Policy Gradient (REINFORCE) implementation for discrete action spaces
# Github: https://github.com/nkxv/ReinforcementLearning/
# Example usage: python pg.py --verbose --plot

import gymnasium as gym

from torch.distributions.categorical import Categorical

import pandas as pd


import torch.nn as nn
import numpy as np
import torch

import torch.optim as optim
import csv
import matplotlib.pyplot as plt

import torch.distributions as D
import argparse
import time
import os


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default="MyPPO")
    parser.add_argument("--env-id", type=str, default="CartPole-v1")
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--n-epochs", type=int, default=50)
    parser.add_argument("--seed", type=int, default=99)
    parser.add_argument('--n-rollout-steps', type=int, default=2000)
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--verbose", action="store_true", default=False)
    parser.add_argument("--logdir", type=str, default="PPOoutputfolder")
    parser.add_argument("--no-eval", action="store_true", default=False)
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--log-std-init", type=float, default=0)
    return parser.parse_args()




def plot(filepath):

    df = pd.read_csv(filepath)
    
    ax1 = plt.subplot(311)
    plt.scatter(df['Global Step'], df['Episodic Return'])
    plt.tick_params('x', labelbottom=False)
    ax1.set_ylabel("Episodic Return")

    ax2 = plt.subplot(312, sharex=ax1)
    plt.plot(df['Global Step'], df['Immediate Return'])
    plt.tick_params('x')
    ax2.set_ylabel("Immediate Return")
    ax2.set_xlabel("Global Timestep")

    returns = df['Episodic Return'].dropna().values
    window = 50
    rolling = pd.Series(returns).rolling(window).mean()
    ax3 = plt.subplot(313)
    plt.plot(rolling)
    ax3.set_xlabel("Episode")
    ax3.set_ylabel("Average Return (last 50)")



    plt.show()

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class Agent(nn.Module):
    def __init__(self, env):
        super().__init__()
        obs_dim = np.array(env.observation_space.shape).prod()
        hidden = 64
        action_dim = env.action_space.n #Discrete Action space
        self.actor = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden), std=0.01),
            nn.ReLU(),
            layer_init(nn.Linear(hidden, action_dim), std=0.01)
        )


    def get_action(self, obs, action=None):
        logits = self.actor.forward(obs)
        policyprobs = Categorical(logits=logits)
        if action is None:
            action = policyprobs.sample()
        return action, policyprobs.log_prob(action)
        
    def eval(self, env, episodes=1):
        returns = []
        with torch.no_grad():
            for i in range(episodes):
                obs, _ = env.reset()
                done = False
                ep_ret = 0
                while not done:
                    logits = self.actor(torch.tensor(obs, dtype=torch.float32))
                    action = logits.argmax().item()
                    obs, r, terminated, truncated, _ = env.step(action)
                    done = terminated or truncated
                    ep_ret += r
                _, _ = env.reset()
                returns.append(ep_ret)
        return np.mean(returns)

if __name__ == "__main__":
    Config = parse_args()

    run_id = f"seed{Config.seed}_gamma{Config.gamma}_lr{Config.lr}_rollout-steps{Config.n_rollout_steps}_log-std-init{Config.log_std_init}_{int(time.time())}"
    outputfilepath = os.path.join(Config.logdir, f"{Config.exp_name}__{run_id}.csv")
    os.makedirs(Config.logdir, exist_ok=True)

    env = gym.make(Config.env_id)
    
    globalstep = 0
    n_epochs = Config.n_epochs
    n_rollout_steps = Config.n_rollout_steps

    agent = Agent(env)
    optimizer = optim.Adam(agent.parameters(), lr=Config.lr, eps=1e-5)



    with open(outputfilepath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(['Global Step', 'Episodic Return', 'Immediate Return'])

        observation, info = env.reset(seed=Config.seed)
        observation = torch.Tensor(observation)

        eval_steps = []
        eval_returns = []

        for epoch in range(n_epochs):

            #Storage for Trajectories
            observations_list = torch.zeros((n_rollout_steps, env.observation_space.shape[0]))
            next_observations_list = torch.zeros((n_rollout_steps, env.observation_space.shape[0]))
            actions_list = torch.zeros(n_rollout_steps)
            rewards_list = torch.zeros(n_rollout_steps)
            dones_list = torch.zeros(n_rollout_steps)

            episode_reward = 0
        
            #Rollout
            end = 0
            for t in range(n_rollout_steps):
                
                observations_list[t] = observation
                #action from policy
                action, _= agent.get_action(observation)
                
                # step (transition) through the environment with the action
                # receiving the next observation, reward and if the episode has terminated or truncated
                observation, reward, terminated, truncated, info = env.step(action.cpu().numpy())
                observation = torch.Tensor(observation)

                next_observations_list[t] = observation
                
                actions_list[t] = action
                rewards_list[t] = reward

                episode_reward += reward

                # If the episode has ended then we can reset to start a new episode
                if terminated or truncated:
                    dones_list[t] = True
                    writer.writerow([globalstep, episode_reward, reward])
                    #print(episode_reward)
                    observation, info = env.reset()
                    observation = torch.Tensor(observation)
                    episode_reward = 0 
                    end = t
                    #print(f"Episode ended at timestep {t}")
                    
                else:
                    writer.writerow([globalstep, "", reward])
                end = t
                globalstep +=1


            #End of Rollout


            if not Config.no_eval:
                eval_ret = agent.eval(env, episodes=Config.eval_episodes)
                eval_steps.append(globalstep)
                eval_returns.append(eval_ret)
                if Config.verbose:
                    print(f"[Eval] Step {globalstep}: Avg Return = {eval_ret:.1f}")

            gamma = .99 #Discount Factor

            observations_list = observations_list[:end+1] #Removing any empty observations off the end
            next_observations_list = next_observations_list[:end+1]
            actions_list = actions_list[:end+1]
            rewards_list = rewards_list[:end+1]
            dones_list = dones_list[:end+1]
            #print(f"Obs: \t{len(observations_list)}\nNObs:\t{len(next_observations_list)}\nRew:\t{len(rewards_list)}\nAct:\t{len(actions_list)}")

            T=end+1
            obs_batch = observations_list[:T]                                # [T, obs_dim]
            act_batch = actions_list[:T].long()                              # [T]
            rew_batch = rewards_list[:T]                                     # [T]
            dones_batch = dones_list[:T]                                     # [T]
            gamma = 0.99

            # Compute return-to-go G_t
            returns = torch.zeros(T, dtype=torch.float32)
            G = 0.0
            for t in reversed(range(T)):
                if dones_batch[t]: #reset for new episode
                    G = 0.0
                G = rew_batch[t] + gamma * G
                returns[t] = G

            # Simple baseline: normalize returns (works fine for CartPole)
            advantages = returns - returns.mean()
            

            _, logprobs= agent.get_action(obs_batch, act_batch)


            loss = -(logprobs * advantages).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


    f.close()

    if Config.plot:
        plot(filepath=outputfilepath)
        plt.figure(figsize=(8, 4))
        plt.plot(eval_steps, eval_returns, marker='o')
        plt.xlabel("Global Timestep")
        plt.ylabel("Average Evaluation Return")
        plt.title("Greedy Policy Evaluation (CartPole)")
        plt.grid(True)
        plt.show()

    env.close()

