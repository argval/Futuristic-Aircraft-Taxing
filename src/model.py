import numpy as np
import networkx as nx
import pygame
import gymnasium as gym
from gymnasium import spaces
import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPO
from ray.rllib.env.multi_agent_env import MultiAgentEnv
import sys
import torch
import os
import matplotlib.pyplot as plt

# Set the CUDA_VISIBLE_DEVICES environment variable to the GPU index of the L4 (e.g., 0)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Replace "0" with the actual index of the L4 GPU

# Build the large taxiway graph (60x60 layout as described)
connected_taxiways = {
    'North Runway': ('A10', 'A9', 'A8', 'A7', 'A6', 'A5', 'A4', 'A3', 'A2', 'A1'),
    'South Runway': ('H10', 'H9', 'H8', 'H7', 'H6', 'H5', 'H4', 'H3', 'H2', 'H1'),
    'A10': ('North Runway', 'A', 'B6'),
    'A9': ('North Runway', 'A'),
    'A8': ('North Runway', 'A', 'K'),
    'A7': ('North Runway', 'A'),
    'A6': ('North Runway', 'A'),
    'A5': ('North Runway', 'A'),
    'A4': ('North Runway', 'A'),
    'A3': ('North Runway', 'A'),
    'A2': ('North Runway', 'A'),
    'A1': ('North Runway', 'A', 'Q'),
    'A': ('A10', 'B6', 'A9', 'A8', 'K', 'A7', 'A6', 'B4', 'A5', 'B3', 'A4', 'A3', 'B2', 'B1', 'A2', 'A1', 'Q'),
    'B6': ('A10', 'A', 'L5'),
    'K': ('A', 'A8', 'L1_K', 'L2_K', 'L3_K'),
    'B4': ('A', 'L1_TW_K', 'L1_TW_B2', 'L2_TW_K', 'L2_TW_B2', 'L3_TW_K', 'L3_TW_B2'),
    'B3': ('A', 'L4'),
    'B2': ('A', 'B', 'L1_TW_B4', 'L2_TW_B4', 'L3_TW_B4', 'L1_TW_C', 'L2_TW_C', 'L3_TW_D'),
    'B1': ('A', 'B', 'C', 'D'),
    'M': ('A', 'B', 'C', 'D'),
    'N': ('B', 'C', 'D', 'N1'),
    'L5': ('B6'),
    'L1_K': ('K', 'L1_TW_K', 'L1_TW_B2'),
    'L2_K': ('K', 'L2_TW_K', 'L2_TW_B2'),
    'L3_K': ('K', 'L3_TW_K', 'L3_TW_B2'),
    'L4': ('A'),
    'L1_TW_K': ('L1_K', 'B4', 'L1_TW_B2'),
    'L2_TW_K': ('L2_K', 'B4', 'L2_TW_B2'),
    'L3_TW_K': ('L3_K', 'B4', 'L2_TW_B2'),
    'L1_TW_B2': ('B4', 'L1_TW_K', 'L1_TW_B4', 'L1_TW_C'),
    'L2_TW_B2': ('B4', 'L2_TW_K', 'L2_TW_B4', 'L2_TW_C', 'L2_TW_D'),
    'L3_TW_B2': ('B4', 'L2_TW_K', 'L3_TW_B4', 'L3_TW_D'),
    'L1_TW_B4': ('L1_TW_B2', 'B2', 'L1_TW_C'),
    'L2_TW_B4': ('L2_TW_B2', 'B2', 'L2_TW_C', 'L2_TW_D'),
    'L3_TW_B4': ('L3_TW_B2', 'B2', 'L3_TW_D'),
    'L1_TW_C': ('B2', 'L1_TW_B4', 'C'),
    'L2_TW_C': ('B2', 'L2_TW_B4', 'C'),
    'L2_TW_D': ('L2_TW_C', 'L2_TW_B4', 'D'),
    'L3_TW_D': ('B2', 'L3_TW_B4', 'D'),
    'B': ('B2', 'B1', 'M', 'N', 'P', 'A1'),
    'C': ('B1', 'M', 'N', 'D', 'L2_TW_C'),
    'D': ('L3_TW_D', 'B1', 'M', 'N', 'P', 'Q', 'C'),
    'H10': ('South Runway', 'H'),
    'H9': ('South Runway', 'H'),
    'H8': ('South Runway', 'H'),
    'H7': ('South Runway', 'H'),
    'H6': ('South Runway', 'H'),
    'H5': ('South Runway', 'H'),
    'H4': ('South Runway', 'H'),
    'H3': ('South Runway', 'H'),
    'H2': ('South Runway', 'H'),
    'H1': ('South Runway', 'H', 'G1'),
    'H': ('H10', 'H9', 'H8', 'H7', 'H6', 'G4', 'H5', 'H4', 'H3', 'R', 'S', 'P', 'Q', 'H2', 'H1', 'G1'),
    'G1': ('G', 'H', 'H1'),
    'G4': ('H', 'G'),
    'G': ('G4', 'R', 'S', 'P', 'Q', 'G1'),
    'R': ('G', 'H'),
    'S': ('G', 'H'),
    'P': ('H', 'G', 'F', 'N2', 'N1', 'D', 'P1', 'B'),
    'Q': ('H', 'G', 'F', 'N2', 'N1', 'D', 'P1', 'A', 'A1'),
    'P1': ('P', 'Q'),
    'N2': ('P', 'Q'),
    'N1': ('P', 'Q', 'N', 'M'),
    'F': ('P', 'Q')
}

taxiways = list(connected_taxiways.keys())
G = nx.Graph()
G.add_nodes_from(taxiways)
for node, connections in connected_taxiways.items():
    for connection in connections:
        if connection in taxiways:
            G.add_edge(node, connection)

# Custom Multi-Agent Environment using the large taxiway graph
class RoadIntersectionEnv(MultiAgentEnv):
    def __init__(self, graph, num_agents=3):
        super().__init__()
        self.graph = graph
        self.total_agents = num_agents
        self.agents = [f"car_{i}" for i in range(self.total_agents)]
        self.pos = {agent: None for agent in self.agents}
        self.distance_travelled = {agent: 0 for agent in self.agents}

        # Update spaces based on the larger graph
        self.observation_space = spaces.Discrete(len(self.graph.nodes))
        self.action_space = spaces.Discrete(len(list(self.graph.neighbors(list(self.graph.nodes)[0]))))

        # Pygame setup for rendering
        pygame.init()
        self.screen = pygame.display.set_mode((800, 800))
        self.clock = pygame.time.Clock()
        self.positions = {node: (np.random.rand() * 800, np.random.rand() * 800) for node in self.graph.nodes}

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)

        nodes = np.array(list(self.graph.nodes)).flatten()
        for agent in self.agents:
            self.pos[agent] = np.random.choice(nodes)
            self.distance_travelled[agent] = 0
        self.done_agents = set()
        obs = {agent: nodes.tolist().index(self.pos[agent]) for agent in self.agents}
        return obs, {}

    def step(self, action_dict):
        rewards = {agent: 0 for agent in self.agents}
        dones = {agent: False for agent in self.agents}
        infos = {}
        truncateds = {agent: False for agent in self.agents}

        for agent, action in action_dict.items():
            if agent in self.done_agents:
                continue

            current_node = self.pos[agent]
            next_node = list(self.graph.nodes)[action]

            if current_node in self.graph and next_node in self.graph[current_node]:
                self.pos[agent] = next_node
                self.distance_travelled[agent] += nx.get_edge_attributes(self.graph, 'weight').get((current_node, next_node), 1)
                rewards[agent] = -1  # Encourage shorter paths

        # Collision detection
        occupied_nodes = {}
        for agent in self.agents:
            if agent in self.done_agents:
                continue

            pos = self.pos[agent]
            if pos not in occupied_nodes:
                occupied_nodes[pos] = agent
            else:
                # Collision occurred
                rewards[agent] -= 10
                other_agent = occupied_nodes[pos]
                rewards[other_agent] -= 10
                dones[agent] = True
                dones[other_agent] = True
                self.done_agents.add(agent)
                self.done_agents.add(other_agent)

        for agent in self.agents:
            if agent not in self.done_agents:
                dones[agent] = False
            else:
                dones[agent] = True
                truncateds[agent] = True

        all_done = len(self.done_agents) == self.total_agents
        dones['__all__'] = all_done
        truncateds['__all__'] = all_done

        obs = {agent: list(self.graph.nodes).index(self.pos[agent]) for agent in self.agents if agent not in self.done_agents}
        infos = {agent: {} for agent in obs}

        return obs, rewards, dones, truncateds, infos

    def render(self, mode='human'):
        self.screen.fill((255, 255, 255))

        for edge in self.graph.edges:
            if edge[0] in self.positions and edge[1] in self.positions:
                pygame.draw.line(self.screen, (0, 0, 0),
                                 (int(self.positions[edge[0]][0]), int(self.positions[edge[0]][1])),
                                 (int(self.positions[edge[1]][0]), int(self.positions[edge[1]][1])), 1)

        for node in self.graph.nodes:
            if node in self.positions:
                pygame.draw.circle(self.screen, (0, 0, 255),
                                   (int(self.positions[node][0]), int(self.positions[node][1])), 10)

        for agent in self.agents:
            if self.pos[agent] is not None and self.pos[agent] in self.positions:
                pygame.draw.circle(self.screen, (255, 0, 0),
                                   (int(self.positions[self.pos[agent]][0]), int(self.positions[self.pos[agent]][1])), 15)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        pygame.display.flip()
        self.clock.tick(30)

# RLlib Training Configuration
ray.init(ignore_reinit_error=True, num_gpus=1)

from ray.tune.registry import register_env

def env_creator(env_config):
    return RoadIntersectionEnv(env_config['graph'], env_config['num_agents'])

register_env("RoadIntersectionEnv", env_creator)

def policy_mapping_fn(agent_id, *args, **kwargs):
    return "shared_policy"

config = {
    "env": "RoadIntersectionEnv",
    "env_config": {
        "graph": G,
        "num_agents": 3
    },
    "framework": "torch",
    "num_workers": 10,
    "num_envs_per_worker": 20,
    "num_gpus": 1,
    "train_batch_size": 8000,
    "sgd_minibatch_size": 128,
    "lr": 1e-5,
    "gamma": 0.99,
    "log_level": "DEBUG",
    "multiagent": {
        "policies": {
            "shared_policy": (None, spaces.Discrete(len(G.nodes)), spaces.Discrete(len(list(G.neighbors(list(G.nodes)[0])))), {})
        },
        "policy_mapping_fn": policy_mapping_fn,
    },
    "exploration_config": {
        "type": "EpsilonGreedy",
        "initial_epsilon": 1.0,
        "final_epsilon": 0.05,
        "epsilon_timesteps": 20000
    },
    "monitor": True,
    "stop": {
        "training_iteration": 500
    },
    "checkpoint_at_end": True,
    "checkpoint_freq": 5
}

# Train the model using PPO
analysis = tune.run(
    PPO,
    config=config,
    verbose=1,
    storage_path="/tmp/ray_results/logs/",
)

# Save the model
if analysis.trials and analysis.get_best_trial(metric="episode_reward_mean", mode="max") is not None:
    best_trial = analysis.get_best_trial(metric="episode_reward_mean", mode="max")
    best_checkpoint = analysis.get_best_checkpoint(trial=best_trial, metric="episode_reward_mean", mode="max")
    if best_checkpoint is not None:
        print(f"Best checkpoint: {best_checkpoint}")

        env = RoadIntersectionEnv(G, num_agents=3)  # Create an instance of the environment
        obs, _ = env.reset()
        trainer = PPO(config=config)
        trainer.restore(best_checkpoint)

        done = {agent: False for agent in env.agents}
        done['__all__'] = False

        should_render = False

        while not done['__all__']:
            actions = {agent: trainer.compute_single_action(obs[agent]) for agent in env.agents if not done[agent]}
            obs, rewards, done, truncateds, info = env.step(actions)

            if should_render:
                env.render()
    else:
        print("No valid checkpoint found.")
else:
    print("No valid trials found.")
