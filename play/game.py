
import importlib
import time
import sys
import torch
import gym
from env.base import Player

# mapping between algorithms and their module paths
algorithm_module_paths = dict(
    dqn = "train.dqn",
    a2c = "train.a2c",
)

# mapping between nets and their classes
net_type_classes = dict(
    fc = "FCNet",
    cnn = "CNNNet",
)

class Game:
    def __init__(self, verbose=False, debug=False, first_player=None):
        self.algorithm = None
        self.algorithm_module = None
        self.net = None
        self.env = None
        self.model = None
        self.verbose = verbose
        self.debug = debug
        self.first_player = first_player

    def load_algorithm(self, algorithm, module_paths=algorithm_module_paths):
        try:
            self.algorithm = algorithm
            self.algorithm_module = importlib.import_module(module_paths[algorithm])
        except Exception as e:
            sys.exit(e)

    def load_net(self, net_type):
        try:
            self.net = getattr(self.algorithm_module, net_type_classes[net_type])
        except Exception as e:
            sys.exit(e)

    def load_env(self, env):
        kwargs = dict()

        kwargs["first_player"] = Player(self.first_player) if self.first_player else None
        if self.verbose:
            kwargs["player1_verbose"] = True
            kwargs["player2_verbose"] = True
            kwargs["board_verbose"] = True
        
        self.env = gym.make(env, **kwargs)

        # seed
        self.env.seed(time.time())

    def load_model(self, policy):
        self.model = self.net(obs_size=9, n_actions=9)
        self.model.load_state_dict(torch.load(f"policies/{policy}"))
        self.model.eval()

    def play(self):
        # environment
        obs = self.env.reset()

        while True:
            x = torch.Tensor(obs).reshape(self.env.observation_space_n)
            mask = torch.zeros(self.env.action_space_n).index_fill(0, torch.LongTensor(self.env.legal_actions),  1)

            if self.algorithm == "dqn":
                y = self.model(x, mask)
            elif self.algorithm == "a2c":
                y, _ = self.model(x, mask)

            action = torch.argmax(y).item()

            if self.debug:
                print(f"action distribution:\n{y.view(3,3)}")
                print(f"action, max(action_dist): {action}, {torch.max(y)}\n")

            obs, reward, done, _ = self.env.step(action)

            if done:
                break