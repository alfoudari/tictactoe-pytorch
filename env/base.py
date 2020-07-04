import abc
import random
import gym
import numpy as np
import warnings

from itertools import chain
from enum import Enum
from gym import spaces
from gym.utils import seeding
from tictactoe import engine as tictactoe

# https://stackoverflow.com/questions/40659212/futurewarning-elementwise-comparison-failed-returning-scalar-but-in-the-futur
warnings.simplefilter(action='ignore', category=FutureWarning)

class Player(Enum):
    X = 1
    O = 2

class TicTacToeEnv(gym.Env):
    environment_name = "TicTacToe Environment"

    def __init__(self, 
                 player1_verbose=False, 
                 player2_verbose=False, 
                 board_verbose=False, 
                 first_player=None,
                 rewards=dict(
                     pos_ep=1,
                     neg_ep=-1,
                     draw=0.5,
                     step=-0.1,
                 ),
                 thresholds=dict(
                     win_rate=0.9,
                     draw_rate=0.1,
                 )):
        # spaces
        self.action_space = spaces.Discrete(9)
        self.observation_space = spaces.Tuple(
            (spaces.Discrete(3),spaces.Discrete(3))
        )

        # game
        self._board = tictactoe.Board(verbose=board_verbose)
        self._player1 = tictactoe.Player(board=self._board, side=1, verbose=player1_verbose) # X
        self._player2 = tictactoe.Player(board=self._board, side=2, verbose=player2_verbose) # O
        self._first_player = first_player
        self._done = False

        # rewards
        self.rewards = rewards

        # thresholds
        self.thresholds = thresholds
        s = sum([n for k,n in self.thresholds.items()])
        if s != 1:
            raise Exception(f"thresholds must equal to 1; got {s} instead.")

        # stats
        self.stats = dict(
            games_played=0
        )

        # do stuff
        self.seed()

    def step(self, action):
        if self._done:
            # The last action ended the episode. Start a new episode.
            return self.reset()

        # e.g.
        #  0 -> (0,0)
        #  2 -> (0,2)
        #  4 -> (1,1)
        coordinates = (int(action/3), int(action % 3))

        # Player 1
        self._player1.mark(*coordinates)

        if self._board.player_won or not self._board.empty_cells:
            self._done = True

        # Player 2
        if not self._done and self._board.empty_cells:
            self.player2_policy()
            if self._board.player_won or not self._board.empty_cells:
                self._done = True

        if self._done:
            if self._board.player_won == self._player1:
                reward = self.rewards['pos_ep']
            elif self._board.player_won == self._player2:
                reward = self.rewards['neg_ep']
            else:
                # draw
                reward = self.rewards['draw']

            # adjust stats
            self.stats['games_played'] += 1
        else:
            reward = self.rewards['step']
            
        return (self.state, reward, self._done, dict())

    def reset(self):
        # reset everything
        self._board.reset()
        self._first_move()
        self._done = False
        # return an initial observation
        return self.state

    def render(self):
        pass

    def close(self):
        pass

    def seed(self, seed=1):
        random.seed(seed) # fixed seed

    @abc.abstractmethod
    def player2_policy(self):
        """
        Define player 2 (non-agent) policy
        """

    @property
    def legal_actions(self):
        cells = self._board.empty_cells
        actions = [x*3+y for (x,y) in cells]
        return actions

    @property
    def state(self):
        # flatten state
        board = self._board.board
        return list(chain.from_iterable(board))

    @property
    def observation_space_n(self):
        return self.observation_space[0].n * self.observation_space[1].n

    @property
    def action_space_n(self):
        return self.action_space.n

    @property
    def performance_threshold(self):
        # ideal moves accumulated rewards
        win = 2 * self.rewards['step'] + self.rewards['pos_ep']
        # play till the end and draw
        first_move_draw = 4 * self.rewards['step'] + self.rewards['draw'] # 4 moves + final move
        second_move_draw = 3 * self.rewards['step'] + self.rewards['draw'] # 3 moves + final move

        # win some % of the time, draw some % of the time
        weighed_rewards = win * self.thresholds['win_rate'] + first_move_draw * self.thresholds['draw_rate']/2 + second_move_draw * self.thresholds['draw_rate']/2

        return weighed_rewards

    def _first_move(self):
        if self._first_player is None:
            # random
            self._first_player = random.choice(list(Player))
        
        if self._first_player == Player.O:
            # Let O (player 2) play first
            self.player2_policy()