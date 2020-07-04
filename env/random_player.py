from .base import TicTacToeEnv as BaseEnv
import random

class TicTacToeEnv(BaseEnv):
    def player2_policy(self):
        random_action = random.choice(self._board.empty_cells)
        self._player2.mark(*random_action)