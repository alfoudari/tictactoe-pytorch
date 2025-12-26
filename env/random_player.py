from .base import TicTacToeEnv as BaseEnv


class TicTacToeEnv(BaseEnv):
    def player2_policy(self):
        random_action = self.np_random.choice(len(self._board.empty_cells))
        self._player2.mark(*self._board.empty_cells[random_action])