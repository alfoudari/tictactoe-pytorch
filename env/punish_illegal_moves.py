from .base import TicTacToeEnv as BaseEnv

class TicTacToeEnv(BaseEnv):
    def step(self, action):
        if self._done:
            # The last action ended the episode. Start a new episode.
            return self.reset()

        # e.g.
        #  0 -> (0,0)
        #  2 -> (0,2)
        #  4 -> (1,1)
        coordinates = (int(action/3), int(action % 3))

        # illegal move
        illegal_move = coordinates not in self._board.empty_cells

        if not illegal_move:
            # Player 1
            self._player1.mark(*coordinates)

            if self._board.player_won or not self._board.empty_cells:
                self._done = True

            # Player 2
            if not self._done and self._board.empty_cells:
                self.player2_policy()
                if self._board.player_won or not self._board.empty_cells:
                    self._done = True

        self._board_state_to_int()

        if self._done:
            # Win: reward is 1 + # of empty cells remaining, this is to 
            # encourage efficient strategies (the ones that win faster).
            # Loss: rewards -1 - # of empty cells; the agent chances to
            # win decrease the longer it plays so early losses are punished.
            # Draw: 0
            if self._board.player_won == self._player1:
                reward = 1+len(self._board.empty_cells)
            elif self._board.player_won == self._player2:
                reward = -1-len(self._board.empty_cells)
            else:
                # draw
                reward = 1

            return (self._state, reward, True, dict())
        else:
            if illegal_move:
                # end episode
                return (self._state, -5, True, dict())
            else:
                return (self._state, 1, False, dict())