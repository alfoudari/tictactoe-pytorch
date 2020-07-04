from .base import TicTacToeEnv as BaseEnv

class TicTacToeEnv(BaseEnv):
    def player2_policy(self):
        while True:
            action = input("Enter cell coordinates (e.g. 1,2): ")
            try:
                action = tuple(map(int, action.split(",")))
            except:
                print("Invalid input")
                continue

            if action in self._board.empty_cells:
                break
            else:
                print("Illegal move")

        self._player2.mark(*action)