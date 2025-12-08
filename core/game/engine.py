import numpy as np
import random

class Board:
    def __init__(self, verbose=False):
        self.reset()
        self.verbose = verbose
        
    def reset(self):
        self.board = [[0]*3 for _ in range(3)]
        self.player_won = None
        self.empty_cells = [(x, y) for x in range(0,3) for y in range(0,3)]

    def winning_row(self, side=None):
        def _consecutive_cells(arr, x):
            for row in arr:
                if row and all([cell==side for cell in row]):
                    return True
            return False

        horizontal = _consecutive_cells(self.board, side)
        vertical   = _consecutive_cells(np.transpose(self.board).tolist(), side)
        diagonal   = all([cell==side for cell in np.diag(self.board).tolist()])
        opp_diag   = all([cell==side for cell in np.diag(np.fliplr(self.board))])

        return horizontal or vertical or diagonal or opp_diag

    def readable(self):
        matrix = np.asarray(self.board)
        v = np.vectorize(lambda x: dict(zip([0, 1, 2], [" ", "X", "O"]))[x])
        matrix = v(matrix)
        result = ""
        for row in matrix:
            row_string = f" {row[0]} | {row[1]} | {row[2]} \n"
            result = result + row_string
            result = result + ("-" * len(row_string)) + "\n"
        return result


class Player:
    def __init__(self, board=None, side=None, verbose=False):
        self.board = board
        self.side = side # X=1 or O=2
        self.verbose = verbose

    def mark(self, x, y):
        if self.verbose:
            print(f"{self}: Marking {x} {y}")

        self.board.board[x][y] = self.side
        self.board.empty_cells.remove((x,y))

        if self.board.verbose:
            print(self.board.readable())
        
        if self.board.winning_row(self.side):
            self.board.player_won = self
            if self.verbose:
                print(f"{self} won!")

    def __str__(self):
        return f"Player {self.side}"
