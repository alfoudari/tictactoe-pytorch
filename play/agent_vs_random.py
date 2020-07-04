import time
import numpy as np
import argparse
from env.base import Player
from .game import Game

def readable_board(board):
    matrix = np.asarray(board)
    # map numbers to letters
    v = np.vectorize(lambda x: dict(zip([0, 1, 2], [" ", "X", "O"]))[x])
    matrix = v(matrix)
    
    readable = []
    for row in matrix:
        row_string = f" {row[0]} | {row[1]} | {row[2]} "
        readable.append(row_string)
        readable.append("-" * len(row_string))
    return readable

if __name__ == '__main__':
    # flags
    parser = argparse.ArgumentParser()
    parser.add_argument('-algorithm', action="store", dest="algorithm", type=str, required=True, choices=["dqn","a2c"])
    parser.add_argument('-net_type', action="store", dest="net_type", type=str, required=True, choices=["fc","cnn"])
    parser.add_argument('-policy', action="store", dest="policy", type=str, default="")
    parser.add_argument('-num_games', action="store", dest="num_games", type=int, default=1000)
    args = parser.parse_args()

    game = Game()
    game.load_algorithm(args.algorithm)
    game.load_net(args.net_type)
    game.load_env("TicTacToeRandomPlayer-v0")
    game.load_model(args.policy)

    x_won = o_won = draw = 0
    won_games = []
    lost_games = []
    for n in range(args.num_games):
        start = time.time()

        game.env.seed(start)
        game.play()
        board = game.env._board

        if board.player_won:
            if board.player_won.side == Player.X.value:
                x_won += 1
                won_games.append((n,readable_board(board.board)))
            elif board.player_won.side == Player.O.value:
                o_won += 1
                if board.board not in [b for _, b in lost_games]:
                    lost_games.append((n,board.board))
        else:
            draw += 1

        end = time.time()

        print(f"Game #{n+1} - Iteration duration: {end - start}", end="\r", flush=True)
        
    print("\n")
    print(f"X Won: {x_won}")
    print(f"O Won: {o_won}")
    print(f"Draw:  {draw}")
    print(f"Win percentage: {x_won / args.num_games * 100}")
    print(f"Win+Draw percentage: {(x_won + draw) / args.num_games * 100}")
    print(f"Loss percentage: {o_won / args.num_games * 100}")

    # print("Lost games:")
    # for idx, (n, board) in enumerate(lost_games):
    #     print(f"#{n}")
    #     for row in readable_board(board):
    #         print(row)

    #     if idx > 10:
    #         break

    # print("Won game: ")
    # for row in won_games[0][1]:
    #     print(row)
