import torch
import argparse
from .game import Game

if __name__ == '__main__':
    # flags
    parser = argparse.ArgumentParser()
    parser.add_argument('-fp', action="store", dest="fp", type=int, default=1, choices=[1,2])
    parser.add_argument('-algorithm', action="store", dest="algorithm", type=str, required=True, choices=["dqn","a2c"])
    parser.add_argument('-net_type', action="store", dest="net_type", type=str, required=True, choices=["fc","cnn"])
    parser.add_argument('-policy', action="store", dest="policy", type=str, required=True)
    parser.add_argument('-debug', action="store_true", dest="debug")
    args = parser.parse_args()

    game = Game(verbose=True, debug=args.debug, first_player=args.fp)
    game.load_algorithm(args.algorithm)
    game.load_net(args.net_type)
    game.load_env("TicTacToeHumanPlayer-v0")
    game.load_model(args.policy)

    # start playing
    game.play()

    board = game.env._board
    if board.player_won:
        if board.player_won.side == "X":
            print("AI won!")
        elif board.player_won.side == "O":
            print("You won!")
    else:
        print("Draw.")