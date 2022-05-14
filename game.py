import typing
import chess
import torch
import utils


class Game:
    def __init__(self,
                 board: typing.Optional(str) = None,
                 is_human_white: typing.Optional(bool) = True,
                 ai_depth: typing.Optional(int) = 2) -> None:

        self.board = chess.Board(board)
        self.is_human_white = is_human_white
        self.ai_depth = ai_depth

        self.net = torch.load('checkpoints/models/Net.pt')
        self.net.load_checkpoint()
        self.net.eval()
        self.net.requires_grad_(False)


    def get_eval(self, board):
        pass

    def choose_move(self, board):
        pass

    def make_human_move(self, move):
        pass

    def make_ai_move(self, move):
        pass

    def _get_bb(board):
        pass
