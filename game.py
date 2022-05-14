import typing
import chess
import torch
import utils


class Game:
    def __init__(self,
                 board: typing.Optional(str) = None,
                 is_human_white: typing.Optional(bool) = True,
                 depth: typing.Optional(int) = 2) -> None:

        self.board = chess.Board(board)
        self.is_human_white = is_human_white
        self.depth = depth

        self.net = torch.load('checkpoints/models/Net.pt')
        self.net.load_checkpoint()
        self.net.eval()
        self.net.requires_grad_(False)

    def get_eval(self, board: chess.Board) -> float:

        # gives board evaluation based on the current side

        bb = self._get_bb(board)
        eval = self.net(bb).squeeze().item()
        eval = eval if board.turn else -eval

        return eval

    def choose_move(self, board: chess.Board, depth=0) :

        best_eval = float('-inf')
        best_move_idx = 0


        for idx,move in enumerate(list(board.legal_moves)):

            board.push(move)

            if depth == self.depth: # if on last branch, return static evaluation
                
                eval = self.get_eval(board)
                
                if eval > best_eval:
                    best_eval = eval
                    best_move_idx = idx

            else: # recusrive evaluation

                recursion_out = self.choose_move(board,depth+1)

                if recursion_out[0] > best_eval:
                    best_eval = recursion_out[0]
                    best_move_idx = idx

            board.pop()

        return best_eval,best_move_idx

    def make_human_move(self, move: str) -> bool:
        try:
            self.board.push_san(move)
            return True

        except ValueError:
            print('Illegal move')
            return False

    def make_ai_move(self,):
        pass

    def _get_bb(board: chess.Board) -> torch.Tensor:

        bb = torch.from_numpy(utils.convert_to_bb(str(board)))\
            .permute(2, 0, 1)\
            .unsqueeze(0)\
            .float()

        return bb
