import typing
import chess
import torch
import utils


class Game:
    def __init__(self,
                 board: typing.Optional[str] = None,
                 is_human_white: typing.Optional[bool] = True,
                 depth: typing.Optional[int] = 2) -> None:

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

    def minimax(self, board: chess.Board, depth=0) -> typing.Tuple[float,int]:

        if depth == self.depth:
            return self.get_eval(board), None

        if board.turn != self.is_human_white:
            max_eval = float('-inf')
            max_idx = 0

            for idx, move in enumerate(list(board.legal_moves)):
                board.push(move)

                eval, _ = self.minimax(board, depth+1)

                if eval > max_eval:
                    max_eval = eval
                    max_idx = idx
                # max_eval = max(max_eval,eval)

                board.pop()

            return max_eval, max_idx

        else:
            min_eval = float('inf')
            min_idx = 0

            for idx, move in enumerate(list(board.legal_moves)):
                board.push(move)

                eval, _ = self.minimax(board, depth+1)

                if eval < min_eval:
                    min_eval = eval
                    min_idx = idx
                # min_eval = min(min_eval,eval)

                board.pop()

            return min_eval, min_idx

    def make_human_move(self, move: str) -> bool:
        try:
            self.board.push_san(move)
            return True

        except ValueError:
            print('Illegal move')
            return False

    def make_ai_move(self,) -> None:
        if self.board.turn != self.is_human_white:
            legal_moves = list(self.board.legal_moves)
            self.board.push(legal_moves[self.minimax(self.board)[0]])

    def _get_bb(self, board: chess.Board) -> torch.Tensor:

        bb = torch.from_numpy(utils.convert_to_bb(str(board)))\
            .permute(2, 0, 1)\
            .unsqueeze(0)\
            .float()

        return bb

    def __repr__(self) -> str:
        return self.board.__repr__()