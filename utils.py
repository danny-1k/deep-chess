import numpy as np


def convert_to_bb(board):
    board = board.replace(' ','')
    bb = np.zeros((8,8,12))
    rows = board.split()
    for r_idx,r in enumerate(rows):

        for c_idx,c in enumerate(r):

            for piece_idx,piece in enumerate('PRNBQKprnbqk'):
                if c == piece:
                    bb[r_idx,c_idx,piece_idx] = 1

    return bb