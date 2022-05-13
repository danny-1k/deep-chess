import os
import chess.pgn
from tqdm import tqdm
import utils
import numpy as np

pgn_files = 'games'
num=0

for f in tqdm(os.listdir(pgn_files)):
    pgn = open(os.path.join(pgn_files, f))
    game = chess.pgn.read_game(pgn)

    for node in game.mainline():

        move_evaluation = node.eval()

        if not move_evaluation:
            continue

        board = str(node.board())

        move_evaluation = move_evaluation.white().wdl(model='lichess').expectation() #prop of winning
        board = utils.convert_to_bb(board)

        move_evaluation = np.array([move_evaluation])

        np.save(f'data/x/{num}.npy',board)
        np.save(f'data/y/{num}.npy',move_evaluation)
        
        num+=1

