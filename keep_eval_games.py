import os
from tqdm import tqdm

def has_eval(f):
    with open(f,'r') as f:
        return '[%eval' in f.read()


def delete_no_eval():
    for f in tqdm(os.listdir('/games')):
        if not has_eval(f):
            os.remove(f)


def count_eval_files():
    count = 0

    for f in tqdm(os.listdir('games')):
        if has_eval(os.path.join('games',f)):
            count+=1
            open('count.txt','w').write(str(count))
    
    print('Number of evaluated pgn files',count)


def keep_eval_files():
    for f in tqdm(os.listdir('games')):
        if not has_eval(os.path.join('games',f)):
            os.remove(os.path.join('games',f))

# count_eval_files()

keep_eval_files()