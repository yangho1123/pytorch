import pyximport; pyximport.install()

from torch import multiprocessing as mp

from Coach import Coach
from NNetWrapper import NNetWrapper as nn
#from connect4.Connect4Game import Connect4Game as Game
from Game import Game
from utils import *
from math import sqrt

args = dotdict({
    'run_name': 'othello_debug',
    'workers': 1,   # mp.cpu_count() - 1
    'startIter': 1,
    'numIters': 2,
    'process_batch_size': 1,    #128
    'train_batch_size': 128,            # 512
    'train_steps_per_iteration': 500,   # 2000
    'gamesPerIteration': 10,      # 4*128*(mp.cpu_count()-1)
    'numItersForTrainExamplesHistory': 100,
    'symmetricSamples': False,
    'numMCTSSims': 10,
    'numFastSims': 5,
    'probFastSim': 0,
    'tempThreshold': 10,
    'temp': 1,
    'compareWithRandom': False,
    'arenaCompareRandom': 10,
    'arenaCompare': 10,
    'arenaTemp': 0.1,
    'arenaMCTS': False,
    'randomCompareFreq': 1,
    'compareWithPast': False,
    'pastCompareFreq': 3,
    'expertValueWeight': dotdict({
        'start': 0,
        'end': 0,
        'iterations': 35
    }),
    'cpuct': sqrt(3),
    'checkpoint': 'checkpoint',
    'data': 'data',
})

if __name__ == "__main__":
    g = Game()
    nnet = nn(g)
    c = Coach(g, nnet, args)
    c.learn()
