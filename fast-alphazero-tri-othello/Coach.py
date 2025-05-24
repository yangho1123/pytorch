from MCTS import MCTS
from SelfPlayAgent import SelfPlayAgent
import torch
from pathlib import Path
from glob import glob
from torch import multiprocessing as mp
from torch.utils.data import TensorDataset, ConcatDataset, DataLoader
from tensorboardX import SummaryWriter
from Arena import Arena
from GenericPlayers import RandomPlayer, NNPlayer
from pytorch_classification.utils import Bar, AverageMeter
from queue import Empty
from time import time
import numpy as np
from math import ceil
import os


class Coach:
    def __init__(self, game, nnet, args):
        np.random.seed()
        self.game = game
        self.nnet = nnet
        self.pnet = self.nnet.__class__(self.game)
        self.args = args

        networks = sorted(glob(self.args.checkpoint+'/*'))
        self.args.startIter = len(networks)
        if self.args.startIter == 0:
            self.nnet.save_checkpoint(
                folder=self.args.checkpoint, filename='iteration-0000.pkl')
            self.args.startIter = 1

        self.nnet.load_checkpoint(
            folder=self.args.checkpoint, filename=f'iteration-{(self.args.startIter-1):04d}.pkl')

        self.agents = []
        self.input_tensors = []
        self.policy_tensors = []
        self.value_tensors = []
        self.batch_ready = []
        self.ready_queue = mp.Queue()
        self.file_queue = mp.Queue()
        self.result_queue = mp.Queue()
        self.completed = mp.Value('i', 0)
        self.games_played = mp.Value('i', 0)
        if self.args.run_name != '':
            self.writer = SummaryWriter(log_dir='runs/'+self.args.run_name)
        else:
            self.writer = SummaryWriter()
        self.args.expertValueWeight.current = self.args.expertValueWeight.start

    def learn(self):
        print('Because of batching, it can take a long time before any games finish.')
        for i in range(self.args.startIter, self.args.numIters + 1):
            print(f'------ITER {i}------')
            self.generateSelfPlayAgents()
            print("generateSelfPlayAgents done")
            self.processSelfPlayBatches()
            print("processSelfPlayBatches done")
            self.saveIterationSamples(i)
            print("saveIterationSamples done")
            self.processGameResults(i)
            print("processGameResults done")
            self.killSelfPlayAgents()
            print("killSelfPlayAgents done")
            self.train(i)
            print("train done")
            if self.args.compareWithRandom and (i-1) % self.args.randomCompareFreq == 0:
                if i == 1:
                    print(
                        'Note: Comparisons with Random do not use monte carlo tree search.')
                self.compareToRandom(i)
            if self.args.compareWithPast and (i - 1) % self.args.pastCompareFreq == 0:
                self.compareToPast(i)
            z = self.args.expertValueWeight
            self.args.expertValueWeight.current = min(
                i, z.iterations)/z.iterations * (z.end - z.start) + z.start
            print()
        self.writer.close()

    def generateSelfPlayAgents(self):
        self.ready_queue = mp.Queue()
        boardx, boardy = self.game.getBoardSize()
        for i in range(self.args.workers):
            self.input_tensors.append(torch.zeros([self.args.process_batch_size, 4, boardx, boardy]))
            self.input_tensors[i].pin_memory()
            self.input_tensors[i].share_memory_()

            self.policy_tensors.append(torch.zeros(
                [self.args.process_batch_size, self.game.getActionSize()]))
            self.policy_tensors[i].pin_memory()
            self.policy_tensors[i].share_memory_()

            self.value_tensors.append(torch.zeros(
                [self.args.process_batch_size, 1]))
            self.value_tensors[i].pin_memory()
            self.value_tensors[i].share_memory_()
            self.batch_ready.append(mp.Event())

            self.agents.append(
                SelfPlayAgent(i, self.game, self.ready_queue, self.batch_ready[i],
                              self.input_tensors[i], self.policy_tensors[i], self.value_tensors[i], self.file_queue,
                              self.result_queue, self.completed, self.games_played, self.args))
            self.agents[i].start()

    def processSelfPlayBatches(self):   #主進程？
        print("開始處理自我對弈批次")
        sample_time = AverageMeter()
        bar = Bar('Generating Samples', max=self.args.gamesPerIteration)
        end = time()
        
        n = 0
        #卡在這個迴圈
        while self.completed.value != self.args.workers:
            try:
                id = self.ready_queue.get(timeout=1)
                self.policy, self.value = self.nnet.process(self.input_tensors[id])
                self.policy_tensors[id].copy_(self.policy)      # policy總和=128.0
                self.value_tensors[id].copy_(self.value)
                self.batch_ready[id].set()
            except Empty:
                pass
            size = self.games_played.value
            if size > n:
                sample_time.update((time() - end) / (size - n), size - n)
                n = size
                end = time()
            bar.suffix = f'({size}/{self.args.gamesPerIteration}) Sample Time: {sample_time.avg:.3f}s | Total: {bar.elapsed_td} | ETA: {bar.eta_td:}'
            bar.goto(size)
        bar.update()
        bar.finish()
        print()

    def killSelfPlayAgents(self):
        for i in range(self.args.workers):
            self.agents[i].join()
            del self.input_tensors[0]
            del self.policy_tensors[0]
            del self.value_tensors[0]
            del self.batch_ready[0]
        self.agents = []
        self.input_tensors = []
        self.policy_tensors = []
        self.value_tensors = []
        self.batch_ready = []
        self.ready_queue = mp.Queue()
        self.completed = mp.Value('i', 0)
        self.games_played = mp.Value('i', 0)

    def saveIterationSamples(self, iteration):
        num_samples = self.file_queue.qsize()
        print(f'Saving {num_samples} samples')
        boardx, boardy = self.game.getBoardSize()
        data_tensor = torch.zeros([num_samples, boardx, boardy])
        policy_tensor = torch.zeros([num_samples, self.game.getActionSize()])
        value_tensor = torch.zeros([num_samples, 1])
        for i in range(num_samples):
            data, policy, value = self.file_queue.get()
            data_tensor[i] = torch.from_numpy(data)
            policy_tensor[i] = torch.tensor(policy)
            value_tensor[i, 0] = float(value)

        os.makedirs(self.args.data, exist_ok=True)

        torch.save(
            data_tensor, f'{self.args.data}/iteration-{iteration:04d}-data.pkl')
        torch.save(policy_tensor,
                   f'{self.args.data}/iteration-{iteration:04d}-policy.pkl')
        torch.save(
            value_tensor, f'{self.args.data}/iteration-{iteration:04d}-value.pkl')
        del data_tensor
        del policy_tensor
        del value_tensor

    def processGameResults(self, iteration):
        num_games = self.result_queue.qsize()
        red_wins = 0
        green_wins = 0
        blue_wins = 0
        draws = 0
        for _ in range(num_games):
            result = self.result_queue.get()  # 結果應該是一個三元素列表 [red_score, green_score, blue_score]
            if result[0] > 0:  # 紅方勝
                red_wins += 1
            elif result[1] > 0:  # 綠方勝
                green_wins += 1
            elif result[2] > 0:  # 藍方勝
                blue_wins += 1
            else:
                draws += 1
        # 更新記錄
        self.writer.add_scalar('win_rate/red', red_wins/num_games, iteration)
        self.writer.add_scalar('win_rate/green', green_wins/num_games, iteration)
        self.writer.add_scalar('win_rate/blue', blue_wins/num_games, iteration)
        self.writer.add_scalar('win_rate/draws', draws/num_games, iteration)

    def train(self, iteration):
        datasets = []
        #currentHistorySize = self.args.numItersForTrainExamplesHistory
        currentHistorySize = min(
            max(4, (iteration + 4)//2),
            self.args.numItersForTrainExamplesHistory)
        for i in range(max(1, iteration - currentHistorySize), iteration + 1):
            data_tensor = torch.load(
                f'{self.args.data}/iteration-{i:04d}-data.pkl')
            policy_tensor = torch.load(
                f'{self.args.data}/iteration-{i:04d}-policy.pkl')
            value_tensor = torch.load(
                f'{self.args.data}/iteration-{i:04d}-value.pkl')
            datasets.append(TensorDataset(
                data_tensor, policy_tensor, value_tensor))

        dataset = ConcatDataset(datasets)
        dataloader = DataLoader(dataset, batch_size=self.args.train_batch_size, shuffle=True,
                                num_workers=self.args.workers, pin_memory=True)

        l_pi, l_v = self.nnet.train(
            dataloader, self.args.train_steps_per_iteration)
        self.writer.add_scalar('loss/policy', l_pi, iteration)
        self.writer.add_scalar('loss/value', l_v, iteration)
        self.writer.add_scalar('loss/total', l_pi + l_v, iteration)

        self.nnet.save_checkpoint(
            folder=self.args.checkpoint, filename=f'iteration-{iteration:04d}.pkl')

        del dataloader
        del dataset
        del datasets

    def compareToPast(self, iteration):
        past = max(0, iteration-self.args.pastCompareFreq)
        self.pnet.load_checkpoint(folder=self.args.checkpoint,
                                  filename=f'iteration-{past:04d}.pkl')
        print(f'PITTING AGAINST ITERATION {past}')
        if(self.args.arenaMCTS):
            pplayer = MCTS(self.game, self.pnet, self.args)
            nplayer = MCTS(self.game, self.nnet, self.args)

            def playpplayer(x, turn):
                if turn <= 2:
                    pplayer.reset()
                temp = self.args.temp if turn <= self.args.tempThreshold else self.args.arenaTemp
                policy = pplayer.getActionProb(x, temp=temp)
                return np.random.choice(len(policy), p=policy)

            def playnplayer(x, turn):
                if turn <= 2:
                    nplayer.reset()
                temp = self.args.temp if turn <= self.args.tempThreshold else self.args.arenaTemp
                policy = nplayer.getActionProb(x, temp=temp)
                return np.random.choice(len(policy), p=policy)

            arena = Arena(playnplayer, playpplayer, self.game)
        else:
            pplayer = NNPlayer(self.game, self.pnet, self.args.arenaTemp)
            nplayer = NNPlayer(self.game, self.nnet, self.args.arenaTemp)

            arena = Arena(nplayer.play, pplayer.play, self.game)
        nwins, pwins, draws = arena.playGames(self.args.arenaCompare)

        print(f'NEW/PAST WINS : {nwins} / {pwins} ; DRAWS : {draws}\n')
        self.writer.add_scalar(
            'win_rate/past', float(nwins + 0.5 * draws) / (pwins + nwins + draws), iteration)

    def compareToRandom(self, iteration):
        r = RandomPlayer(self.game)
        nnplayer = NNPlayer(self.game, self.nnet, self.args.arenaTemp)
        print('PITTING AGAINST RANDOM')

        arena = Arena(nnplayer.play, r.play, self.game)
        nwins, pwins, draws = arena.playGames(self.args.arenaCompareRandom)

        print(f'NEW/RANDOM WINS : {nwins} / {pwins} ; DRAWS : {draws}\n')
        self.writer.add_scalar(
            'win_rate/random', float(nwins + 0.5 * draws) / (pwins + nwins + draws), iteration)
