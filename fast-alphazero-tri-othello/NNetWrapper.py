from NNetArchitecture import NNetArchitecture as nnetarch
import torch.optim as optim
import torch
from time import time
from pytorch_classification.utils import Bar, AverageMeter
from utils import *
import os
import numpy as np
import math
import sys

sys.path.append('../../')


args = dotdict({
    'lr': 0.01,
    'cuda': torch.cuda.is_available(),
    'num_channels': 128,
    'depth': 5,
})


class NNetWrapper():
    def __init__(self, game):
        self.nnet = nnetarch(game, args)
        self.board_x, self.board_y = game.getBoardSize()
        self.action_size = game.getActionSize()
        self.optimizer = optim.SGD(
            self.nnet.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-3)
        # self.scheduler = optim.lr_scheduler.MultiStepLR(
        #    self.optimizer, milestones=[200,400], gamma=0.1)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, cooldown=10)

        if args.cuda:
            self.nnet.cuda()

    def train(self, batches, train_steps):
        self.nnet.train()

        data_time = AverageMeter()
        batch_time = AverageMeter()
        pi_losses = AverageMeter()
        v_losses = AverageMeter()
        end = time()

        #print(f'Current LR: {self.scheduler.get_lr()[0]}')
        bar = Bar(f'Training Net', max=train_steps)
        current_step = 0
        while current_step < train_steps:
            for batch_idx, batch in enumerate(batches):
                if current_step == train_steps:
                    break
                current_step += 1
                boards, target_pis, target_vs = batch

                # predict
                if args.cuda:
                    boards, target_pis, target_vs = boards.contiguous().cuda(
                    ), target_pis.contiguous().cuda(), target_vs.contiguous().cuda()

                # measure data loading time
                data_time.update(time() - end)

                # compute output
                out_pi, out_v = self.nnet(boards)
                l_pi = self.loss_pi(target_pis, out_pi)
                l_v = self.loss_v(target_vs, out_v)
                total_loss = l_pi + l_v
                # record loss
                pi_losses.update(l_pi.item(), boards.size(0))
                v_losses.update(l_v.item(), boards.size(0))

                # compute gradient and do SGD step
                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()

                # measure elapsed time
                batch_time.update(time() - end)
                end = time()

                # plot progress
                bar.suffix = '({step}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss_pi: {lpi:.4f} | Loss_v: {lv:.3f}'.format(
                    step=current_step,
                    size=train_steps,
                    data=data_time.avg,
                    bt=batch_time.avg,
                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    lpi=pi_losses.avg,
                    lv=v_losses.avg,
                )
                bar.next()
        self.scheduler.step(pi_losses.avg+v_losses.avg)
        bar.finish()
        print()

        return pi_losses.avg, v_losses.avg

    def predict(self, board):
        """
        board: np.ndarray with shape (5, 11, 11)
        return: pi, v: 策略和價值
        
        新版：確保通道順序固定為：通道0-紅色、通道1-綠色、通道2-藍色
        """
        # 檢查輸入格式
        if not isinstance(board, np.ndarray):
            print("警告：board不是numpy數組")
            board = np.array(board)
        
        # 獲取當前玩家順序
        player_order = int(board[3, 0, 0] * 2 + 0.5)  # 0:紅, 1:綠, 2:藍
        
        # 創建標準化的輸入張量（4通道）
        standard_board = np.zeros((4, 11, 11), dtype=np.float32)
        
        # 根據當前玩家順序重新排列通道0-2
        if player_order == 0:  # 紅方回合
            # 標準順序：[紅,綠,藍]，當前順序：[紅,綠,藍]
            standard_board[0:3, :, :] = board[0:3, :, :]
        elif player_order == 1:  # 綠方回合
            # 標準順序：[紅,綠,藍]，當前順序：[綠,藍,紅]
            standard_board[0, :, :] = board[2, :, :]  # 紅方 (位於通道2)
            standard_board[1, :, :] = board[0, :, :]  # 綠方 (位於通道0)
            standard_board[2, :, :] = board[1, :, :]  # 藍方 (位於通道1)
        else:  # 藍方回合
            # 標準順序：[紅,綠,藍]，當前順序：[藍,紅,綠]
            standard_board[0, :, :] = board[1, :, :]  # 紅方 (位於通道1)
            standard_board[1, :, :] = board[2, :, :]  # 綠方 (位於通道2)
            standard_board[2, :, :] = board[0, :, :]  # 藍方 (位於通道0)
        
        # 保留玩家順序標記（通道3）
        standard_board[3, :, :] = board[3, :, :]
        
        # 轉換為張量
        board_tensor = torch.FloatTensor(standard_board).unsqueeze(0)  # 添加批次維度
        
        if args.cuda:
            board_tensor = board_tensor.contiguous().cuda()
        
        with torch.no_grad():
            pi, v = self.nnet.forward(board_tensor)
        
        return pi[0].cpu().numpy(), v[0].cpu().numpy()

    def process(self, batch):
        """
        對批次進行處理，標準化通道順序後再送入神經網絡
        """
        # 轉換通道順序以保持一致（紅-綠-藍）
        batch_size = batch.size(0)
        standard_batch = torch.zeros_like(batch)
        
        for i in range(batch_size):
            # 獲取當前棋盤的玩家順序
            player_order = int(batch[i, 3, 0, 0] * 2 + 0.5)  # 0:紅, 1:綠, 2:藍
            
            # 根據當前玩家順序重新排列通道0-2
            if player_order == 0:  # 紅方回合
                # 標準順序：[紅,綠,藍]，當前順序：[紅,綠,藍]
                standard_batch[i, 0:3, :, :] = batch[i, 0:3, :, :]
            elif player_order == 1:  # 綠方回合
                # 標準順序：[紅,綠,藍]，當前順序：[綠,藍,紅]
                standard_batch[i, 0, :, :] = batch[i, 2, :, :]  # 紅方 (位於通道2)
                standard_batch[i, 1, :, :] = batch[i, 0, :, :]  # 綠方 (位於通道0)
                standard_batch[i, 2, :, :] = batch[i, 1, :, :]  # 藍方 (位於通道1)
            else:  # 藍方回合
                # 標準順序：[紅,綠,藍]，當前順序：[藍,紅,綠]
                standard_batch[i, 0, :, :] = batch[i, 1, :, :]  # 紅方 (位於通道1)
                standard_batch[i, 1, :, :] = batch[i, 2, :, :]  # 綠方 (位於通道2)
                standard_batch[i, 2, :, :] = batch[i, 0, :, :]  # 藍方 (位於通道0)
            
            # 保留玩家順序標記（通道3）
            standard_batch[i, 3, :, :] = batch[i, 3, :, :]
        
        if args.cuda:
            standard_batch = standard_batch.cuda()
            
        self.nnet.eval()
        with torch.no_grad():
            pi, v = self.nnet(standard_batch)
            return torch.exp(pi), v

    def loss_pi(self, targets, outputs):
        return -torch.sum(targets * outputs) / targets.size()[0]

    def loss_v(self, targets, outputs):
        return torch.sum((targets - outputs) ** 2) / targets.size()[0]

    def save_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(folder):
            os.mkdir(folder)
        torch.save({
            'state_dict': self.nnet.state_dict(),
            'opt_state': self.optimizer.state_dict(),
            'sch_state': self.scheduler.state_dict()
        }, filepath)

    def load_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        # https://github.com/pytorch/examples/blob/master/imagenet/main.py#L98
        filepath = os.path.join(folder, filename)
        if not os.path.exists(filepath):
            raise ("No model in path {}".format(filepath))
        checkpoint = torch.load(filepath)
        self.nnet.load_state_dict(checkpoint['state_dict'])
        if 'opt_state' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['opt_state'])
        if 'sch_state' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['sch_state'])
