#!/usr/bin/env python3
import logging
import os
import sys
import time
import numpy as np
import torch
import torch.multiprocessing as mp
from tqdm import tqdm
from collections import deque
import pickle
import argparse

# 設置系統路徑，確保可以導入所需的模塊
sys.path.append('.')
from Game import Game
from NNetWrapper import NNetWrapper, args as nnet_args
from SelfPlayAgent import SelfPlayAgent
from utils import *

# 配置日誌
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Parameter:
    """
    可調整參數類，支持線性調整
    """
    def __init__(self, value):
        self.current = value
        self.init = value
        
    def adjust(self, progress, param_range=None):
        """根據訓練進程調整參數值"""
        if param_range is None:
            return
        self.current = self.init + progress * (param_range - self.init)
        
    def __getstate__(self):
        """支持序列化的狀態獲取方法"""
        return {'current': self.current, 'init': self.init}
        
    def __setstate__(self, state):
        """從序列化狀態恢復"""
        self.current = state['current']
        self.init = state['init']

def self_play(args):
    """
    使用Cython加速的自我對弈數據生成主函數
    
    參數:
        args: 配置參數
    """
    # 初始化遊戲環境
    game = Game()
    
    # 更新神經網絡參數
    nnet_args.lr = args.lr
    nnet_args.dropout = args.dropout
    nnet_args.cuda = args.cuda
    nnet_args.num_channels = 128  # 改為128，與模型定義匹配
    
    # 初始化神經網絡
    nnet = NNetWrapper(game)
    
    # 確定設備（CPU或CUDA）
    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    
    # 如果存在預訓練模型，載入它
    if args.load_model:
        logger.info(f"載入預訓練模型: {args.load_folder_file}")
        nnet.load_checkpoint(args.load_folder_file[0], args.load_folder_file[1])
    else:
        logger.warning("未載入預訓練模型！從頭開始訓練。")
    
    # 確保模型處於評估模式
    nnet.nnet.eval()
    
    # 設置多進程數據共享
    model_params = nnet.nnet.state_dict()
    
    # 設置共享記憶體張量，用於神經網絡的輸入和輸出
    # 神經網絡輸入張量: 形狀為[進程數 * args.nn_batch_size, 3, 11, 11]的棋盤狀態
    batch_tensor = torch.zeros(args.cpus * args.nn_batch_size, 3, 11, 11).share_memory_()
    
    # 策略輸出張量：形狀為[進程數 * args.nn_batch_size, 動作空間大小]的動作概率
    policy_tensor = torch.zeros(args.cpus * args.nn_batch_size, game.getActionSize()).share_memory_()
    
    # 價值輸出張量：形狀為[進程數 * args.nn_batch_size, 1]的勝率預測
    value_tensor = torch.zeros(args.cpus * args.nn_batch_size, 1).share_memory_()
    
    # 設置多進程通信隊列和同步原語
    ready_queue = mp.Queue()  # 準備好的批次隊列
    batch_ready = mp.Event()  # 批次準備就緒的事件
    batch_ready.clear()
    output_queue = mp.Queue()  # 輸出訓練樣本的隊列
    result_queue = mp.Queue()  # 遊戲結果的隊列
    
    # 多進程共享計數器
    games_played = mp.Value('i', 0)  # 已完成的遊戲數量
    complete_count = mp.Value('i', 0)  # 完成的進程計數
    
    # 創建並啟動自我對弈代理進程
    agents = []
    for i in range(args.cpus):
        agent = SelfPlayAgent(
            i, game, ready_queue, batch_ready, 
            batch_tensor[i*args.nn_batch_size:(i+1)*args.nn_batch_size],
            policy_tensor[i*args.nn_batch_size:(i+1)*args.nn_batch_size],
            value_tensor[i*args.nn_batch_size:(i+1)*args.nn_batch_size],
            output_queue, result_queue, complete_count, games_played, args
        )
        agent.start()
        agents.append(agent)
    
    # 記錄結果
    wins = {1: 0, -1: 0, 0: 0}
    game_lengths = []
    examples = []
    
    # 主循環：處理批次預測任務
    with torch.no_grad():
        pbar = tqdm(total=args.gamesPerIteration, desc="自我對弈進度")
        old_games_played = 0
        
        while games_played.value < args.gamesPerIteration:
            # 等待代理提交批次
            ids = []
            while not ready_queue.empty():
                ids.append(ready_queue.get())
            
            if not ids:
                # 如果沒有代理提交批次，檢查是否有新的遊戲結果
                while not result_queue.empty():
                    winner = result_queue.get()
                    wins[winner] += 1
                
                # 更新進度條
                if games_played.value > old_games_played:
                    pbar.update(games_played.value - old_games_played)
                    old_games_played = games_played.value
                
                time.sleep(0.001)  # 避免過度佔用CPU
                continue
            
            try:
                # 運行神經網絡前向傳播
                nn_input = batch_tensor.to(device)
                policy, value = nnet.nnet(nn_input)
                
                # 將結果存入共享張量
                policy_tensor.copy_(policy.cpu())
                value_tensor.copy_(value.cpu())
                
                # 通知所有代理批次已處理完畢
                batch_ready.set()
                
            except Exception as e:
                logger.error(f"處理批次時出錯: {e}")
                continue
            
            # 收集來自輸出隊列的訓練樣本
            while not output_queue.empty():
                examples.append(output_queue.get())
            
            # 收集遊戲結果
            while not result_queue.empty():
                winner = result_queue.get()
                wins[winner] += 1
                
            # 更新進度條
            if games_played.value > old_games_played:
                pbar.update(games_played.value - old_games_played)
                old_games_played = games_played.value
                
        pbar.close()
    
    # 等待所有代理完成
    logger.info("等待所有代理進程完成...")
    while complete_count.value < args.cpus:
        # 收集剩餘的樣本和結果
        while not output_queue.empty():
            examples.append(output_queue.get())
        
        while not result_queue.empty():
            winner = result_queue.get()
            wins[winner] += 1
            
        time.sleep(0.1)
    
    # 顯示結果統計
    logger.info(f"總共生成 {len(examples)} 個訓練樣本")
    logger.info(f"勝利統計: {wins}")
    
    # 保存訓練樣本
    folder = args.checkpoint
    if not os.path.exists(folder):
        os.makedirs(folder)
        
    filename = os.path.join(folder, f"iter_{args.cur_iter:04d}.examples")
    with open(filename, "wb+") as f:
        pickle.dump(examples, f)
    
    logger.info(f"已將訓練樣本保存至 {filename}")


if __name__ == "__main__":
    # 解析命令行參數
    parser = argparse.ArgumentParser(description='自我對弈數據生成腳本')
    
    # 基本設置
    parser.add_argument('--cuda', action='store_true', default=True, help='是否使用CUDA加速')
    parser.add_argument('--board_size', type=int, default=11, help='棋盤大小')
    parser.add_argument('--cpus', type=int, default=12, help='並行進程數量')
    
    # 神經網絡設置
    parser.add_argument('--lr', type=float, default=0.001, help='學習率')
    parser.add_argument('--dropout', type=float, default=0.3, help='Dropout率')
    parser.add_argument('--epochs', type=int, default=10, help='訓練輪數')
    parser.add_argument('--batch_size', type=int, default=64, help='訓練批次大小')
    parser.add_argument('--num_channels', type=int, default=512, help='卷積層通道數')
    
    # 自我對弈設置
    parser.add_argument('--gamesPerIteration', type=int, default=100, help='每次迭代的遊戲數量')
    parser.add_argument('--nn_batch_size', type=int, default=16, help='神經網絡預測的批次大小')
    parser.add_argument('--numMCTSSims', type=int, default=800, help='MCTS模擬次數')
    parser.add_argument('--numFastSims', type=int, default=50, help='快速MCTS模擬次數')
    parser.add_argument('--probFastSim', type=float, default=0.75, help='執行快速模擬的概率')
    parser.add_argument('--tempThreshold', type=int, default=15, help='溫度閾值')
    parser.add_argument('--symmetricSamples', action='store_true', default=True, help='是否生成對稱樣本')
    
    # UCT參數
    parser.add_argument('--cpuct', type=float, default=1.0, help='PUCT算法中的c_puct參數')
    
    # 模型設置
    parser.add_argument('--load_model', action='store_true', default=False, help='是否載入預訓練模型')
    parser.add_argument('--load_folder_file', type=tuple, default=('./temp/', 'best.pth.tar'), help='模型文件路徑')
    parser.add_argument('--checkpoint', type=str, default='./temp/', help='模型保存路徑')
    parser.add_argument('--cur_iter', type=int, default=0, help='當前迭代次數')
    
    args = parser.parse_args()
    
    # 添加可調整參數
    args.expertValueWeight = Parameter(0.5)  # 專家價值權重
    
    # 設置多進程啟動方法
    mp.set_start_method('spawn')
    
    # 運行自我對弈
    self_play(args) 