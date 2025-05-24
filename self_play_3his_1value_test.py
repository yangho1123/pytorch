from game_nogroup import State
from game_nogroup import maxn_action, random_choose
from game_nogroup import FlipTable
import torch
from pv_mcts_3his_fast import pv_mcts_scores, pv_mcts_action
from dual_network_3his import DN_OUTPUT_SIZE,DN_RESIDUAL_NUM,DN_FILTERS,DN_INPUT_SHAPE
from dual_network_3his import DualNetwork
from datetime import datetime
from itertools import permutations
import  numpy as np
import pickle
import os, time
import logging
import sys
import torch.multiprocessing as mp
from tqdm import tqdm
from collections import deque
import argparse

SP_GAME_COUNT = 100     # 自我對弈的局數
SP_TEMPERATURE = 1.0    #波茲曼分佈的溫度參數
# 目前是用pv-mcts vs pv-mcts vs pv-mcts，下面可以修改成maxn vs maxn vs maxn
FLIPTABLE = FlipTable
# 正規化FLIPTABLE到0-1範圍
FLIPTABLE_MIN = min(x for x in FLIPTABLE if x != -1)  # 忽略-1（非法位置）
FLIPTABLE_MAX = max(FLIPTABLE)
NORMALIZED_FLIPTABLE = [(x - FLIPTABLE_MIN) / (FLIPTABLE_MAX - FLIPTABLE_MIN) if x != -1 else -1 for x in FLIPTABLE]

def first_player_value(ended_state):  # 計算先手的局勢價值
    if ended_state.is_lose():
        return -1 if ended_state.is_first_player() else 1
    return 0
# 印出盤面檢查用
def save_board_to_file(state, step_number, filename="game_states.txt"):
    with open(filename, "a") as file:
        file.write(f"Step {step_number}:\n")
        board_str = str(state)  # 假設 State 類有一個 __str__ 方法可以返回棋盤的字符串表示
        file.write(board_str + "\n\n")
# 印出history檢查用
def save_history_to_text_file(history, filename="game_history.txt"):
    with open(filename, "w") as file:
        for step, data in enumerate(history):
            state_repr, policies, value = data
            # 将状态矩阵、策略和值格式化为字符串
            state_str = "\n".join([' '.join(map(str, row)) for row in state_repr])
            policies_str = ' '.join(map(str, policies))
            value_str = str(value)
            # 写入文件
            file.write(f"Step {step + 1}:\n")
            file.write("State:\n" + state_str + "\n")
            file.write("Policies: " + policies_str + "\n")
            file.write("Value: " + value_str + "\n")
            file.write("-" * 40 + "\n\n")

def play(model, next_actions): 
    # 模拟棋盘状态
    class node:
        def __init__(self, state, prev_node = None, path = None):
            self.state = state
            self.parent = prev_node
            self.path = path if path is not None else []       # 歷史路徑串列
            
    filename="12步結束之後盤面(輪到紅色還沒下).txt"
    history = []
    state = State()         # 產生新的對局 
    cur_node = node(state)  # 創建初始節點
    cur_node.path.append(cur_node)  # 將初始節點添加到路徑中
    
    round = 0
    action_counts0to11 = 0
    action_counts = 0
    while True:
        
        if state.is_done():
            break
        # if round == 12:         # 印出round11結束後(第13步還沒下棋)的盤面
        #     with open(filename, "a") as file:
        #         file.write(f"Round {round+1}:\n")
        #         board_str = str(state)
        #         file.write(board_str + "\n\n")
        current_player = state.get_player_order()
        next_action, action_type = next_actions[current_player]  # 获取当前玩家的行动函数
        
        if action_type == "pv_mcts":
            scores = pv_mcts_scores(model, state, SP_TEMPERATURE)  # 取得MCTS算出的概率分布
            #print(f"\n=== 第 {round + 1} 步 (玩家 {current_player}) 完成 ===")
            policies = [0] * DN_OUTPUT_SIZE
            for action, policy in zip(state.all_legal_actions(), scores):
                policies[action] = policy
            action = np.random.choice(state.all_legal_actions(), p=scores)  # 根据概率分布挑选动作
        elif action_type == "onlymodel":
            action = next_action(state, cur_node.path)  # 使用path作為参数传递给next_action函数
            policies = [0] * DN_OUTPUT_SIZE
            policies[action] = 1
        else:            
            action = next_action(state)  # 取得动作
            
            policies = [0] * DN_OUTPUT_SIZE
            policies[action] = 1  # 转换成概率分布，才能存进history
        
        # 创建玩家通道，形状为(1, 121)，並將player值正規化到0-1範圍
        player_channel = np.full((1, 121), current_player / 2.0)  # 除以2使值範圍為0~1
        if state.get_player_order() == 0:
            red_pieces = state.mine_pieces
            green_pieces = state.next_pieces
            blue_pieces = state.prev_pieces
        elif state.get_player_order() == 1:
            red_pieces = state.prev_pieces
            green_pieces = state.mine_pieces
            blue_pieces = state.next_pieces
        else:
            red_pieces = state.next_pieces
            green_pieces = state.prev_pieces
            blue_pieces = state.mine_pieces
        # 模型的輸入狀態始终按照紅、綠、藍的順序
        state_representation = [
            red_pieces,
            green_pieces,
            blue_pieces,
            player_channel.flatten(),
            NORMALIZED_FLIPTABLE  # 使用正規化後的FLIPTABLE
        ]     
                
        # 将所有部分重塑为(5, 11, 11)，其中包括3个棋子状态和1个玩家通道和一個翻轉次數表
        state_representation = np.array(state_representation).reshape(5, 11, 11)
        history.append([state_representation, policies, None])
        
        # 更新游戏状态
        new_state = state.next(action)  # 生成新的狀態
        # 更新cur_node和path
        new_node = node(new_state, cur_node, cur_node.path.copy())  # 創建新節點，保存路徑
        new_node.path.append(new_node)  # 將新節點添加到路徑中
        cur_node = new_node  # 更新當前節點
        state = new_state  # 更新遊戲狀態
        
        round += 1       

 # 在訓練資料中增加價值(這場分出勝負後將結果當作價值存入history)
    values = state.finish()  # 傳回一個包含三個玩家结果的串列
    #value = first_player_value(state)
    for i in range(len(history)):   # i == 0 ~ 83
        #history[i][2] = values      # 將包含三個玩家結果的串列作為訓練的標籤
        if i%3 == 0:
            history[i][2] = values[0]  # 將對應的玩家的比賽結果加入到history中
        elif i%3 ==1:
            history[i][2] = values[1]  
        else:
            history[i][2] = values[2]
     
    return history

def write_data(history, game_lengths):        # 用檔案將自我對弈產生的訓練資料存起來
    now = datetime.now()
    os.makedirs('../torchdata/', exist_ok=True)
    
    # 添加進程ID和微秒以確保檔案名稱唯一
    process_id = os.getpid()
    microsecond = now.microsecond
    
    # 新的檔案名格式: 年月日時分秒_微秒_進程ID
    history_path = '../torchdata/{:04}{:02}{:02}{:02}{:02}{:02}_{:06d}_{}.history'.format(
        now.year, now.month, now.day, now.hour, now.minute, now.second, 
        microsecond, process_id
    )
    
    lengths_path = history_path.replace('.history', '_lengths.pkl')  # 生成對應的 game_lengths 檔案名稱
    
    # 保存 history
    with open(history_path, mode='wb') as f:
        pickle.dump(history, f)
    print(f"History saved to: {history_path}")
    
    # 保存 game_lengths
    with open(lengths_path, mode='wb') as f:
        pickle.dump(game_lengths, f)
    print(f"Game lengths saved to: {lengths_path}")

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
    nnet_args.num_channels = 128  # 使用128通道，與三通道輸入匹配
    
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
    # 注意：我們使用三通道輸入，對應三方玩家的棋盤狀態
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
    parser.add_argument('--num_channels', type=int, default=128, help='卷積層通道數，與三通道輸入匹配')
    
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