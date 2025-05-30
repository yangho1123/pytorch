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

SP_GAME_COUNT = 200     # 自我對弈的局數
SP_TEMPERATURE = 1.0    #波茲曼分佈的溫度參數
# 目前是用pv-mcts vs pv-mcts vs pv-mcts，下面可以修改成maxn vs maxn vs maxn
FLIPTABLE = FlipTable
# 正規化FLIPTABLE到0-1範圍
FLIPTABLE_MIN = min(x for x in FLIPTABLE if x != -1)  # 忽略-1（非法位置）
FLIPTABLE_MAX = max(FLIPTABLE)
# 修改正規化邏輯，將-1轉換為0
NORMALIZED_FLIPTABLE = [(x - FLIPTABLE_MIN) / (FLIPTABLE_MAX - FLIPTABLE_MIN) if x != -1 else 0 for x in FLIPTABLE]

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
            red_pieces = state.mine_pieces.copy()
            green_pieces = state.next_pieces.copy()
            blue_pieces = state.prev_pieces.copy()
        elif state.get_player_order() == 1:
            red_pieces = state.prev_pieces.copy()
            green_pieces = state.mine_pieces.copy()
            blue_pieces = state.next_pieces.copy()
        else:
            red_pieces = state.next_pieces.copy()
            green_pieces = state.prev_pieces.copy()
            blue_pieces = state.mine_pieces.copy()
            
        # 將-1邊界值轉換為0，使所有特徵都在0~1範圍內
        red_pieces = [0 if x == -1 else x for x in red_pieces]
        green_pieces = [0 if x == -1 else x for x in green_pieces]
        blue_pieces = [0 if x == -1 else x for x in blue_pieces]
        
        # 模型的輸入狀態始终按照紅、綠、藍的順序
        state_representation = [
            red_pieces,
            green_pieces,
            blue_pieces,
            player_channel.flatten(),
            NORMALIZED_FLIPTABLE  # 使用正規化後的FLIPTABLE，FLIPTABLE中的-1也會被轉換為0
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
    values = state.finish_reward()  # 傳回一個包含三個玩家结果的串列，使用修改後的獎勵分配
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
    os.makedirs('../../torchdata/iter2', exist_ok=True)
    
    # 添加進程ID和微秒以確保檔案名稱唯一
    process_id = os.getpid()
    microsecond = now.microsecond
    
    # 新的檔案名格式: 年月日時分秒_微秒_進程ID
    history_path = '../../torchdata/iter2/{:04}{:02}{:02}{:02}{:02}{:02}_{:06d}_{}.history'.format(
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

def self_play():
    start_time = time.time()
    print("selfplay_3his_1value_test.py")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DualNetwork(DN_INPUT_SHAPE, DN_FILTERS, DN_RESIDUAL_NUM, DN_OUTPUT_SIZE).to(device)
    history = []
    game_lengths = []  # 新增列表來記錄每場比賽的回合數
    model_path = '../model/250505/22layers/best_val.pt'
    model = torch.jit.load(model_path)

    # model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    maxn = maxn_action(depth=3)
    #onlymodel = model_score(model)
    pv_mcts = pv_mcts_action(model, SP_TEMPERATURE)
    random = random_choose()
    # 使用元組來標記策略類型
    strategies = [(pv_mcts, "pv_mcts"), (pv_mcts, "pv_mcts"), (pv_mcts, "pv_mcts")]

        
    strategy_permutations = list(permutations(strategies))
    total_games_played = 0

    while total_games_played < SP_GAME_COUNT:
        for strategy in strategy_permutations:
            h = play(model, strategy)
            history.extend(h)
            game_lengths.append(len(h))  # 記錄這場比賽的回合數
            total_games_played += 1
            print(f'\rselfplay {total_games_played}/{SP_GAME_COUNT}', end='')
            
            if total_games_played >= SP_GAME_COUNT:
                break  
        
    print('')
    write_data(history, game_lengths)     
    save_history_to_text_file(history)  # 保存历史记录到文本文件 
    selfplay_time = time.time() - start_time 
    print("time: ", selfplay_time, "s")

if __name__ == '__main__':
    self_play()