from game import State
from game_nogroup import maxn_action
import torch
from pv_mcts_3his_1value import pv_mcts_scores, pv_mcts_action
from dual_network_3his import DN_OUTPUT_SIZE,DN_RESIDUAL_NUM,DN_FILTERS,DN_INPUT_SHAPE
from dual_network_3his import DualNetwork
from datetime import datetime
from itertools import permutations
import  numpy as np
import pickle
import os, time

SP_GAME_COUNT = 10     # 自我對弈的局數
SP_TEMPERATURE = 1.0    #波茲曼分佈的溫度參數
# 目前是用pv-mcts vs pv-mcts vs pv-mcts，下面可以修改成pv-mcts vs maxn vs maxn

def first_player_value(ended_state):  # 計算先手的局勢價值
    if ended_state.is_lose():
        return -1 if ended_state.is_first_player() else 1
    return 0
# 印出盤面檢查用
# def save_board_to_file(state, step_number, filename="game_states.txt"):
#     with open(filename, "a") as file:
#         file.write(f"Step {step_number}:\n")
#         board_str = str(state)  # 假設 State 類有一個 __str__ 方法可以返回棋盤的字符串表示
#         file.write(board_str + "\n\n")
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

def play(model, next_actions):            # 進行一次完整對戰
    history = []
    state = State()         # 產生新的對局   
    
    while True:
        if state.is_done():
            break
        current_player = state.get_player_order()
        next_action = next_actions[current_player]  # 获取当前玩家的行动函数
        
        if "pv_mcts" in next_action.__str__():
            scores = pv_mcts_scores(model, state, SP_TEMPERATURE)  # 取得MCTS算出的概率分布
            policies = [0] * DN_OUTPUT_SIZE
            for action, policy in zip(state.all_legal_actions(), scores):
                policies[action] = policy
            action = np.random.choice(state.all_legal_actions(), p=scores)  # 根据概率分布挑选动作
        else:
            action = next_action(state)  # 取得动作
            policies = [0] * DN_OUTPUT_SIZE
            policies[action] = 1  # 转换成概率分布，才能存进history
        
        # 创建玩家通道，形状为(1, 121)
        player_channel = np.full((1, 121), current_player)
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
            player_channel.flatten()
        ]     
                
        # 将所有部分重塑为(4, 11, 11)，其中包括3个棋子状态和1个玩家通道
        state_representation = np.array(state_representation).reshape(4, 11, 11)
        history.append([state_representation, policies, None])
        state = state.next(action)  # 更新狀態       

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

def write_data(history):        # 用檔案將自我對弈產生的訓練資料存起來
    now = datetime.now()
    os.makedirs('../torchdata/', exist_ok=True)
    path = '../torchdata/{:04}{:02}{:02}{:02}{:02}{:02}.history'.format(now.year, now.month, now.day, now.hour, now.minute, now.second)
    with open(path, mode='wb') as f:
        pickle.dump(history, f)

def self_play():
    start_time = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DualNetwork(DN_INPUT_SHAPE, DN_FILTERS, DN_RESIDUAL_NUM, DN_OUTPUT_SIZE).to(device)
    history = []
    model_path = './model/1106/40layers/best.pt'
    model = torch.jit.load(model_path)

    # model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    maxn = maxn_action(depth=3)
    pv_mcts = pv_mcts_action(model, SP_TEMPERATURE)
    strategies = [maxn, maxn, maxn]  # 可以根据需要修改这里的策略组合
        
    strategy_permutations = list(permutations(strategies))
    total_games_played = 0

    while total_games_played < SP_GAME_COUNT:
        for strategy in strategy_permutations:
            h = play(model, strategy)
            history.extend(h)
            total_games_played += 1
            print(f'\rselfplay {total_games_played}/{SP_GAME_COUNT}', end='')
            
            if total_games_played >= SP_GAME_COUNT:
                break  
        
    print('')
    write_data(history)     
    #save_history_to_text_file(history)  # 保存历史记录到文本文件 
    selfplay_time = time.time() - start_time 
    print("time: ", selfplay_time, "s")

if __name__ == '__main__':
    self_play()