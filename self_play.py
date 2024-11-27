from game import State
import torch
from pv_mcts import pv_mcts_scores
from dual_network import DN_OUTPUT_SIZE,DN_RESIDUAL_NUM,DN_FILTERS,DN_INPUT_SHAPE
from dual_network import DualNetwork
from datetime import datetime
from pathlib import Path
import  numpy as np
import pickle
import os, time

SP_GAME_COUNT = 19     # 自我對弈的局數
SP_TEMPERATURE = 1.0    #波茲曼分佈的溫度參數


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

def play(model):            # 進行一次完整對戰
    history = []
    state = State() 
    step_number = 0         # 初始化步數計數器    
    # 產生新的對局
    while True:      
        if state.is_done():          
            #save_board_to_file(state, step_number)  
            break
        #save_board_to_file(state, step_number)  # 儲存當前棋盤狀態及步數
        scores = pv_mcts_scores(model, state, 1.0)       # 取得MCTS算出的機率分佈   正式要是1.0、測試要是0.0保證每次選到機率最大的步
        # print("len(actions): ", len(state.all_legal_actions()), "actions: ", state.all_legal_actions())
        policies = [0]*DN_OUTPUT_SIZE
        for action, policy in zip(state.all_legal_actions(), scores):
            policies[action]=policy
        history.append([[state.mine_pieces, state.next_pieces, state.prev_pieces], policies, None])
        action = np.random.choice(state.all_legal_actions(), p=scores)   # 根據機率分佈挑選動作        
        state = state.next(action)
        step_number += 1     # 更新步數        

 # 在訓練資料中增加價值(這場分出勝負後將結果當作價值存入history)
    values = state.finish()  # 应返回一个包含三个玩家结果的串列
    #value = first_player_value(state)
    for i in range(len(history)):
        history[i][2] = values      # 將包含三個玩家結果的串列作為訓練的標籤
        # history[i][2] = values[state.get_player_order()]  # 每次循环应考虑当前玩家顺序
        # values = [-v for v in values]  # 反转值供下一玩家使用

    # for i in range(len(history)):
    #     history[i][2] =  value
    #     value = -value
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
    model_path = './model/19layers/best.pt'
    model = torch.jit.load(model_path)

    # model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    for i in range(SP_GAME_COUNT):
        h = play(model)
        history.extend(h)
        print('\rSelfPlay {}/{}'.format(i+1, SP_GAME_COUNT), end='')
    print('')
    write_data(history)      
    selfplay_time = time.time() - start_time 
    print("time: ", selfplay_time, "s")
    

if __name__ == '__main__':
    self_play()