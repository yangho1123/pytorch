from game import State, random_action, alpha_beta_action, mcts_action
from pytorch.pv_mcts_old import pv_mcts_action
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
from pathlib import Path
import numpy as np

EP_GAME_COUNT = 10

def first_player_point(ended_state):
    if ended_state.is_lose():
        return 0 if ended_state.is_first_player() else 1
    return 0.5

def play(next_actions):
    state = State()    
    while True:
        
        if state.is_done():
            break
        #取得先手玩家與後手玩家的動作
        next_action = next_actions[0] if state.is_first_player() else next_actions[1]
        
        action = next_action(state)  #alpha_beta_action(state) 卡在這行
        
        state = state.next(action)    #取得下一個狀態
        
    return first_player_point(state)  #傳回先手分數

def evaluate_algorithm_of(label, next_actions):
    total_point = 0
    for i in range(EP_GAME_COUNT):
        if i%2 == 0:
            total_point += play(next_actions)
        else:
            total_point += 1 - play(list(reversed(next_actions)))

        #輸出結果
        print('\rEvaluate {}/{}'.format(i+1, EP_GAME_COUNT), end='')
    print('')

    average_point = total_point / EP_GAME_COUNT
    print(label,  average_point)    # 對戰的演算法名稱及AlphaZero與其對戰的勝率

def evaluate_best_player():
    model = load_model('./model/best.h5')

    next_pv_mcts_action = pv_mcts_action(model, 0.0)    # 最佳玩家的動作    

   #最佳玩家VS蒙地卡羅
    next_actions = (next_pv_mcts_action, mcts_action)
    evaluate_algorithm_of('BestModel VS MCTS', next_actions)

   #最佳玩家VSAlpha-beta剪枝
    #next_actions = (next_pv_mcts_action, alpha_beta_action)
    #evaluate_algorithm_of('BestModel VS AlphaBeta', next_actions)

    #最佳玩家VS隨機下法
    next_actions = (next_pv_mcts_action, random_action)
    evaluate_algorithm_of('BestModel VS Random', next_actions)  

    #清除session與刪除模型
    K.clear_session()
    del model
    
if __name__ == '__main__':
    print('start')
    evaluate_best_player()
    print('end')