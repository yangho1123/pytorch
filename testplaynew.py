from game_nogroup import State
from game_nogroup import maxn_action
from game import random_action
import torch
from pv_mcts_3his_1value_test import pv_mcts_action, PV_EVALUATE_COUNT
import time
import statistics
import numpy as np

TEMPERATURE = 0

def calculate_points(state):
    return state.finish()

def play(next_actions):  # 傳進動作順序必為紅、綠、藍
    state = State()  # 初始化遊戲狀態
    while not state.is_done():  # 循環直到遊戲結束
        current_player = state.get_player_order()  # 當下輪到哪位玩家(0、1、2)
        
        next_action = next_actions[current_player]  # 根據玩家順序選擇對應動作函式
        
        action = next_action(state)  # 執行動作
        state = state.next(action)  # 更新狀態
    mine_pieces = state.piece_count(state.mine_pieces)  
    next_pieces = state.piece_count(state.next_pieces)  
    prev_pieces = state.piece_count(state.prev_pieces)  
    if state.get_player_order() == 0: # red
        red_pieces = mine_pieces
        green_pieces = next_pieces
        blue_pieces = prev_pieces
    elif state.get_player_order() == 1: # green
        red_pieces = prev_pieces
        green_pieces = mine_pieces
        blue_pieces = next_pieces
    else:   # blue
        red_pieces = next_pieces
        green_pieces = prev_pieces
        blue_pieces = mine_pieces
    print(state)
    print(f"Red: {red_pieces}, Green: {green_pieces}, Blue: {blue_pieces}")
    return calculate_points(state)  # 返回得分(紅綠藍)

if __name__ == '__main__': # 我方不等於state.mine_pieces
    game_count = 25
    print("模擬次數:", PV_EVALUATE_COUNT)
    model_path = './model/1217/22layers/lab/best.pt'              #40layers+512filters
    
    model = torch.jit.load(model_path) 
    #init_model = torch.jit.load(init_model_path)
    maxnA = maxn_action(depth=3)
    #maxnB = maxn_action(depth=3)
    pv_mctsA = pv_mcts_action(model, TEMPERATURE)
    #pv_mctsB = pv_mcts_action(model, TEMPERATURE)
    random = random_action
    #pv_mcts2 = pv_mcts_action(init_model, TEMPERATURE)
    #win_count_pv_mcts = 0
    # possible_players = [[maxnA, pv_mctsA, random], [maxnA, random, pv_mctsA], 
    #                     [pv_mctsA, maxnA, random], [random, maxnA, pv_mctsA],
    #                     [pv_mctsA, random, maxnA], [random, pv_mctsA, maxnA]]
    possible_players = [[random, pv_mctsA, random], [random, random, pv_mctsA], 
                        [pv_mctsA, random, random], [random, random, pv_mctsA],
                        [pv_mctsA, random, random], [random, pv_mctsA, random]]
    win_maxn = 0
    win_maxn_colors = [0, 0, 0]
    win_random = 0
    win_random_colors = [0, 0, 0]
    win_pvA = 0
    win_pvA_colors = [0, 0, 0]
    
    average_game_time = 0
    all_game_time = []
    num = 1
    for i in range(game_count):      
        print(f"Iteration:{i+1}")  
        for players in possible_players:
            print("==== game num:", num, "====")    
            num += 1
            start_time = time.time()
            results = play(players)
            game_time = time.time() - start_time
             # 計算勝利玩家
            winner_index = np.argmax(np.array(results))
            winner = players[winner_index]

            # 根據函數對應顯示策略名稱
            player_strategies = {
                maxnA: "MaxN",
                pv_mctsA: "PV_MCTS",
                random: "Random"
            }
            strategies_used = [player_strategies[player] for player in players]
             
            # 統計勝利次數
            if winner == maxnA:
                win_maxn += 1
                win_maxn_colors[winner_index] += 1   
            elif winner == pv_mctsA:
                win_pvA += 1
                win_pvA_colors[winner_index] += 1   
            else:
                win_random += 1
                win_random_colors[winner_index] += 1   
                  
            print(f"Results: {results}")
            print(f"Red Player: {strategies_used[0]}, Green Player: {strategies_used[1]}, Blue Player: {strategies_used[2]}")
            print(f"一場時間: {game_time:.2f} seconds")
            all_game_time.append(game_time)
    average_game_time = statistics.mean(all_game_time)
    print("平均一場時間：", average_game_time, "秒")
    print("model勝場數:", win_pvA)
    print("model勝率:", (win_pvA/(game_count*6)))
    # print("maxn勝場數:", win_maxn)
    # print("maxn勝率:", (win_maxn/(game_count*6)))
    print("random勝場數:", win_random)
    print("random勝率:", (win_random/(game_count*6)))
    print("model in Red勝率:", (win_pvA_colors[0]/(game_count*2)))
    print("model in Green勝率:", (win_pvA_colors[1]/(game_count*2)))
    print("model in Blue勝率:", (win_pvA_colors[2]/(game_count*2)))
    #print("maxn in Red勝率:", (win_maxn_colors[0]/(game_count*2)))
    #print("maxn in Green勝率:", (win_maxn_colors[1]/(game_count*2)))
    #print("maxn in Blue勝率:", (win_maxn_colors[2]/(game_count*2)))
    print("random in Red勝率:", (win_random_colors[0]/(game_count*2)))
    print("random in Green勝率:", (win_random_colors[1]/(game_count*2)))
    print("random in Blue勝率:", (win_random_colors[2]/(game_count*2)))
   