from game import State
from game_nogroup import random_action, maxn_action, maxn_actionnw, TEMPERATURE
import torch
from pv_mcts import pv_mcts_action, PV_EVALUATE_COUNT
import time
import statistics

def calculate_points(state):
    return state.finish()

def play(next_actions):  # 傳進動作順序必為紅、綠、藍
    state = State()  # 初始化遊戲狀態
    while not state.is_done():  # 循環直到遊戲結束
        current_player = state.get_player_order()  # 當下輪到哪位玩家(0、1、2)
        
        next_action = next_actions[current_player]  # 根據玩家順序選擇對應動作函式
        action = next_action(state)  # 執行動作
        state = state.next(action)  # 更新狀態
    print(state)
    return calculate_points(state)  # 返回得分(紅綠藍)

if __name__ == '__main__': # 我方不等於state.mine_pieces
    game_count = 300
    print("模擬次數:", PV_EVALUATE_COUNT)
    model_path = './model/lab/22layers/best.pt'              #22layers+512filter
    
    model = torch.jit.load(model_path) 
    #init_model = torch.jit.load(init_model_path)
    maxnA = maxn_action(depth=3)
    maxnB = maxn_action(depth=3)
    pv_mctsA = pv_mcts_action(model, TEMPERATURE)
    #pv_mctsB = pv_mcts_action(model, TEMPERATURE)
    random = random_action
    #pv_mcts2 = pv_mcts_action(init_model, TEMPERATURE)
    #win_count_pv_mcts = 0
    
    win_maxnA = 0
    win_maxnB = 0
    win_pvA = 0
    #win_pvB = 0
    win_count_red = 0
    win_count_green = 0
    win_count_blue = 0
    average_game_time = 0
    all_game_time = []

    for i in range(game_count):        
        start_time = time.time()
        print("==== game:", i+1, "====")    
        if (i//3)%2 == 0:
            if i%3 == 0:
                results = play([pv_mctsA,maxnA,maxnB]) 
                
                if results[0] == 1:
                    win_pvA += 1
                    win_count_red += 1
                elif results[1] == 1:
                    win_maxnA += 1
                elif results[2] == 1:
                    win_maxnB += 1
            elif i%3 == 1:
                results = play([maxnA,pv_mctsA,maxnB])
                
                if results[0] == 1:
                    win_maxnA += 1
                elif results[1] == 1:
                    win_pvA += 1
                    win_count_green += 1
                elif results[2] == 1:
                    win_maxnB += 1
            else:
                results = play([maxnA,maxnB,pv_mctsA])   
                
                if results[0] == 1:
                    win_pvA += 1
                elif results[1] == 1:
                    win_maxnB += 1
                elif results[2] == 1:
                    win_pvA += 1
                    win_count_blue += 1
        else:
            if i%3 == 0:
                results = play([pv_mctsA,maxnB,maxnA])                 
                if results[0] == 1:
                    win_pvA += 1
                    win_count_red += 1
                elif results[1] == 1:
                    win_maxnB += 1
                elif results[2] == 1:
                    win_maxnA += 1
            elif i%3 == 1:
                results = play([maxnB,pv_mctsA,maxnA])                 
                if results[0] == 1:
                    win_maxnB += 1
                elif results[1] == 1:
                    win_pvA += 1
                    win_count_green += 1
                elif results[2] == 1:
                    win_maxnA += 1  
            else:
                results = play([maxnB,maxnA,pv_mctsA])                
                if results[0] == 1:
                    win_maxnB += 1
                elif results[1] == 1:
                    win_maxnA += 1
                elif results[2] == 1:
                    win_pvA += 1
                    win_count_blue += 1

        #pv_mcts_score = results[0] if i % 3 == 0 else results[1] if i % 3 == 1 else results[2]
        
        # if pv_mcts_score > 0:
        #     win_count_pv_mcts += 1         
        #     if i % 3 == 0:
        #         win_count_red += 1
        #     if i % 3 == 1:
        #         win_count_green += 1
        #     if i % 3 == 2:
        #         win_count_blue += 1  
        
        #print("results:", results)
        #print("maxn score:", maxn_score) 
        game_time = time.time() - start_time           
        print("一場時間{:.2f}秒".format(game_time))
        all_game_time.append(game_time)
        # time.sleep(2)
    average_game_time = statistics.mean(all_game_time)
    print("平均一場時間：", average_game_time, "秒")
    print("pvA勝場數:", win_pvA)
    print("pvA勝率:", (win_pvA/game_count))
    print("maxnA勝場數:", win_maxnA)
    print("maxnA勝率:", (win_maxnA/game_count))
    print("maxnB勝場數:", win_maxnB)
    print("maxnB勝率:", (win_maxnB/game_count))
    print("model in Red勝率:", (win_count_red/(game_count/3)))
    print("model in Green勝率:", (win_count_green/(game_count/3)))
    print("model in Blue勝率:", (win_count_blue/(game_count/3)))
   