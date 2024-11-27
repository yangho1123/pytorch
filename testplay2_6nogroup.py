from game_nogroup import State
from game_nogroup import random_action, maxn_action, maxn_actionnw, TEMPERATURE
#import torch
#from pv_mcts import pv_mcts_action, PV_EVALUATE_COUNT
import time
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import statistics

def calculate_points(state):
    return state.finish()

def play(next_actions):  # 傳進動作順序必為紅、綠、藍
    state = State()  # 初始化遊戲狀態
    times_per_move = []  # 儲存每一步所花的時間
    while not state.is_done():  # 循環直到遊戲結束
        current_player = state.get_player_order()  # 當下輪到哪位玩家(0、1、2)
        
        next_action = next_actions[current_player]  # 根據玩家順序選擇對應動作函式
        start_time = time.time()  # 計時開始
        action = next_action(state)  # 執行動作
        end_time = time.time()  # 計時結束
        if next_action == maxn6:
            times_per_move.append(end_time - start_time)  # 計算並儲存此步執行時間
        if action == None:
            action = 121
        state = state.next(action)  # 更新狀態
    print(state)
    print("depth:",state.depth)
    return calculate_points(state), times_per_move  # 返回得分(紅綠藍)和每步時間

if __name__ == '__main__': # 我方不等於state.mine_pieces
    game_count = 102
    all_times = defaultdict(list)
    # print("模擬次數:", PV_EVALUATE_COUNT)
    # model_path = './model/best.pt'
    # model = torch.jit.load(model_path)                               
    
    maxn3 = maxn_action(depth=3)
    maxn3nw = maxn_actionnw(depth=3)
    maxn6 = maxn_action(depth=6)
    # pv_mcts = pv_mcts_action(model, TEMPERATURE)
    win_count_p1 = 0
    win_count_red = 0
    win_count_green = 0
    win_count_blue = 0
    average_game_time = 0
    all_game_time = []
    # print("無權重+無分組")
    print("6層1vs3層2、有權重無分組測時間")
    for i in range(game_count):
        start_time = time.time()
        print("==== game:", i+1, "====")
        if i%3 == 0:
            results, times = play([maxn6,maxn3,maxn3])   # 這局紅用maxn6下法
        elif i%3 == 1:
            results, times = play([maxn3,maxn6,maxn3])   # 這局綠用maxn6下法
        elif i%3 == 2:
            results, times = play([maxn3,maxn3,maxn6])   # 這局藍用maxn6下法

        for j, time_taken in enumerate(times):
            all_times[j].append(time_taken)        
        p1_score = results[0] if i % 3 == 0 else results[1] if i % 3 == 1 else results[2]
        #p1_score = results[1]
        
        if p1_score > 0:
            win_count_p1 += 1 
            if i % 3 == 0:
                win_count_red += 1
            if i % 3 == 1:
                win_count_green += 1
            if i % 3 == 2:
                win_count_blue += 1
        print("results:", results)
        print("player score:", p1_score) 
        game_time = time.time() - start_time           
        print("一場時間{:.2f}秒".format(game_time) )
        all_game_time.append(game_time)
    # 計算每步的平均時間
    average_times_per_move = [np.mean(times) for times in all_times.values()]
    average_game_time = statistics.mean(all_game_time)
    # 繪製折線圖
    # plt.figure(figsize=(10, 5))
    # plt.plot(average_times_per_move, marker='o', label='Average Time per Move')
    # plt.xlabel('round')
    # plt.ylabel('time(s)')
    # plt.title('Maxn6 Average Time no group')
    # plt.legend()
    # plt.grid(True)
    # plt.savefig('./plots/Maxn6_no_group.png')
    print("平均一場時間：", average_game_time, "秒")
    print("player勝場數:", win_count_p1)
    print("player勝率:", (win_count_p1/game_count))
    print("Red勝率:", (win_count_red/(game_count/3)))
    print("Green勝率:", (win_count_green/(game_count/3)))
    print("Blue勝率:", (win_count_blue/(game_count/3)))
   