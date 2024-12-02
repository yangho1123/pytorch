from game import State, random_action, maxn_action
from pv_mcts_3his_1value_test import pv_mcts_action
import torch
from shutil import copy
import time
import multiprocessing

EN_GAME_COUNT = 60
EN_TEMPERATURE = 1.0  


def update_best_player():
    copy('./model/1116maxn/22layers/best_val.pt', './model/1116maxn/22layers/best.pt')
    print('Change BestPlayer')

def calculate_points(state):    
    return state.finish()

def play(next_actions):  # 进行一次完整三人对战
    state = State()  
    while not state.is_done():  
        current_player = state.get_player_order()  
        
        next_action = next_actions[current_player]  # 根據當前玩家順序選擇對應的動作函式
        action = next_action(state)  
        state = state.next(action)  
    # print(state)    
    return calculate_points(state)  # 返回得分


def evaluate_network(index, points_latest, points_best, no1_latest, no1_best):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 载入模型
    model_latest = torch.jit.load('./model/lab/1201/22layers/best.pt').to(device)
    model_best = torch.jit.load('./model/1106/40layers/best_val.pt').to(device)
    # 評估模式
    # model_latest.eval()
    # model_best.eval()

    # 建立PV MCTS选择动作的函数
    next_action_latest = pv_mcts_action(model_latest, EN_TEMPERATURE)
    next_action_best = pv_mcts_action(model_best, EN_TEMPERATURE)
    next_action_maxn = maxn_action(depth=3)
    point_latest = 0
    point_best = 0
    n1_latest = 0
    n1_best = 0
    
    for i in range(EN_GAME_COUNT):
        
        score0 = 0
        score1 = 0
        if (i//3) % 2 == 0:
            if i % 3 == 0:
                # latest 1，best 2，隨機 3
                results = play([next_action_latest, next_action_best, random_action])
                score0 = results[0]
                score1 = results[1]
                if results[0] == 1:
                    n1_latest += 1
                elif results[1] == 1:
                    n1_best += 1
            elif i % 3 == 1:
                # latest 2，隨機 1，best 3
                results = play([random_action, next_action_latest, next_action_best])
                score0 = results[1]
                score1 = results[2]
                if results[1] == 1:
                    n1_latest += 1
                elif results[2] == 1:
                    n1_best += 1
            else:
                # latest 3，best 1，隨機 2
                results = play([next_action_best, random_action, next_action_latest])
                score0 = results[2]
                score1 = results[0]
                if results[2] == 1:
                    n1_latest += 1
                elif results[0] == 1:
                    n1_best += 1
        else:
            if i % 3 == 0:
                # latest 1，best 3，隨機 2
                results = play([next_action_latest, random_action, next_action_best])
                score0 = results[0]
                score1 = results[2]
                if results[0] == 1:
                    n1_latest += 1
                elif results[2] == 1:
                    n1_best += 1
            elif i % 3 == 1:
                # latest 2，隨機 3，best 1
                results = play([next_action_best, next_action_latest, random_action])
                score0 = results[1]
                score1 = results[0]
                if results[1] == 1:
                    n1_latest += 1
                elif results[0] == 1:
                    n1_best += 1
            else:
                # latest 3，best 2，隨機 1
                results = play([random_action, next_action_best, next_action_latest])
                score0 = results[2]
                score1 = results[1]
                if results[2] == 1:
                    n1_latest += 1
                elif results[1] == 1:
                    n1_best += 1
        point_latest += score0
        point_best += score1
        print('\rEvaluate {}/{}'.format(i+1, EN_GAME_COUNT), end='')
    points_latest[index] = point_latest
    points_best[index] = point_best
    no1_latest[index] = n1_latest
    no1_best[index] = n1_best

    del model_latest
    del model_best
    torch.cuda.empty_cache()

def evaluate_network_multiprocess(num_processes, update=0):
    manager = multiprocessing.Manager()
    points_latest = manager.dict()
    points_best = manager.dict()
    no1_latest = manager.dict()
    no1_best = manager.dict()
    jobs = []
    start = time.time()
    
    for i in range(1, num_processes + 1):
        p = multiprocessing.Process(target=evaluate_network, args=(i, points_latest, points_best, no1_latest, no1_best))
        jobs.append(p)
        p.start()
        print(f"Process{i} start!")
    
    for proc in jobs:
        proc.join()
    
    end = time.time()
    total_game = EN_GAME_COUNT * num_processes
    total_point0 = sum(points_latest.values())
    total_point1 = sum(points_best.values())
    total_no1_0 = sum(no1_latest.values())
    total_no1_1 = sum(no1_best.values())
    
    print(f"score: latest({total_point0}), no.1({total_no1_0}) vs best({total_point1}), no.1({total_no1_1})")
    if total_point0 > total_point1:
        #update_best_player()
        update = 1  #是否有更新
    print(f"evaluate time: {end - start:.2f} seconds")
    
    return total_point0, total_point1, total_no1_0, total_no1_1, update
    
if __name__ == '__main__':
    num_processes = 4  # 默认进程数，可以根据需要修改
    evaluate_network_multiprocess(num_processes)