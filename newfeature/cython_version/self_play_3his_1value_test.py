from game_nogroup import State
from game_nogroup import maxn_action, random_choose
from game_nogroup import FlipTable
import torch
from pv_mcts_3his_fast_cy import pv_mcts_scores, pv_mcts_action  # 導入 Cython 版本
from dual_network_3his import DN_OUTPUT_SIZE,DN_RESIDUAL_NUM,DN_FILTERS,DN_INPUT_SHAPE
from dual_network_3his import DualNetwork
from datetime import datetime
from itertools import permutations
import numpy as np
import pickle
import os, time
import sys
import traceback

SP_GAME_COUNT = 100     # 自我對弈的局數
SP_TEMPERATURE = 1.0    # 波茲曼分佈的溫度參數
# 目前是用 pv-mcts vs pv-mcts vs pv-mcts，下面可以修改成 maxn vs maxn vs maxn
FLIPTABLE = FlipTable
# 正規化 FLIPTABLE 到 0-1 範圍
FLIPTABLE_MIN = min(x for x in FLIPTABLE if x != -1)  # 忽略 -1（非法位置）
FLIPTABLE_MAX = max(FLIPTABLE)
# 修改正規化邏輯，將 -1 轉換為 0
NORMALIZED_FLIPTABLE = [(x - FLIPTABLE_MIN) / (FLIPTABLE_MAX - FLIPTABLE_MIN) if x != -1 else 0 for x in FLIPTABLE]

# 添加上級目錄到路徑中，以便導入原始Python版本的MCTS
sys.path.append('..')
try:
    # 嘗試導入Python版本的MCTS作為備用
    from pv_mcts_3his_fast import pv_mcts_scores as py_pv_mcts_scores
    from pv_mcts_3his_fast import pv_mcts_action as py_pv_mcts_action
    HAVE_PY_VERSION = True
    print("已成功導入Python版本的MCTS作為備用")
except ImportError:
    HAVE_PY_VERSION = False
    print("未能導入Python版本的MCTS")

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

# 印出 history 檢查用
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
    print("開始play函數...")
    class node:
        def __init__(self, state, prev_node = None, path = None):
            self.state = state
            self.parent = prev_node
            self.path = path if path is not None else []  # 歷史路徑串列
            
    history = []
    state = State()  # 產生新的對局 
    cur_node = node(state)  # 創建初始節點
    cur_node.path.append(cur_node)  # 將初始節點添加到路徑中
    
    round = 0
    action_counts0to11 = 0
    action_counts = 0
    # 是否使用Python版本的MCTS而非Cython版本
    use_py_version = False
    
    while True:
        print(f"回合 {round}...")
        if state.is_done():
            print("遊戲結束")
            break
            
        current_player = state.get_player_order()
        next_action, action_type = next_actions[current_player]  # 获取当前玩家的行动函数
        print(f"玩家 {current_player} 使用策略: {action_type}")
        
        if action_type == "pv_mcts":
            try:
                if not use_py_version:
                    print("調用Cython版本pv_mcts_scores...")
                    # 嘗試使用Cython版本，設置超時為10秒
                    import signal
                    
                    def timeout_handler(signum, frame):
                        raise TimeoutError("Cython版本的pv_mcts_scores執行超時")
                    
                    # 在Windows上不支持設置signal超時，所以直接調用
                    try:
                        scores = pv_mcts_scores(model, state, SP_TEMPERATURE)  # 取得 MCTS 算出的概率分布
                        print("Cython版pv_mcts_scores完成，生成概率分布")
                    except Exception as e:
                        print(f"Cython版本失敗，錯誤: {e}，嘗試使用Python版本")
                        if HAVE_PY_VERSION:
                            use_py_version = True
                            scores = py_pv_mcts_scores(model, state, SP_TEMPERATURE)
                            print("Python版pv_mcts_scores完成")
                        else:
                            # 如果沒有Python版本，使用隨機策略
                            legal_actions = state.all_legal_actions()
                            scores = np.ones(len(legal_actions)) / len(legal_actions)
                            print("使用隨機策略")
                else:
                    # 直接使用Python版本
                    print("調用Python版本pv_mcts_scores...")
                    scores = py_pv_mcts_scores(model, state, SP_TEMPERATURE)
                    print("Python版pv_mcts_scores完成")
                    
                policies = [0] * DN_OUTPUT_SIZE
                for action, policy in zip(state.all_legal_actions(), scores):
                    policies[action] = policy
                print("從概率分布中選擇動作...")
                action = np.random.choice(state.all_legal_actions(), p=scores)  # 根据概率分布挑选动作
                print(f"選擇動作: {action}")
            except Exception as e:
                print(f"pv_mcts_scores或動作選擇過程中出錯: {e}")
                traceback.print_exc()
                # 發生錯誤時使用隨機策略
                legal_actions = state.all_legal_actions()
                scores = np.ones(len(legal_actions)) / len(legal_actions)
                policies = [0] * DN_OUTPUT_SIZE
                for i, action in enumerate(legal_actions):
                    policies[action] = scores[i]
                action = np.random.choice(legal_actions)
                print(f"使用隨機策略選擇動作: {action}")
        elif action_type == "onlymodel":
            action = next_action(state, cur_node.path)  # 使用 path 作為参数传递给 next_action 函数
            policies = [0] * DN_OUTPUT_SIZE
            policies[action] = 1
        else:            
            action = next_action(state)  # 取得动作
            policies = [0] * DN_OUTPUT_SIZE
            policies[action] = 1  # 转换成概率分布，才能存进 history
        
        # 创建玩家通道，形状为(1, 121)，並將 player 值正規化到 0-1 範圍
        player_channel = np.full((1, 121), current_player / 2.0)  # 除以 2 使值範圍為 0~1
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
            
        # 將 -1 邊界值轉換為 0，使所有特徵都在 0~1 範圍內
        red_pieces = [0 if x == -1 else x for x in red_pieces]
        green_pieces = [0 if x == -1 else x for x in green_pieces]
        blue_pieces = [0 if x == -1 else x for x in blue_pieces]
        
        # 模型的輸入狀態始终按照紅、綠、藍的順序
        state_representation = [
            red_pieces,
            green_pieces,
            blue_pieces,
            player_channel.flatten(),
            NORMALIZED_FLIPTABLE  # 使用正規化後的 FLIPTABLE，FLIPTABLE 中的 -1 也會被轉換為 0
        ]     
                
        # 将所有部分重塑为(5, 11, 11)，其中包括 3 个棋子状态和 1 个玩家通道和一個翻轉次數表
        state_representation = np.array(state_representation).reshape(5, 11, 11)
        history.append([state_representation, policies, None])
        
        # 更新游戏状态
        new_state = state.next(action)  # 生成新的狀態
        # 更新 cur_node 和 path
        new_node = node(new_state, cur_node, cur_node.path.copy())  # 創建新節點，保存路徑
        new_node.path.append(new_node)  # 將新節點添加到路徑中
        cur_node = new_node  # 更新當前節點
        state = new_state  # 更新遊戲狀態
        
        round += 1       

    # 在訓練資料中增加價值(這場分出勝負後將結果當作價值存入 history)
    values = state.finish_reward()  # 傳回一個包含三個玩家结果的串列，使用修改後的獎勵分配
    for i in range(len(history)):
        if i % 3 == 0:
            history[i][2] = values[0]  # 將對應的玩家的比賽結果加入到 history 中
        elif i % 3 == 1:
            history[i][2] = values[1]  
        else:
            history[i][2] = values[2]
     
    return history

def write_data(history, game_lengths):  # 用檔案將自我對弈產生的訓練資料存起來
    now = datetime.now()
    os.makedirs('../../torchdata', exist_ok=True)
    
    # 添加進程 ID 和微秒以確保檔案名稱唯一
    process_id = os.getpid()
    microsecond = now.microsecond
    
    # 新的檔案名格式: 年月日時分秒_微秒_進程ID
    history_path = '../../torchdata/{:04}{:02}{:02}{:02}{:02}{:02}_{:06d}_{}.history'.format(
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
    print("self_play_3his_1value_test_cy.py - Cython accelerated version")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用設備: {device}")
    
    history = []
    game_lengths = []  # 新增列表來記錄每場比賽的回合數
    model_path = '../../../model/250505/22layers/best_val.pt'
    
    print(f"載入模型: {model_path}")
    try:
        model = torch.jit.load(model_path)
        print("模型載入成功")
        model.eval()
        print("模型設為評估模式")
    except Exception as e:
        print(f"模型載入錯誤: {e}")
        # 嘗試後備路徑
        try:
            backup_path = '../../model/250505/22layers/best_val.pt'
            print(f"嘗試載入備用路徑模型: {backup_path}")
            model = torch.jit.load(backup_path)
            print("備用路徑模型載入成功")
            model.eval()
        except Exception as e2:
            print(f"備用路徑模型載入錯誤: {e2}")
            return
    
    try:
        print("初始化策略...")
        maxn = maxn_action(depth=3)
        pv_mcts = pv_mcts_action(model, SP_TEMPERATURE)
        random_action = random_choose()
        print("策略初始化完成")
        
        # 使用元組來標記策略類型
        strategies = [(pv_mcts, "pv_mcts"), (pv_mcts, "pv_mcts"), (pv_mcts, "pv_mcts")]
        
        strategy_permutations = list(permutations(strategies))
        total_games_played = 0

        print(f"開始自我對弈，目標場數: {SP_GAME_COUNT}")
        while total_games_played < SP_GAME_COUNT:
            for strategy in strategy_permutations:
                print(f"開始第 {total_games_played + 1} 場遊戲...")
                game_start_time = time.time()
                try:
                    h = play(model, strategy)
                    print(f"第 {total_games_played + 1} 場遊戲完成，回合數: {len(h)}")
                    history.extend(h)
                    game_lengths.append(len(h))  # 記錄這場比賽的回合數
                    total_games_played += 1
                    
                    game_time = time.time() - game_start_time
                    print(f'\r自我對弈進度: {total_games_played}/{SP_GAME_COUNT}, 本局耗時: {game_time:.2f}秒, 平均每步: {game_time/len(h):.4f}秒')
                except Exception as e:
                    print(f"遊戲進行中出錯: {e}")
                
                if total_games_played >= SP_GAME_COUNT:
                    break
            
        print('\n自我對弈完成')
        write_data(history, game_lengths)
        
        total_time = time.time() - start_time 
        avg_game_time = total_time / total_games_played if total_games_played > 0 else 0
        avg_step_time = total_time / sum(game_lengths) if game_lengths else 0
        
        print(f"總耗時: {total_time:.2f}秒")
        print(f"平均每局耗時: {avg_game_time:.2f}秒")
        print(f"平均每步耗時: {avg_step_time:.4f}秒")
        print(f"資料數量: {len(history)}")
        print(f"平均每局步數: {sum(game_lengths)/len(game_lengths):.2f}" if game_lengths else "沒有完成任何遊戲")
    except Exception as e:
        print(f"自我對弈過程中出錯: {e}")

if __name__ == '__main__':
    self_play() 