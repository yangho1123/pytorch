from pathlib import Path
import numpy as np
import pickle, os
from datetime import datetime
from collections import defaultdict



def load_data(last_n=5):    
    history_paths = sorted(Path('../torchdata/').glob('*.history'))    
    history = []    
    for path in history_paths[-last_n:]:
        with path.open(mode='rb') as f:
            batch_data = pickle.load(f)
            history.extend(batch_data)
            print("Loaded:", path)
    # 打印出第一个數據項的結構
    if history:
        print("Example data structure:")
        example = history[0]
        print("Board state:", np.array(example[0]).shape)  # 输出棋盤狀態的形狀
        print("Policy:", np.array(example[1]).shape)       # 输出策略的形狀
        print("Value:", example[2])                        # 輸出價值

    print("Total games loaded:", len(history))
    return history

def load(data_path='../torchdata/'):
    last_n = 1
    history_paths = sorted(Path(data_path).glob('*.history'))
    histories = []
    game_lengths = []

    for history_path in history_paths[-last_n:]:
        lengths_path = history_path.with_name(history_path.stem + '_lengths.pkl')
        
        with history_path.open(mode='rb') as f:
            batch_history = pickle.load(f)
            histories.extend(batch_history)
        
        with lengths_path.open(mode='rb') as f:
            batch_lengths = pickle.load(f)
            game_lengths.extend(batch_lengths)

        print(f"Loaded: {history_path}")

    print(f"Total records loaded: {len(histories)}")
    print(f"Total games loaded: {len(game_lengths)}")

    # 根據回合數切分 histories
    grouped_histories = []
    start = 0
    for length in game_lengths:
        grouped_histories.append(histories[start:start + length])
        start += length

    return grouped_histories

def extract_12th_round_state(history):
    """
    從單場比賽的 history 中提取第 12 回合的 state_representation。
    回合數從 0 開始計算，第 12 回合即為索引 11。
    """
    
    if len(history) > 11:
        return history[11][0]  # history[11] 是第 12 回合的資料，索引 0 是 state_representation
    return None

def compare_12th_round_states(histories):
    """
    比較多場比賽中第 12 回合的 state_representation，找出相同的盤面。
    histories: 包含多場比賽的 history 的列表。
    """
    state_dict = defaultdict(list)
    
    for i, history in enumerate(histories):
        state = extract_12th_round_state(history)       # 傳回某一場的第12回合盤面
        if state is not None:
            # 將 state_representation 轉換為元組形式，使其可哈希並用於字典的鍵
            state_tuple = tuple(state.flatten())  # 將多維陣列展平成一維，再轉成元組
            state_dict[state_tuple].append(f"Game {i + 1}")

    # 找出重複的盤面
    duplicates = {state: games for state, games in state_dict.items() if len(games) > 1}

    return duplicates

if __name__=='__main__':
    histories = load()
    # 比較第 12 回合的盤面
    duplicates = compare_12th_round_states(histories)
    # 輸出結果
    if duplicates:
        for state, games in duplicates.items():
            print(f"第 12 回合相同的盤面出現在: {', '.join(games)}")
    else:
        print("沒有發現相同的第 12 回合盤面。")
        
    
        