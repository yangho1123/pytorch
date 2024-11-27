import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from pathlib import Path
import numpy as np
import pickle, os
from datetime import datetime
import matplotlib.pyplot as plt

def load_data(last_n=4):    
    history_paths = sorted(Path('../torchdata/').glob('*.history'))    
    history = []    
    for path in history_paths[-last_n:]:
        with path.open(mode='rb') as f:
            batch_data = pickle.load(f)
            history.extend(batch_data)
            print("Loaded:", path)
    # 打印出第一个數據項的結構
    # if history:
    #     print("Example data structure:")
    #     example = history[0]
    #     print("Board state:", np.array(example[0]).shape)  # 输出棋盤狀態的形狀
    #     print("Policy:", np.array(example[1]).shape)       # 输出策略的形狀
    #     print("Value:", example[2])                        # 輸出價值

    print("Total games loaded:", len(history))
    return history

def prepare_data(history, steps=7):
    prepared_data = []
    # 假设每个状态都有4个通道，每个通道的形状为11x11    
    zero_state = np.zeros((5, 11, 11))
    for i in range(steps, len(history)):
        # 组装当前步骤和前7步的状态
        state_histories = []
        for j in range(steps + 1):  # steps + 1 to include the current step
            index = i - j
            if index >= 0:                
                state_histories.append(history[index][0])
            else:
                state_histories.append(zero_state)

         # 检查每个状态的形状
        # for state in state_histories:
        #     print(f"State shape: {np.array(state).shape}")  # 打印形状进行检查
        # 尝试组合状态
        try:
            combined_states = np.stack(state_histories, axis=0)
        except ValueError as e:
            print(f"Error stacking states: {e}")
            continue  # Skip this entry if error

        policy = history[i][1]
        value = history[i][2]
        prepared_data.append((combined_states, policy, value))
        
    return prepared_data

def write_prepared_data(prepared_history, directory='../torchdata/3his'):
    now = datetime.now()
    os.makedirs(directory, exist_ok=True)
    path = f"{directory}/prepared_{now.strftime('%Y%m%d%H%M%S')}.history"
    
    with open(path, 'wb') as f:
        pickle.dump(prepared_history, f)
    print(f"Prepared data saved to {path}")

if __name__=='__main__':
    history = load_data()    
    prepared_history = prepare_data(history)
    #write_prepared_data(prepared_history)
    # 印出新的history
    if prepared_history:
        print("Extended example data structure:")
        example = prepared_history[0]
        print("Board states history:", example[0].shape)  # 输出包含歷史三步的棋盤狀態
        print("Policy:", np.array(example[1]).shape)      # 输出策略的形狀
        print("Value:", example[2])                        # 輸出價值
        print("len:", len(example[0]))
        #print("player:", example[0][1])