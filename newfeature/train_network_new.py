from dual_network_3his import DN_INPUT_SHAPE,DN_FILTERS,DN_OUTPUT_SIZE,DN_RESIDUAL_NUM
from dual_network_3his import DualNetwork
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from pathlib import Path
import numpy as np
import pickle
import time
import matplotlib.pyplot as plt
from datetime import datetime
import contextlib
import random

# 设置随机种子以确保可重复性
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

set_seed(42)  # 设置随机种子

# 訓練配置
TRAIN_STEPS = 10000   # 總訓練步數
BATCH_SIZE = 128     # 批量大小
EVAL_INTERVAL = 100  # 每100步驗證一次
SAVE_INTERVAL = 2500  # 每2500步儲存一次模型

def augment_board(board, policy):
    board_size = 11 * 11  # 假设棋盘是 11x11
    if len(policy) != board_size + 1:  # 检查策略长度是否匹配棋盘 + 特殊动作
        raise ValueError(f"Policy size {len(policy)} does not match expected size {board_size + 1}")
    # 生成所有旋转和镜像变体
    boards = []
    policies = []
    # 拆分棋盘动作和特殊动作
    board_policy = np.array(policy[:board_size])  # 棋盘动作部分
    special_action = np.array([policy[board_size]])  # 确保特殊动作是数组形式

    for i in range(4):  # 旋转0°, 90°, 180°, 270°
        rotated_board = np.rot90(board, i)
        rotated_policy = np.rot90(board_policy.reshape(11, 11), i).flatten()  # 假设策略可以被重塑和旋转
        
        # 使用 np.concatenate 来合并数组，确保维度一致
        concatenated_policy = np.concatenate((rotated_policy, special_action))
        boards.append(rotated_board)
        policies.append(concatenated_policy)
        
    return boards, policies

# def load_prepare_data(last_n=1):
#     # 获取所有 .history 文件并按文件名排序
#     history_paths = sorted(Path('../torchdata/3his').glob('*.history'))
#     # 初始化历史数据列表
#     history = []
#     # 從最新的last_n個文件中讀取資料
#     for path in history_paths[-last_n:]:
#         with path.open(mode='rb') as f:
#             batch_data = pickle.load(f)
#             for state, policy, value in batch_data:
#                 # 确保state和policy是正确的NumPy数组格式
#                 state_array = np.array(state)
#                 policy_array = np.array(policy)
#                 augmented_states, augmented_policies = augment_board(state_array, policy_array)
#                 for aug_state, aug_policy in zip(augmented_states, augmented_policies):
#                     history.append((aug_state, aug_policy, value))  # 新增加的state和相应的policy
#     return history

def load_prepare_data(last_n=1):
    # 获取所有 .history 文件并按文件名排序
    history_paths = sorted(Path('../../torchdata/3his').glob('*.history'))
    # 初始化历史数据列表
    history = []
    # 從最新的last_n個文件中讀取資料
    for path in history_paths[-last_n:]:
        with path.open(mode='rb') as f:
            history.extend(pickle.load(f))
    return history

def train_network():
    def accuracy(output, target):
        with torch.no_grad():
            pred = torch.argmax(output, dim=1)
            assert pred.shape[0] == len(target)
            correct = 0
            correct += torch.sum(pred == target).item()
        return correct / len(target)

    history = load_prepare_data()
    print("Total number of data points:", len(history))

    # 计算数据集大小
    total_size = len(history)
    train_size = int(0.8 * total_size)
    val_size = total_size - train_size
    
    # 创建数据集类
    class GameDataset(torch.utils.data.Dataset):
        def __init__(self, history_data, start_idx, end_idx):
            self.history_data = history_data
            self.start_idx = start_idx
            self.end_idx = end_idx
            self.time_steps, self.channels, self.high, self.width = DN_INPUT_SHAPE
            self.new_channels = self.time_steps * self.channels
            
        def __len__(self):
            return self.end_idx - self.start_idx
            
        def __getitem__(self, idx):
            actual_idx = self.start_idx + idx
            state, policy, value = self.history_data[actual_idx]
            
            # 转换state为tensor
            state = np.array(state)
            state = state.reshape(self.new_channels, self.high, self.width)
            state = torch.tensor(state, dtype=torch.float32)
            
            # 转换policy和value为tensor
            policy = torch.tensor(policy, dtype=torch.float32)
            value = torch.tensor(value, dtype=torch.float32)
            
            return state, policy, value
    
    # 创建训练集和验证集
    train_dataset = GameDataset(history, 0, train_size)
    val_dataset = GameDataset(history, train_size, total_size)
    
    # 批量大小(128)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # 創建無限循環的訓練數據迭代器
    def infinite_train_loader(data_loader):
        while True:
            for data in data_loader:
                yield data

    train_iterator = infinite_train_loader(train_loader)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DualNetwork(DN_INPUT_SHAPE, DN_FILTERS, DN_RESIDUAL_NUM, DN_OUTPUT_SIZE).to(device)
    model = torch.jit.load('../model/250505/iter2/best.pt').to(device)
    model.train()
    
    optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)
    # criterion_policy = nn.KLDivLoss(reduction='batchmean')
    criterion_policy = nn.CrossEntropyLoss()
    criterion_value = nn.MSELoss()
    
    # 使用每500步遞減一次學習率，而不是每個epoch
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.8)

    train_policy_losses, train_value_losses = [], []
    val_policy_losses, val_value_losses = [], []
    train_accs, val_accs = [], []
    best_val_loss = float('inf')
    step_times = []

    # 打开文件以保存损失和权重
    with open('training_log.txt', 'w') as f:
        print("開始訓練，總步數：", TRAIN_STEPS)
        #f.write(f"開始訓練，總步數：{TRAIN_STEPS}，批量大小：{BATCH_SIZE}\n")
        
        for step in range(1, TRAIN_STEPS + 1):
            step_start_time = time.time()
            model.train()
            
            # 獲取一個批次的數據
            xs_batch, y_policy_batch, y_value_batch = next(train_iterator)
            xs_batch = xs_batch.to(device)
            y_policy_batch = y_policy_batch.to(device)
            y_value_batch = y_value_batch.to(device)

            # 前向傳播
            policy_pred, value_pred = model(xs_batch)
            loss_policy = criterion_policy(policy_pred, y_policy_batch)
            loss_value = criterion_value(value_pred.squeeze(), y_value_batch)
            loss = loss_policy + loss_value
            
            # 計算準確率
            acc = accuracy(policy_pred, y_policy_batch.argmax(dim=1))
            
            # 反向傳播與優化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            # 記錄訓練損失和準確率
            train_policy_losses.append(loss_policy.item())
            train_value_losses.append(loss_value.item())
            train_accs.append(acc)
            
            step_end_time = time.time()
            step_time = step_end_time - step_start_time
            step_times.append(step_time)
            
            # 每EVAL_INTERVAL步進行一次驗證
            if step % EVAL_INTERVAL == 0:
                model.eval()
                val_policy_loss, val_value_loss = 0, 0
                val_acc = []
                with torch.no_grad():
                    for xs_batch, y_policy_batch, y_value_batch in val_loader:
                        xs_batch = xs_batch.to(device)
                        y_policy_batch = y_policy_batch.to(device)
                        y_value_batch = y_value_batch.to(device)

                        policy_pred, value_pred = model(xs_batch)
                        loss_policy = criterion_policy(policy_pred, y_policy_batch)
                        loss_value = criterion_value(value_pred.squeeze(), y_value_batch)
                        
                        acc = accuracy(policy_pred, y_policy_batch.argmax(dim=1))
                        val_acc.append(acc)

                        val_policy_loss += loss_policy.item()
                        val_value_loss += loss_value.item()

                # 計算平均驗證損失與準確率
                avg_val_policy_loss = val_policy_loss / len(val_loader)
                avg_val_value_loss = val_value_loss / len(val_loader)
                avg_val_loss = avg_val_policy_loss + avg_val_value_loss
                avg_val_acc = sum(val_acc) / len(val_acc) if val_acc else 0
                
                # 記錄驗證結果
                val_policy_losses.append(avg_val_policy_loss)
                val_value_losses.append(avg_val_value_loss)
                val_accs.append(avg_val_acc)
                
                # 計算平均訓練損失和準確率(從上一次評估到現在)
                avg_train_policy_loss = sum(train_policy_losses[-EVAL_INTERVAL:]) / EVAL_INTERVAL
                avg_train_value_loss = sum(train_value_losses[-EVAL_INTERVAL:]) / EVAL_INTERVAL
                avg_train_acc = sum(train_accs[-EVAL_INTERVAL:]) / EVAL_INTERVAL
                
                # 打印訓練狀態
                avg_step_time = sum(step_times[-EVAL_INTERVAL:]) / EVAL_INTERVAL
                output = f"Step {step}/{TRAIN_STEPS} (time: {avg_step_time:.3f}s/step), " \
                         f"Train Loss: {avg_train_policy_loss:.4f}/{avg_train_value_loss:.4f}, " \
                         f"Val Loss: {avg_val_policy_loss:.4f}/{avg_val_value_loss:.4f}, " \
                         f"Train Acc: {avg_train_acc:.4f}, Val Acc: {avg_val_acc:.4f}"
                
                print(output)
                #f.write(output + '\n')
                
                # 如果驗證損失更優，保存模型
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    model.eval()
                    script_model = torch.jit.script(model)
                    script_model.save('../model/250505/iter2/best_val.pt')
                    print(f" - Saved best model at step {step}, val_loss: {avg_val_loss:.4f}")
                    #f.write(f" - Saved best model at step {step}, val_loss: {avg_val_loss:.4f}\n")
            
            # 每SAVE_INTERVAL步保存一次階段性模型
            # if step % SAVE_INTERVAL == 0:
            #     model.eval()
            #     script_model = torch.jit.script(model)
            #     script_model.save(f'../model/250505/22layers/step_{step}.pt')
            #     print(f" - Saved checkpoint at step {step}")
                #f.write(f" - Saved checkpoint at step {step}\n")
        
        print('\n訓練完成，共計 {} 步'.format(TRAIN_STEPS))
        #f.write('\n訓練完成，共計 {} 步\n'.format(TRAIN_STEPS))
    
    # 嘗試繪製損失和準確率圖表
    try:
        current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
        
        # 損失圖表
        plt.figure(figsize=(12, 6))
        # 訓練損失曲線（每EVAL_INTERVAL點取平均）
        train_policy_avg = [sum(train_policy_losses[i:i+EVAL_INTERVAL])/EVAL_INTERVAL 
                           for i in range(0, len(train_policy_losses), EVAL_INTERVAL)]
        train_value_avg = [sum(train_value_losses[i:i+EVAL_INTERVAL])/EVAL_INTERVAL 
                          for i in range(0, len(train_value_losses), EVAL_INTERVAL)]
        
        plt.plot(range(EVAL_INTERVAL, TRAIN_STEPS + 1, EVAL_INTERVAL), train_policy_avg, label='Train Policy Loss')
        plt.plot(range(EVAL_INTERVAL, TRAIN_STEPS + 1, EVAL_INTERVAL), train_value_avg, label='Train Value Loss')
        plt.plot(range(EVAL_INTERVAL, TRAIN_STEPS + 1, EVAL_INTERVAL), val_policy_losses, label='Val Policy Loss')
        plt.plot(range(EVAL_INTERVAL, TRAIN_STEPS + 1, EVAL_INTERVAL), val_value_losses, label='Val Value Loss')
        plt.xlabel('Step')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.savefig(f'training_validation_losses_{current_time}.png')
        
        # 準確率圖表
        plt.figure(figsize=(12, 6))
        train_acc_avg = [sum(train_accs[i:i+EVAL_INTERVAL])/EVAL_INTERVAL 
                        for i in range(0, len(train_accs), EVAL_INTERVAL)]
        
        plt.plot(range(EVAL_INTERVAL, TRAIN_STEPS + 1, EVAL_INTERVAL), train_acc_avg, label='Train Accuracy')
        plt.plot(range(EVAL_INTERVAL, TRAIN_STEPS + 1, EVAL_INTERVAL), val_accs, label='Validation Accuracy')
        plt.xlabel('Step')
        plt.ylabel('Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.legend()
        plt.savefig(f'training_validation_acc_{current_time}.png')        

    except Exception as e:
        print(f"Failed to plot due to an error: {e}")

    # 保存最终模型
    try:
        model.eval()
        script_model = torch.jit.script(model)
        script_model.save('../model/250505/iter2/final.pt')
        print("Final model saved successfully.")
    except Exception as e:
        print(f"Failed to save the final model due to an error: {e}")

if __name__=='__main__':
    start_time = time.time()
    train_network()
    print("train_network() time: {:.2f} seconds".format(time.time() - start_time))