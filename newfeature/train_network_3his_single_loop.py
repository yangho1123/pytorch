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

# 保持與原始訓練相同的總步數
RN_EPOCHS = 225  # 原始的epoch數
EVAL_FREQUENCY = 1  # 評估頻率

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

    history = load_prepare_data()   # 載入歷史資料
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
    
    batch_size = 128
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # 創建無限循環訓練數據迭代器
    def infinite_train_loader(data_loader):
        while True:
            for data in data_loader:
                yield data
                
    train_iterator = infinite_train_loader(train_loader)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DualNetwork(DN_INPUT_SHAPE, DN_FILTERS, DN_RESIDUAL_NUM, DN_OUTPUT_SIZE).to(device)
    model = torch.jit.load('../model/250506/22layers/best.pt').to(device)
    model.train()
    
    optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)
    criterion_policy = nn.CrossEntropyLoss()
    criterion_value = nn.MSELoss()    
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.8)

    train_policy_losses, train_value_losses, val_policy_losses, val_value_losses = [], [], [], []
    train_accuracy, val_accuracy = [], []
    best_val_loss = float('inf')
    patience = 15
    patience_counter = 0
    
    # 計算每個epoch需要的總批次數
    steps_per_epoch = (train_size + batch_size - 1) // batch_size
    
    # 計算總訓練步數
    total_steps = RN_EPOCHS * steps_per_epoch
    
    # 保存需要進行評估的步數
    eval_steps = []
    for i in range(RN_EPOCHS):
        if (i + 1) % EVAL_FREQUENCY == 0 or i == 0 or i == RN_EPOCHS - 1:
            eval_steps.append((i + 1) * steps_per_epoch)
    
    print(f"單層循環版本開始訓練，總步數：{total_steps}，原本每個epoch {steps_per_epoch} 步")
    print(f"評估頻率：原本每 {EVAL_FREQUENCY} 個epoch評估一次")
    
    step_times = []
    total_start_time = time.time()
    
    # 初始化訓練損失累積
    accumulate_policy_loss = 0.0
    accumulate_value_loss = 0.0
    accumulate_acc = []
    
    # 改為單層循環結構
    for step in range(1, total_steps + 1):
        step_start_time = time.time()
        model.train()
        
        # 從無限迭代器獲取一個批次的數據
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
        
        # 累積當前epoch的損失和準確率
        accumulate_policy_loss += loss_policy.item()
        accumulate_value_loss += loss_value.item()
        accumulate_acc.append(acc)
        
        # 記錄步驟時間
        step_end_time = time.time()
        step_time = step_end_time - step_start_time
        step_times.append(step_time)
        
        # 檢查當前步數是否為一個epoch的結束
        current_epoch = (step - 1) // steps_per_epoch + 1
        is_epoch_end = step % steps_per_epoch == 0 or step == total_steps
        
        # 每個epoch的結束計算平均損失
        if is_epoch_end:
            # 計算當前epoch的平均損失和準確率
            epoch_steps = step % steps_per_epoch if step % steps_per_epoch != 0 else steps_per_epoch
            avg_policy_loss = accumulate_policy_loss / epoch_steps
            avg_value_loss = accumulate_value_loss / epoch_steps
            avg_train_acc = sum(accumulate_acc) / len(accumulate_acc) if accumulate_acc else 0
            
            # 記錄訓練損失
            train_policy_losses.append(avg_policy_loss)
            train_value_losses.append(avg_value_loss)
            train_accuracy.append(avg_train_acc)
            
            # 重置累積值
            accumulate_policy_loss = 0.0
            accumulate_value_loss = 0.0
            accumulate_acc = []
            
            # 更新學習率調度器（每個epoch更新一次）
            scheduler.step()
            
            # 檢查是否需要進行驗證集評估
            eval_needed = step in eval_steps
            
            if eval_needed:
                model.eval()
                val_policy_loss, val_value_loss = 0, 0
                epoch_val_acc = []
                
                with torch.no_grad():
                    for val_batch in val_loader:
                        val_xs_batch, val_y_policy_batch, val_y_value_batch = val_batch
                        val_xs_batch = val_xs_batch.to(device)
                        val_y_policy_batch = val_y_policy_batch.to(device)
                        val_y_value_batch = val_y_value_batch.to(device)
                        
                        val_policy_pred, val_value_pred = model(val_xs_batch)
                        val_loss_policy = criterion_policy(val_policy_pred, val_y_policy_batch)
                        val_loss_value = criterion_value(val_value_pred.squeeze(), val_y_value_batch)
                        
                        batch_acc = accuracy(val_policy_pred, val_y_policy_batch.argmax(dim=1))
                        epoch_val_acc.append(batch_acc)
                        
                        val_policy_loss += val_loss_policy.item()
                        val_value_loss += val_loss_value.item()
                
                # 計算平均驗證損失和準確率
                avg_val_policy_loss = val_policy_loss / len(val_loader)
                avg_val_value_loss = val_value_loss / len(val_loader)
                avg_val_acc = sum(epoch_val_acc) / len(epoch_val_acc) if epoch_val_acc else 0
                
                val_policy_losses.append(avg_val_policy_loss)
                val_value_losses.append(avg_val_value_loss)
                val_accuracy.append(avg_val_acc)
                
                val_loss = avg_val_policy_loss + avg_val_value_loss
                
                # 檢查是否為最佳模型
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    model.eval()
                    script_model = torch.jit.script(model)
                    script_model.save('../model/250506/22layers/best_val_single_loop.pt')
                    print(f" - Saved best model at epoch {current_epoch}, val_loss: {val_loss:.4f}")
                else:
                    patience_counter += 1
                    
                # 輸出訓練狀態
                output = f"Step {step}/{total_steps} (Epoch {current_epoch}/{RN_EPOCHS}) - " \
                         f"Train Loss: {avg_policy_loss:.4f}/{avg_value_loss:.4f}, " \
                         f"Val Loss: {avg_val_policy_loss:.4f}/{avg_val_value_loss:.4f}, " \
                         f"Train Acc: {avg_train_acc:.4f}, Val Acc: {avg_val_acc:.4f}"
                print(output)
            else:
                # 在非評估epoch僅輸出訓練指標
                output = f"Step {step}/{total_steps} (Epoch {current_epoch}/{RN_EPOCHS}) - " \
                         f"Train Loss: {avg_policy_loss:.4f}/{avg_value_loss:.4f}, " \
                         f"Train Acc: {avg_train_acc:.4f}"
                print(output)
                
            # 顯示每個epoch的平均步驟時間
            epoch_steps = step % steps_per_epoch if step % steps_per_epoch != 0 else steps_per_epoch
            if epoch_steps > 0:
                avg_step_time = sum(step_times[-epoch_steps:]) / epoch_steps
                print(f"Epoch {current_epoch} 平均步驟時間: {avg_step_time:.6f} 秒/步")
            
            # 檢查提前停止條件
            if patience_counter >= patience:
                print(f"\n提前停止觸發，在 {current_epoch} epoch 後停止訓練")
                break
                
        # 顯示進度（只在特定位置顯示，減少輸出）
        if step % (steps_per_epoch // 4) == 0:
            current_epoch_step = step % steps_per_epoch if step % steps_per_epoch != 0 else steps_per_epoch
            print(f"    進度: Step {step}/{total_steps} (Epoch {current_epoch}, {current_epoch_step}/{steps_per_epoch})")

    total_time = time.time() - total_start_time
    avg_step_time = sum(step_times) / len(step_times)
    print(f'\n訓練完成，耗時 {total_time:.2f} 秒, 平均步驟時間: {avg_step_time:.6f} 秒/步')

    # 繪製訓練和驗證損失
    try:
        current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
        plt.figure(figsize=(12, 6))
        plt.plot(train_policy_losses, label='Train Policy Loss')
        plt.plot(train_value_losses, label='Train Value Loss')
        
        # 正確處理驗證損失的索引（只在特定epoch有驗證）
        eval_indices = [i for i, val in enumerate(range(1, RN_EPOCHS + 1)) 
                        if val % EVAL_FREQUENCY == 0 or val == 1 or val == RN_EPOCHS]
        if len(eval_indices) == len(val_policy_losses):
            plt.plot([eval_indices[i] for i in range(len(val_policy_losses))], val_policy_losses, 'o-', label='Validation Policy Loss')
            plt.plot([eval_indices[i] for i in range(len(val_value_losses))], val_value_losses, 'o-', label='Validation Value Loss')
        
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.savefig(f'training_validation_losses_single_loop_{current_time}.png')
        
        # 繪製準確率圖表
        plt.figure(figsize=(12, 6))
        plt.plot(train_accuracy, label='Train Accuracy')
        
        if len(eval_indices) == len(val_accuracy):
            plt.plot([eval_indices[i] for i in range(len(val_accuracy))], val_accuracy, 'o-', label='Validation Accuracy')
        
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.legend()
        plt.savefig(f'training_validation_acc_single_loop_{current_time}.png')

    except Exception as e:
        print(f"繪圖失敗: {e}")

    # 保存最终模型
    try:
        model.eval()
        script_model = torch.jit.script(model)
        script_model.save('../model/250506/22layers/final_single_loop.pt')
        print("最終模型保存成功")
    except Exception as e:
        print(f"保存最終模型失敗: {e}")

if __name__=='__main__':
    start_time = time.time()
    train_network()
    print("train_network() 總時間: {:.2f} 秒".format(time.time() - start_time)) 