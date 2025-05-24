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

RN_EPOCHS = 225     # 訓練次數
EVAL_FREQUENCY = 1  # 每5個epoch評估一次驗證集，降低評估頻率

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
    
    # 創建無限循環訓練數據迭代器，避免每個epoch重新初始化資料載入器
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
    
    print(f"開始訓練，總epoch數：{RN_EPOCHS}，每個epoch包含{steps_per_epoch}個批次")
    print(f"評估頻率：每{EVAL_FREQUENCY}個epoch")
    
    epoch_times = []
    total_start_time = time.time()
    
    for epoch in range(RN_EPOCHS):
        epoch_start_time = time.time()
        model.train()
        total_policy_loss, total_value_loss = 0, 0
        epoch_train_acc = []
        
        # 使用計數器來跟蹤當前epoch的步數
        for step in range(steps_per_epoch):
            # 從無限迭代器獲取一個批次的數據
            xs_batch, y_policy_batch, y_value_batch = next(train_iterator)
            xs_batch = xs_batch.to(device)
            y_policy_batch = y_policy_batch.to(device)
            y_value_batch = y_value_batch.to(device)

            policy_pred, value_pred = model(xs_batch)
            loss_policy = criterion_policy(policy_pred, y_policy_batch)
            loss_value = criterion_value(value_pred.squeeze(), y_value_batch)
            loss = loss_policy + loss_value
            
            acc = accuracy(policy_pred, y_policy_batch.argmax(dim=1))
            epoch_train_acc.append(acc)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_policy_loss += loss_policy.item()
            total_value_loss += loss_value.item()
            
            # 只在每個epoch的1/4和3/4處顯示進度，減少輸出
            if step in [steps_per_epoch // 4, steps_per_epoch * 3 // 4]:
                print(f"Epoch {epoch+1}, Step {step+1}/{steps_per_epoch}, Loss: {loss.item():.4f}")

        # 計算並記錄當前epoch的平均損失和準確率
        avg_policy_loss = total_policy_loss / steps_per_epoch
        avg_value_loss = total_value_loss / steps_per_epoch
        avg_train_acc = sum(epoch_train_acc) / len(epoch_train_acc) if epoch_train_acc else 0
        
        train_policy_losses.append(avg_policy_loss)
        train_value_losses.append(avg_value_loss)
        train_accuracy.append(avg_train_acc)
        
        # 更新學習率
        scheduler.step()
        
        # 僅在指定頻率評估驗證集，降低評估開銷
        if (epoch + 1) % EVAL_FREQUENCY == 0 or epoch == 0 or epoch == RN_EPOCHS - 1:
            model.eval()
            val_policy_loss, val_value_loss = 0, 0
            epoch_val_acc = []
            
            with torch.no_grad():
                for xs_batch, y_policy_batch, y_value_batch in val_loader:
                    xs_batch = xs_batch.to(device)
                    y_policy_batch = y_policy_batch.to(device)
                    y_value_batch = y_value_batch.to(device)

                    policy_pred, value_pred = model(xs_batch)
                    loss_policy = criterion_policy(policy_pred, y_policy_batch)
                    loss_value = criterion_value(value_pred.squeeze(), y_value_batch)
                    
                    acc = accuracy(policy_pred, y_policy_batch.argmax(dim=1))
                    epoch_val_acc.append(acc)

                    val_policy_loss += loss_policy.item()
                    val_value_loss += loss_value.item()

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
                script_model.save('../model/250506/22layers/best_val.pt')
                print(f" - Saved best model at epoch {epoch+1}, val_loss: {val_loss:.4f}")
            else:
                patience_counter += 1
                
            # 輸出訓練狀態
            output = f"Epoch {epoch+1}/{RN_EPOCHS} - " \
                     f"Train Loss: {avg_policy_loss:.4f}/{avg_value_loss:.4f}, " \
                     f"Val Loss: {avg_val_policy_loss:.4f}/{avg_val_value_loss:.4f}, " \
                     f"Train Acc: {avg_train_acc:.4f}, Val Acc: {avg_val_acc:.4f}"
            print(output)
        else:
            # 在非評估epoch僅輸出訓練指標
            output = f"Epoch {epoch+1}/{RN_EPOCHS} - " \
                     f"Train Loss: {avg_policy_loss:.4f}/{avg_value_loss:.4f}, " \
                     f"Train Acc: {avg_train_acc:.4f}"
            print(output)
            
        epoch_end_time = time.time()
        epoch_time = epoch_end_time - epoch_start_time
        epoch_times.append(epoch_time)
        print(f"Epoch {epoch+1} completed in {epoch_time:.2f} seconds")
        
        # 檢查提前停止條件
        if patience_counter >= patience:
            print(f"\nEarly stopping triggered after {epoch+1} epochs")
            break

    total_time = time.time() - total_start_time
    print(f'\nTraining completed in {total_time:.2f} seconds.')
    print(f'Average epoch time: {sum(epoch_times)/len(epoch_times):.2f} seconds')

    # 繪製訓練和驗證損失
    try:
        current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
        plt.figure(figsize=(12, 6))
        plt.plot(train_policy_losses, label='Train Policy Loss')
        plt.plot(train_value_losses, label='Train Value Loss')
        x_val = list(range(0, len(train_policy_losses), EVAL_FREQUENCY))
        plt.plot(x_val, val_policy_losses, 'o-', label='Validation Policy Loss')
        plt.plot(x_val, val_value_losses, 'o-', label='Validation Value Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.savefig(f'training_validation_losses_{current_time}.png')
        
        # 繪製準確率圖表
        plt.figure(figsize=(12, 6))
        plt.plot(train_accuracy, label='Train Accuracy')
        plt.plot(x_val, val_accuracy, 'o-', label='Validation Accuracy')
        plt.xlabel('Epoch')
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
        script_model.save('../model/250506/22layers/final.pt')
        print("Final model saved successfully.")
    except Exception as e:
        print(f"Failed to save the final model due to an error: {e}")

if __name__=='__main__':
    start_time = time.time()
    train_network()
    print("train_network() time: {:.2f} seconds".format(time.time() - start_time)) 