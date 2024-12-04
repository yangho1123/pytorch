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

RN_EPOCHS = 5     # 訓練次數

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
    history_paths = sorted(Path('../torchdata/3his').glob('*.history'))
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

    history = load_prepare_data()   # 載入重塑好的歷史資料
    print("Total number of data points:", len(history))
    xs, y_policies, y_values = zip(*history)
    
    # # 在尝试转换之前，先检查数据
    # print("Example of raw state data (xs[0]):", xs[0])
    # print("Type of xs[0]:", type(xs[0]))
    # print("Example shape (if applicable):", np.array(xs[0]).shape if isinstance(xs[0], (list, np.ndarray)) else "N/A")
    
    time_steps, channels, high, width = DN_INPUT_SHAPE
    new_channels = time_steps * channels
    xs = np.array(xs)
    xs = xs.reshape(len(xs), new_channels, high, width)    
    xs = torch.tensor(np.array(xs), dtype=torch.float32)
    y_policies = torch.tensor(np.array(y_policies), dtype=torch.float32)
    y_values = torch.tensor(np.array(y_values), dtype=torch.float32)
    dataset = TensorDataset(xs, y_policies, y_values)
    
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DualNetwork(DN_INPUT_SHAPE, DN_FILTERS, DN_RESIDUAL_NUM, DN_OUTPUT_SIZE).to(device)
    model = torch.jit.load('./model/1201/22layers/best.pt').to(device)
    model.train()
    
    optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)
    # criterion_policy = nn.KLDivLoss(reduction='batchmean')
    criterion_policy = nn.CrossEntropyLoss()
    #criterion_policy = nn.NLLLoss()
    criterion_value = nn.MSELoss()    
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.8)

    train_policy_losses, train_value_losses, val_policy_losses, val_value_losses = [], [], [], []
    best_val_loss = float('inf')
    patience = 15
    patience_counter = 0
    epoch_train_accs = []                           # for accuracy
    epoch_val_accs = []                             # for accuracy
    # 打开文件以保存损失和权重
    with open('training_log.txt', 'w') as f:
        for epoch in range(RN_EPOCHS):
            model.train()
            total_policy_loss, total_value_loss = 0, 0
            epoch_train_acc = []       # for accuracy
            epoch_val_acc = []         # for accuracy
            for batch_idx, (xs_batch, y_policy_batch, y_value_batch) in enumerate(train_loader):
                xs_batch = xs_batch.to(device)
                y_policy_batch = y_policy_batch.to(device)
                y_value_batch = y_value_batch.to(device)

                policy_pred, value_pred = model(xs_batch)
                #log_probs = torch.log(policy_pred + 1e-9)
                loss_policy = criterion_policy(policy_pred, y_policy_batch)
                loss_value = criterion_value(value_pred.squeeze(), y_value_batch)
                loss = loss_policy + loss_value
                
                acc = accuracy(policy_pred, y_policy_batch.argmax(dim=1))   # for accuracy
                epoch_train_acc.append(acc)                                # for accuracy
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_policy_loss += loss_policy.item()
                total_value_loss += loss_value.item()
                if batch_idx % 100 == 0:
                    output = f"Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item():.4f}"
                    print(output)  # 打印到终端
                    f.write(output + '\n')  # 写入文件

            train_policy_losses.append(total_policy_loss / len(train_loader))
            train_value_losses.append(total_value_loss / len(train_loader))
            
            epoch_train_acc = sum(epoch_train_acc) / len(epoch_train_acc) if epoch_train_acc else 0
            epoch_train_accs.append(epoch_train_acc)                                             # 记录每个epoch的平均准确率

            model.eval()
            val_policy_loss, val_value_loss = 0, 0
            with torch.no_grad():
                for xs_batch, y_policy_batch, y_value_batch in val_loader:
                    xs_batch = xs_batch.to(device)
                    y_policy_batch = y_policy_batch.to(device)
                    y_value_batch = y_value_batch.to(device)

                    policy_pred, value_pred = model(xs_batch)
                    #log_probs = torch.log(policy_pred + 1e-9)
                    loss_policy = criterion_policy(policy_pred, y_policy_batch)
                    loss_value = criterion_value(value_pred.squeeze(), y_value_batch)
                    
                    acc = accuracy(policy_pred, y_policy_batch.argmax(dim=1))   # for accuracy
                    epoch_val_acc.append(acc)                                  # for accuracy

                    val_policy_loss += loss_policy.item()
                    val_value_loss += loss_value.item()

            val_policy_losses.append(val_policy_loss / len(val_loader))
            val_value_losses.append(val_value_loss / len(val_loader))
            
            epoch_val_acc = sum(epoch_val_acc) / len(epoch_val_acc) if epoch_val_acc else 0
            epoch_val_accs.append(epoch_val_acc)  # 记录每个epoch的平均准确率
            
            
            val_loss = val_policy_loss + val_value_loss
            scheduler.step()
            # 检查模型参数是否在更新
            for name, param in model.named_parameters():
                if param.requires_grad:
                    f.write(f"{name}: {param.data.sum().item()}\n")

            # 在终端打印当前的 Epoch 和损失值
            output = f'\rEpoch {epoch + 1}/{RN_EPOCHS}, ' \
                     f'Train Loss: {train_policy_losses[-1]:.4f}/{train_value_losses[-1]:.4f}, ' \
                     f'Val Loss: {val_policy_losses[-1]:.4f}/{val_value_losses[-1]:.4f}'
            print(output)
            print(f"Epoch {epoch + 1}, Train Accuracy: {epoch_train_acc:.4f}, Validation Accuracy: {epoch_val_acc:.4f}")
            f.write(output + '\n')  # 写入文件

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                model.eval()
                script_model = torch.jit.script(model)
                script_model.save('./model/1201/22layers/best_val.pt')
                print(" - Saved best model")
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                break

        print('\nTraining completed.')
    # Try to plot the training and validation losses
    try:
        current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
        plt.figure(figsize=(12, 6))
        plt.plot(train_policy_losses, label='Train Policy Loss')
        plt.plot(train_value_losses, label='Train Value Loss')
        plt.plot(val_policy_losses, label='Validation Policy Loss')
        plt.plot(val_value_losses, label='Validation Value Loss')
        plt.xlabel('Epoch')
        plt.ylabel('loss')
        plt.title('Training and Validation loss')
        plt.legend()
        plt.savefig(f'training_validation_losses_{current_time}.png')
        
        plt.figure(figsize=(12, 6))
        plt.plot(epoch_train_accs, label='Train Accuracy')
        plt.plot(epoch_val_accs, label='Validation Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Training and Validation acc')
        plt.legend()
        plt.savefig(f'training_validation_acc_{current_time}.png')        

    except Exception as e:
        print(f"Failed to plot due to an error: {e}")

    # 保存最终模型
    try:
        model.eval()
        script_model = torch.jit.script(model)
        script_model.save('./model/1201/22layers/latest.pt')
        print("Final model saved successfully.")
    except Exception as e:
        print(f"Failed to save the final model due to an error: {e}")
   

if __name__=='__main__':
    start_time = time.time()
    train_network()
    print("train_network() time: {:.2f} seconds".format(time.time() - start_time))