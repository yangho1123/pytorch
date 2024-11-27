from dual_network import DN_INPUT_SHAPE,DN_FILTERS,DN_OUTPUT_SIZE,DN_RESIDUAL_NUM
from dual_network import DualNetwork
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from pathlib import Path
import numpy as np
import pickle
import time
import matplotlib.pyplot as plt
import datetime

RN_EPOCHS = 10     # 訓練次數

def load_data(last_n=4):
    # 获取所有 .history 文件并按文件名排序
    history_paths = sorted(Path('../torchdata').glob('*.history'))
    # 初始化历史数据列表
    history = []
    # 從最新的last_n個文件中讀取資料
    for path in history_paths[-last_n:]:
        with path.open(mode='rb') as f:
            history.extend(pickle.load(f))
    return history


def train_network():
    history = load_data()
    print("Total number of data points:", len(history))
    xs, y_policies, y_values = zip(*history)
    a, b, c = DN_INPUT_SHAPE
    xs = np.array(xs)
    xs = xs.reshape(len(xs), a, b, c)
    # print(np.array(y_policies).shape)
    # print(np.array(y_values).shape)
    xs = torch.tensor(np.array(xs), dtype=torch.float32)
    y_policies = torch.tensor(np.array(y_policies), dtype=torch.float32)
    y_values = torch.tensor(np.array(y_values), dtype=torch.float32)
    dataset = TensorDataset(xs, y_policies, y_values)
    # Split the dataset into training and validation set
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # 重塑訓練資料的shape
    # a, b, c = DN_INPUT_SHAPE
    # xs = np.array(xs)
    # xs = xs.reshape(len(xs), a, b, c)
    # #xs = xs.reshape(len(xs), c, a, b).transpose(0, 2, 3, 1)

    # y_policies = np.array(y_policies)
    # y_values = np.array(y_values)

    # # 转换为 PyTorch 张量
    # xs = torch.tensor(xs, dtype=torch.float32)
    # y_policies = torch.tensor(y_policies, dtype=torch.float32)
    # y_values = torch.tensor(y_values, dtype=torch.float32)

    # 创建 DataLoader
    #dataset = TensorDataset(xs, y_policies, y_values)
    #dataloader = DataLoader(dataset, batch_size=128, shuffle=True)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)

    # 加载模型并移动到 GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = DualNetwork(DN_INPUT_SHAPE, DN_FILTERS, DN_RESIDUAL_NUM, DN_OUTPUT_SIZE).to(device)
    # model.load_state_dict(torch.load('./model/best.pt'))
    model = torch.jit.load('./model/19layers/best.pt').to(device)
    model.train()
    # 定义优化器和损失函数
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion_policy = nn.CrossEntropyLoss()
    criterion_value = nn.MSELoss()
    # 学习率调整
    #scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
    #scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, min_lr=0.000625, verbose=True)
    # 損失記錄
    #average_policy_losses = []
    #average_value_losses = []
    train_policy_losses, train_value_losses, val_policy_losses, val_value_losses = [], [], [], []
    # 训练循环
    for epoch in range(RN_EPOCHS):
        model.train()
        total_policy_loss, total_value_loss = 0, 0
        for xs_batch, y_policy_batch, y_value_batch in train_loader:
            xs_batch = xs_batch.to(device)
            y_policy_batch = y_policy_batch.to(device)
            y_value_batch = y_value_batch.to(device)

            policy_pred, value_pred = model(xs_batch)   #將data傳入model進行forward propagation
            loss_policy = criterion_policy(policy_pred, y_policy_batch)
            loss_value = criterion_value(value_pred.squeeze(), y_value_batch)
            loss = loss_policy + loss_value     #計算loss

            optimizer.zero_grad()   # 清空前一次的gradient
            loss.backward()         # 根據loss進行back propagation，計算gradient
            optimizer.step()        # 做gradient descent

            total_policy_loss += loss_policy.item()
            total_value_loss += loss_value.item()

        # Calculate average losses for the current epoch
        train_policy_losses.append(total_policy_loss / len(train_loader))
        train_value_losses.append(total_value_loss / len(train_loader))

        # Validation phase
        model.eval()

        with torch.no_grad():
            total_policy_loss, total_value_loss = 0, 0
            for xs_batch, y_policy_batch, y_value_batch in val_loader:
                xs_batch = xs_batch.to(device)
                y_policy_batch = y_policy_batch.to(device)
                y_value_batch = y_value_batch.to(device)

                policy_pred, value_pred = model(xs_batch)
                loss_policy = criterion_policy(policy_pred, y_policy_batch)
                loss_value = criterion_value(value_pred.squeeze(), y_value_batch)

                total_policy_loss += loss_policy.item()
                total_value_loss += loss_value.item()

            val_policy_losses.append(total_policy_loss / len(val_loader))
            val_value_losses.append(total_value_loss / len(val_loader))

        # val_loss = (val_policy_losses + val_value_losses) / 2   # Your validation process to compute loss
        val_loss = (sum(val_policy_losses) / len(val_policy_losses) + sum(val_value_losses) / len(val_value_losses)) / 2
        scheduler.step(val_loss)  # Adjust learning rate based on the validation loss
        #scheduler.step()
        print(f'\rTrain {epoch+1}/{RN_EPOCHS}', end='')

    print('')
    # Try to plot the training and validation losses
    try:
        current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
        plt.figure(figsize=(12, 6))
        plt.plot(train_policy_losses, label='Train Policy Loss')
        plt.plot(train_value_losses, label='Train Value Loss')
        plt.plot(val_policy_losses, label='Validation Policy Loss')
        plt.plot(val_value_losses, label='Validation Value Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Losses')
        plt.legend()
        plt.savefig(f'training_validation_losses_{current_time}.png')
    except Exception as e:
        print(f"Failed to plot due to an error: {e}")

    # Save the model regardless of the plotting outcome
    try:
        model.eval()
        script_model = torch.jit.script(model)
        script_model.save('./model/19layers/latest.pt')
        print("Model saved successfully.")
    except Exception as e:
        print(f"Failed to save the model due to an error: {e}")
   

if __name__=='__main__':
    start_time = time.time()
    train_network()
    print("train_network() time: {:.2f} seconds".format(time.time() - start_time))