from dual_network import DN_INPUT_SHAPE,DN_FILTERS,DN_OUTPUT_SIZE,DN_RESIDUAL_NUM
from dual_network import DualNetwork
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
import numpy as np
import pickle

RN_EPOCHS = 125     # 訓練次數

def load_data():

    # 从最新的文件中读取所有数据
    history_path = sorted(Path('../torchdata').glob('*.history'))[-1]     # 依照檔名由小到大排列
    with history_path.open(mode='rb') as f:
        return pickle.load(f)

def train_network():
    
    history = load_data()
    print("Total number of data points:", len(history))
    xs, y_policies, y_values = zip(*history)  #將資料轉換為3個獨立的串列：棋子配置(訓練對象)、策略(標籤)、價值(標籤)

    # 重塑訓練資料的shape
    a, b, c = DN_INPUT_SHAPE
    xs = np.array(xs)
    xs = xs.reshape(len(xs), c, a, b).transpose(0, 2, 3, 1)

    y_policies = np.array(y_policies)
    y_values = np.array(y_values)

    # 转换为 PyTorch 张量
    xs = torch.tensor(xs, dtype=torch.float32)
    y_policies = torch.tensor(y_policies, dtype=torch.float32)
    y_values = torch.tensor(y_values, dtype=torch.float32)

    # 创建 DataLoader
    dataset = TensorDataset(xs, y_policies, y_values)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True)

    # 加载模型并移动到 GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DualNetwork(DN_INPUT_SHAPE, DN_FILTERS, DN_RESIDUAL_NUM, DN_OUTPUT_SIZE).to(device)
    model.load_state_dict(torch.load('./model/best.pth'))
    model.train()
    # 定义优化器和损失函数
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion_policy = nn.CrossEntropyLoss()
    criterion_value = nn.MSELoss()
    # 学习率调整
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
    # 训练循环
    for epoch in range(RN_EPOCHS):
        for xs_batch, y_policy_batch, y_value_batch in dataloader:
            xs_batch, y_policy_batch, y_value_batch = xs_batch.to(device), y_policy_batch.to(device), y_value_batch.to(device)

            # 前向传播
            policy_pred, value_pred = model(xs_batch)

            # 计算损失
            loss_policy = criterion_policy(policy_pred, y_policy_batch)
            loss_value = criterion_value(value_pred.squeeze(), y_value_batch)
            loss = loss_policy + loss_value

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # 每个 epoch 后调整学习率
        scheduler.step()

        print(f'\rTrain {epoch+1}/{RN_EPOCHS}', end='')

    print('')
    # 保存模型
    torch.save(model.state_dict(), './model/latest.pth')
   

if __name__=='__main__':
    train_network()