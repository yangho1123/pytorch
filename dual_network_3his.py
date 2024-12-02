import torch
import torch.nn as nn
import torch.nn.functional as F


# Define model parameters as global variables
DN_FILTERS = 512        # 改256
DN_RESIDUAL_NUM = 22    # 改19
DN_INPUT_SHAPE = (8, 5, 11, 11)
DN_OUTPUT_SIZE = 122

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_rate=0.0):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(num_features=out_channels)
        self.dropout = nn.Dropout2d(p=dropout_rate) if dropout_rate > 0 else nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x)
        x = self.dropout(x)
        return x

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_rate=0.1):
        super(ResidualBlock, self).__init__()
        self.conv1 = ConvBlock(in_channels, out_channels, dropout_rate)
        self.conv2 = ConvBlock(out_channels, out_channels, dropout_rate)
        self.match_channels = in_channels != out_channels
        if self.match_channels:
            self.channel_matching_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.channel_matching_conv = nn.Identity()

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.conv2(x)        
        residual = self.channel_matching_conv(residual)
        x = x + residual
        x = F.relu(x)
        return x

class DualNetwork(nn.Module):
    def __init__(self, input_shape, filters, residual_blocks, output_size):
        super(DualNetwork, self).__init__()
        time_steps, channels, height, width = input_shape  # 接收 8, 5, 11, 11
        self.initial_conv = ConvBlock(in_channels=time_steps * channels, out_channels=filters)
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(filters, filters, dropout_rate=0.1) for _ in range(residual_blocks)]
        )
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        # Policy Head Adjustment
        self.policy_head_conv = nn.Conv2d(filters, 2, kernel_size=1)  # 1x1 conv
        self.policy_head_bn = nn.BatchNorm2d(2)
        self.policy_head_activation = nn.ReLU()
        self.policy_dropout = nn.Dropout(p=0.5)
        self.policy_head_fc = nn.Linear(2 * height * width, output_size)
        
        # Value Head Adjustment
        self.value_head_conv = nn.Conv2d(filters, 1, kernel_size=1)  # 1x1 conv
        self.value_head_bn = nn.BatchNorm2d(1)
        self.value_head_activation = nn.ReLU()
        self.value_head_fc1 = nn.Linear(height * width, 256)
        self.value_head_activation2 = nn.ReLU()
        self.value_dropout = nn.Dropout(p=0.5)
        self.value_head_fc2 = nn.Linear(256, 1)
        self.value_head_tanh = nn.Tanh()

    def forward(self, x):
        x = self.initial_conv(x)
        x = self.res_blocks(x)
        #x = self.global_avg_pool(x)
       # x = x.view(x.size(0), -1)  # Flatten
        # Policy Head
        p = self.policy_head_conv(x)
        p = self.policy_head_bn(p)
        p = self.policy_head_activation(p)
        p = p.view(p.size(0), -1)  # Flatten
        p = self.policy_dropout(p)
        p = self.policy_head_fc(p)
        # p = F.softmax(p, dim=1)
        
        # Value Head
        v = self.value_head_conv(x)
        v = self.value_head_bn(v)
        v = self.value_head_activation(v)
        v = v.view(v.size(0), -1)  # Flatten
        v = self.value_head_fc1(v)
        v = self.value_head_activation2(v)
        v = self.value_dropout(v) 
        v = self.value_head_fc2(v)
        v = self.value_head_tanh(v)
        return p, v

# def save_model(model):
#     script_model = torch.jit.script(model)
#     script_model.save('./model/best.pt')

# def load_model(model, path='./model/best.pt'):
#     if os.path.exists(path):
#         model = torch.jit.load(path)    

if __name__ == '__main__':
    # Initialize model with global variables
    model = DualNetwork(DN_INPUT_SHAPE, DN_FILTERS, DN_RESIDUAL_NUM, DN_OUTPUT_SIZE)
    script_model = torch.jit.script(model)
    script_model.save('./model/1201/22layers/best.pt')
    
