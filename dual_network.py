import torch
import torch.nn as nn
import torch.nn.functional as F


# Define model parameters as global variables
DN_FILTERS = 256        # 改256
DN_RESIDUAL_NUM = 19    # 改19
DN_INPUT_SHAPE = (3, 11, 11)
DN_OUTPUT_SIZE = 122

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(num_features=out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x)
        return x

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = ConvBlock(in_channels, out_channels)
        self.conv2 = ConvBlock(out_channels, out_channels)
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
        c, a, b = input_shape
        self.initial_conv = ConvBlock(in_channels=c, out_channels=filters)
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(filters, filters) for _ in range(residual_blocks)]
        )
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.policy_head = nn.Linear(filters, output_size)
        self.value_head = nn.Linear(filters, 3)

    def forward(self, x):
        x = self.initial_conv(x)
        x = self.res_blocks(x)
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)  # Flatten
        p = self.policy_head(x)
        p = F.softmax(p, dim=1)
        v = self.value_head(x)
        v = torch.tanh(v)
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
    script_model.save('./model/0829/19layers/best.pt')
    
