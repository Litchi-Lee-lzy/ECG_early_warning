import torch
import torch.nn as nn

class UnidirectionalLSTM(nn.Module):
    def __init__(self, input_dim=12, seq_len=1000, hidden_dim=256, num_layers=2, num_classes=5, bidirectional=True):
        super(UnidirectionalLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        # 定义LSTM层
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, bidirectional=bidirectional)

        # 使用concat pooling，因此全连接层的输入是隐藏层的两倍（双向LSTM的情况）
        self.fc = nn.Linear(hidden_dim * 4 if bidirectional else hidden_dim * 2, num_classes)

    def forward(self, x):
        # 这里假设输入的尺寸为 (batch_size, channels=1, length=1000)
        # 转置输入使其符合LSTM的要求：batch_size, seq_len, input_dim
        x = x.transpose(1, 2)  # 将 (batch_size, 1, length) 转换为 (batch_size, length, 1)

        # # 初始化LSTM的隐藏状态和细胞状态
        # h0 = torch.zeros(self.num_layers * 2 if self.bidirectional else self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        # c0 = torch.zeros(self.num_layers * 2 if self.bidirectional else self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        # # 通过LSTM层
        # out, _ = self.lstm(x, (h0, c0))

        out, _ = self.lstm(x)
        # 使用concat pooling：最大池化和平均池化
        avg_pool = torch.mean(out, 1)
        max_pool, _ = torch.max(out, 1)

        # 将池化结果拼接
        out = torch.cat((avg_pool, max_pool), 1)

        # 全连接层分类
        out = self.fc(out)
        return out