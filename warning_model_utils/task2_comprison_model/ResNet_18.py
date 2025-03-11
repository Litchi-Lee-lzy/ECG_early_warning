import torch
import torch.nn as nn
import torch.functional as F



class Residual(nn.Module):
    def __init__(self, input_channel, out_channel, use_conv1x1=False, strides=1):
        super().__init__()
        self.conv1 = nn.Conv1d(input_channel, out_channel, kernel_size=3, stride=strides, padding=1)
        self.conv2 = nn.Conv1d(out_channel, out_channel, kernel_size=3, stride=1, padding=1)
        if use_conv1x1:
            self.conv3 = nn.Conv1d(input_channel, out_channel, kernel_size=1, stride=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm1d(out_channel)
        self.bn2 = nn.BatchNorm1d(out_channel)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, X):
        # x = X
        # a = self.conv1(x)
        Y = self.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        Y += X
        return self.relu(Y)


# 多个残差单元组成的残差块
def res_block(input_channel, out_channel, num_residuals, first_block=False):
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(Residual(input_channel, out_channel, use_conv1x1=True, strides=2))
        else:
            blk.append(Residual(out_channel, out_channel))
    return blk



# ECGNET
class ResNet_18(nn.Module):
    def __init__(self, input_channel=1, num_classes=3):
        super(ResNet_18, self).__init__()
        block_1 = nn.Sequential(
            nn.Conv1d(input_channel, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        )
        block_2 = nn.Sequential(*res_block(64, 64, 2))
        block_3 = nn.Sequential(*res_block(64, 128, 2))
        block_4 = nn.Sequential(*res_block(128, 256, 2))
        block_5 = nn.Sequential(*res_block(256, 512, 2))
        self.net18 = nn.Sequential(block_1, block_2, block_3, block_4, block_5,
                              nn.AdaptiveAvgPool1d((1)),
                              nn.Flatten(),
                              nn.Linear(512, num_classes)
                              )

    def forward(self, x):
        x = x.unsqueeze(dim=1)
        x = x.to(torch.float)
        x = self.net18(x)
        return x


if __name__ == "__main__":
    model = ResNet_18()

    data = torch.rand((1,1, 1280))

    print(model(data))

