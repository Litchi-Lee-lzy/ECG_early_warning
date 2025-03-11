import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath
from warning_model_utils.task2_comprison_model.convneXtUtils import LayerNorm, GRN


class Block(nn.Module):
    """ ConvNeXtV2 Block.

    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
    """

    def __init__(self, dim, drop_path=0.):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=(1, 7), padding=(0, 3), groups=dim)  # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.grn = GRN(4 * dim)
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x


class Decoder_block(nn.Module):

    def __init__(self, inplanes, outplanes, kernel_size=(5,), stride=(5,), padding=(1,), dilation=(1,)):
        super(Decoder_block, self).__init__()
        self.upsample = nn.ConvTranspose1d(in_channels=inplanes, out_channels=outplanes,
                                           padding=padding, kernel_size=kernel_size, stride=stride, bias=False,
                                           dilation=dilation)
        self.conv1 = nn.Conv1d(in_channels=outplanes, out_channels=outplanes, kernel_size=kernel_size, bias=False,
                               stride=1, padding=(kernel_size // 2,))
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(in_channels=outplanes, out_channels=outplanes, kernel_size=kernel_size, bias=False,
                               stride=1, padding=(kernel_size // 2,))

    def forward(self, x1):
        out = self.upsample(x1)
        # print(out.shape)
        out = self.conv1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.relu(out)
        # print(out.shape)
        return out


class leadATT(nn.Module):
    def __init__(self, channel=1024, reduction=16):
        super(leadATT, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
        self.leadfusion = nn.Sequential(
            nn.Linear(12, 4, bias=False),
            nn.ReLU(),
            nn.Linear(4, 12, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, h, w = x.size()  # b c h w
        y = self.avg_pool(x).view(b, c)
        channelWeight = self.fc(y).view(b, c, 1)
        # lead weight
        x1 = x.permute(0, 2, 1, 3)  # b h c w
        x1 = self.avg_pool(x1).view(b, h)
        leadWeight = self.leadfusion(x1)
        leadWeight = leadWeight.view(b, 1, h, 1)
        leadWeight = leadWeight.expand(b, c, h, w)
        x = torch.sum(x * leadWeight, dim=2)
        x = x.view(b, c, w)
        return x * channelWeight.expand_as(x)


class ConvNeXtV2(nn.Module):
    """ ConvNeXt V2

    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
    """

    def __init__(self, in_chans=1, num_classes=3,
                 depths=[3, 3, 9, 3], dims=[96, 192, 384, 768],
                 drop_path_rate=0., head_init_scale=1.
                 ):
        super().__init__()
        self.depths = depths
        self.downsample_layers = nn.ModuleList()  # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=(1, 10), stride=(1, 10)),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                nn.Conv2d(dims[i], dims[i + 1], kernel_size=(1, 2), stride=(1, 2)),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()  # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[Block(dim=dims[i], drop_path=dp_rates[cur + j]) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]



        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)  # final norm layer
        self.head = nn.Linear(dims[-1], num_classes)

        self.apply(self._init_weights)
        self.head.weight.data.mul_(head_init_scale)
        self.head.bias.data.mul_(head_init_scale)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            # nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        x = x.unsqueeze(1).unsqueeze(1)
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        # x = self.leadATT(x)
        return x

    def forward(self, x, mask=None, type="classification"):

        # if type == "classification":
        x = self.forward_features(x)
        x = self.norm(x.mean([-2, -1]))
        x = self.head(x)
        return x
        # else:
        #     # mask
        #     assert mask is not None
        #     unmask = np.invert(mask)
        #     mask = torch.as_tensor(mask, device=x.device)
        #     unmask = torch.as_tensor(unmask, device=x.device)
        #     x1 = x * mask.unsqueeze(1)
        #     x1 = self.forward_features(x1)
        #     ecg_pred = self.tar_net(x1)
        #     loss = (x - ecg_pred) ** 2
        #     loss = (loss.mean(dim=1))
        #     mask_loss = (loss * mask).sum() / mask.sum()
        #     unmasked_loss = (loss * unmask).sum() / unmask.sum()
        #     return mask_loss, unmasked_loss


def convnextv2_atto(**kwargs):
    model = ConvNeXtV2(depths=[2, 2, 6, 2], dims=[40, 80, 160, 320], **kwargs)
    return model


def convnextv2_femto(**kwargs):
    model = ConvNeXtV2(depths=[2, 2, 6, 2], dims=[48, 96, 192, 384], **kwargs)
    return model


def convnext_pico(**kwargs):
    model = ConvNeXtV2(depths=[2, 2, 6, 2], dims=[64, 128, 256, 512], **kwargs)
    return model


def convnextv2_nano(**kwargs):
    model = ConvNeXtV2(depths=[2, 2, 8, 2], dims=[80, 160, 320, 640], **kwargs)
    return model


def convnextv2_tiny(**kwargs):
    model = ConvNeXtV2(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], **kwargs)
    return model


def convnextv2_base(**kwargs):
    model = ConvNeXtV2(depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024], **kwargs)
    return model


def convnextv2_large(**kwargs):
    model = ConvNeXtV2(depths=[3, 3, 27, 3], dims=[192, 384, 768, 1536], **kwargs)
    return model


def convnextv2_huge(**kwargs):
    model = ConvNeXtV2(depths=[3, 3, 27, 3], dims=[352, 704, 1408, 2816], **kwargs)
    return model


if __name__ == "__main__":
    model = convnextv2_atto()
    ECG = torch.rand((1,1280))
    print(model(ECG))
    from torchsummary import summary

    # summary(model.to(device), input_size=(1, 12, 5000), batch_size=-1)
