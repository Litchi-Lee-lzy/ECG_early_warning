import torch.nn as nn
import numpy as np
import torch
# from fastai.layers import *
# from fastai.core import *

def _padding(downsample, kernel_size):
    """Compute required padding"""
    padding = max(0, int(np.floor((kernel_size - downsample + 1) / 2)))
    return padding


def _downsample(n_samples_in, n_samples_out):
    """Compute downsample rate"""
    downsample = int(n_samples_in // n_samples_out)
    if downsample < 1:
        raise ValueError("Number of samples should always decrease")
    if n_samples_in % n_samples_out != 0:
        raise ValueError("Number of samples for two consecutive blocks "
                         "should always decrease by an integer factor.")
    return downsample


class ResBlock1d(nn.Module):
    """Residual network unit for unidimensional signals."""

    def __init__(self, n_filters_in, n_filters_out, downsample, kernel_size, dropout_rate):
        if kernel_size % 2 == 0:
            raise ValueError("The current implementation only support odd values for `kernel_size`.")
        super(ResBlock1d, self).__init__()
        # Forward path
        padding = _padding(1, kernel_size)
        self.conv1 = nn.Conv1d(n_filters_in, n_filters_out, kernel_size, padding=padding, bias=False)
        self.bn1 = nn.BatchNorm1d(n_filters_out)
        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout_rate)
        padding = _padding(downsample, kernel_size)
        self.conv2 = nn.Conv1d(n_filters_out, n_filters_out, kernel_size,
                               stride=downsample, padding=padding, bias=False)
        self.bn2 = nn.BatchNorm1d(n_filters_out)
        self.dropout2 = nn.Dropout(dropout_rate)

        # Skip connection
        skip_connection_layers = []
        # Deal with downsampling
        if downsample > 1:
            maxpool = nn.MaxPool1d(downsample, stride=downsample)
            skip_connection_layers += [maxpool]
        # Deal with n_filters dimension increase
        if n_filters_in != n_filters_out:
            conv1x1 = nn.Conv1d(n_filters_in, n_filters_out, 1, bias=False)
            skip_connection_layers += [conv1x1]
        # Build skip conection layer
        if skip_connection_layers:
            self.skip_connection = nn.Sequential(*skip_connection_layers)
        else:
            self.skip_connection = None

    def forward(self, x, y):
        """Residual unit."""
        if self.skip_connection is not None:
            y = self.skip_connection(y)
        else:
            y = y
        # 1st layer
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout1(x)

        # 2nd layer
        x = self.conv2(x)
        x += y  # Sum skip connection and main connection
        y = x
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout2(x)
        return x, y

class Decoder_block(nn.Module):

    def __init__(self, inplanes, outplanes, kernel_size=5, stride=5, output_padding=1):
        super(Decoder_block, self).__init__()
        self.upsample = nn.Upsample(scale_factor=stride)
        # self.upsample = nn.ConvTranspose1d(in_channels=inplanes, out_channels=outplanes,
        #                    padding=1, kernel_size=kernel_size, stride=stride, output_padding=output_padding,bias=False)
        self.conv1 = nn.Conv1d(in_channels=inplanes, out_channels=outplanes, kernel_size=kernel_size, bias=False,
                               stride=1, padding=(kernel_size//2,))
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(in_channels=outplanes, out_channels=outplanes, kernel_size=kernel_size, bias=False,
                               stride=1, padding=(kernel_size//2,))

    def forward(self, x1):
        out = self.upsample(x1)
        out = self.conv1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.relu(out)

        return out

class ResNet1d(nn.Module):
    """Residual network for unidimensional signals.
    Parameters
    ----------
    input_dim : tuple
        Input dimensions. Tuple containing dimensions for the neural network
        input tensor. Should be like: ``(n_filters, n_samples)``.
    blocks_dim : list of tuples
        Dimensions of residual blocks.  The i-th tuple should contain the dimensions
        of the output (i-1)-th residual block and the input to the i-th residual
        block. Each tuple shoud be like: ``(n_filters, n_samples)``. `n_samples`
        for two consecutive samples should always decrease by an integer factor.
    dropout_rate: float [0, 1), optional
        Dropout rate used in all Dropout layers. Default is 0.8
    kernel_size: int, optional
        Kernel size for convolutional layers. The current implementation
        only supports odd kernel sizes. Default is 17.
    References
    ----------
    .. [1] K. He, X. Zhang, S. Ren, and J. Sun, "Identity Mappings in Deep Residual Networks,"
           arXiv:1603.05027, Mar. 2016. https://arxiv.org/pdf/1603.05027.pdf.
    .. [2] K. He, X. Zhang, S. Ren, and J. Sun, "Deep Residual Learning for Image Recognition," in 2016 IEEE Conference
           on Computer Vision and Pattern Recognition (CVPR), 2016, pp. 770-778. https://arxiv.org/pdf/1512.03385.pdf
    """

    def __init__(self, input_dim, blocks_dim, n_classes, kernel_size=17, dropout_rate=0.8, classifier="Linear"):
        super(ResNet1d, self).__init__()
        # First layers
        n_filters_in, n_filters_out = input_dim[0], blocks_dim[0][0]
        n_samples_in, n_samples_out = input_dim[1], blocks_dim[0][1]
        downsample = _downsample(n_samples_in, n_samples_out)
        padding = _padding(downsample, kernel_size)
        self.conv1 = nn.Conv1d(n_filters_in, n_filters_out, kernel_size, bias=False,
                               stride=downsample, padding=padding)
        self.bn1 = nn.BatchNorm1d(n_filters_out)

        # Residual block layers
        self.res_blocks = []
        for i, (n_filters, n_samples) in enumerate(blocks_dim):
            n_filters_in, n_filters_out = n_filters_out, n_filters
            n_samples_in, n_samples_out = n_samples_out, n_samples
            downsample = _downsample(n_samples_in, n_samples_out)
            resblk1d = ResBlock1d(n_filters_in, n_filters_out, downsample, kernel_size, dropout_rate)
            self.add_module('resblock1d_{0}'.format(i), resblk1d)
            self.res_blocks += [resblk1d]
        self.avg = nn.AdaptiveAvgPool1d(1)
        # Linear layer
        n_filters_last, n_samples_last = blocks_dim[-1]
        last_layer_dim = n_filters_last
        if classifier == "Linear":
            self.lin = nn.Linear(last_layer_dim, n_classes)

        self.n_blk = len(blocks_dim)
        self.Sig = nn.Sigmoid()

        self.tar_net = nn.Sequential(
            Decoder_block(256, 256, kernel_size=5, stride=2, output_padding=2), # 64
            Decoder_block(256, 128, kernel_size=5, stride=2, output_padding=2), #256
            Decoder_block(128, 64, kernel_size=3, stride=2),  # 1024
            Decoder_block(64, 32, kernel_size=3, stride=5),  # 4096
            nn.Conv1d(in_channels=32, out_channels=12, kernel_size=(1,),bias=False)

            # Emd2ECG()
        )

    def auto_encoder(self, x):

        x1 = self.conv1(x)
        x1 = self.bn1(x1)

        # Residual blocks
        y = x1
        for blk in self.res_blocks:
            x1, y = blk(x1, y)

        ecg = self.tar_net(x1)
        return ecg

    def forward(self, x, task_type="classification", mask=None):
        """Implement ResNet1d forward propagation"""
        # Flatten array
        if task_type=="classification":
            # First layers
            b, h, w = x.size()
            x = x.view(b, h, w)
            x = self.conv1(x)
            x = self.bn1(x)

            # Residual blocks
            y = x
            for blk in self.res_blocks:
                x, y = blk(x, y)
            f = self.avg(x)
            f = f.view(f.size(0), -1)
            x = self.lin(f)
            return x
        elif task_type=="reconstruction":
            #mask
            b, h, w = x.size()
            x = x.view(b, h, w)
            unmask = np.invert(mask)
            mask = torch.as_tensor(mask, device=x.device)
            unmask = torch.as_tensor(unmask, device=x.device)

            x1 = x * mask
            x1 = self.conv1(x1)
            x1 = self.bn1(x1)

            # Residual blocks
            y = x1
            for blk in self.res_blocks:
                x1, y = blk(x1, y)
            ecg = self.tar_net(x1)
            # print(ecg.shape)

            loss = (x - ecg) ** 2
            mask_loss = (loss * mask).sum() / mask.sum()
            unmasked_loss = (loss * unmask).sum() / unmask.sum()
            return mask_loss, unmasked_loss






if __name__ == "__main__":

    # decoder = Decoder_block(12,64, kernel_size=4, stride=4)
    a = torch.rand((2,1,12, 5000))
    mask = np.full((2, 12, 5000), True)
    # print(decoder(a).shape)
    N_LEADS = 12  # the 12 leads
    N_CLASSES = 2  # just the AF Normal
    seq_length = 5000
    net_filter_size = [64, 128, 196, 256, 256]
    net_seq_length = [5000, 1000, 500, 250, 125]
    kernel_size = 17
    dropout_rate = 0.8
    lr = 0.001
    patience = 7
    min_lr = 1e-7
    lr_factor = 0.1
    epoch = 100
    folder = "./modelResult/"
    model = ResNet1d(input_dim=(N_LEADS, seq_length),
                     blocks_dim=list(zip(net_filter_size, net_seq_length)),
                     n_classes=N_CLASSES,
                     kernel_size=kernel_size,
                     dropout_rate=dropout_rate, classifier="Linear")

    print(model(a)[1].shape)
    print(model(a, task_type="reconstruction", mask=mask).shape)

