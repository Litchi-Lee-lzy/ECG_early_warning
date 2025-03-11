import math

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from torch.optim.optimizer import Optimizer

from torch import nn
from torch.nn import functional as F


from warning_model_utils.pretrain_model import ViT1D


class Projection(nn.Module):
    def __init__(self, input_dim=2048, hidden_dim=2048, output_dim=128):
        super().__init__()
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.model = nn.Sequential(
            # nn.AdaptiveAvgPool2d((1, 1)),
            # Flatten(),
            nn.Linear(self.input_dim, self.hidden_dim, bias=True),
            # nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.output_dim, bias=True))

        self.avgpool = nn.AdaptiveAvgPool1d(1)
    def forward(self, x):
        x = self.avgpool(x.permute(0, 2, 1)).squeeze()
        x = self.model(x)
        return F.normalize(x, dim=1)


class SyncFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, tensor):
        ctx.batch_size = tensor.shape[0]

        gathered_tensor = [torch.zeros_like(tensor) for _ in range(torch.distributed.get_world_size())]

        torch.distributed.all_gather(gathered_tensor, tensor)
        gathered_tensor = torch.cat(gathered_tensor, 0)

        return gathered_tensor

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        torch.distributed.all_reduce(grad_input, op=torch.distributed.ReduceOp.SUM, async_op=False)

        return grad_input[torch.distributed.get_rank() * ctx.batch_size:(torch.distributed.get_rank() + 1) *
                                                                        ctx.batch_size]


class CustomSimCLR(nn.Module):

    def __init__(self,
                 encoder="vit",num_classes=5):
        """
        Args:
            batch_size: the batch size
            num_samples: num samples in the dataset
            warmup_epochs: epochs to warmup the lr for
            lr: the optimizer learning rate
            opt_weight_decay: the optimizer weight decay
            loss_temperature: the loss temperature
        """

        super(CustomSimCLR, self).__init__()

        self.encoder_name = encoder
        if "vit" in encoder:
            self.frontend = ViT1D()
            dim = 256
        else:
            print("wrong encoder name : {}".format(self.encoder_name))
            pass
        self.projection =Projection(input_dim=dim, hidden_dim=512, output_dim=128)
        # pdb.set_trace()
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.multi_label_classifier = nn.Linear(dim, num_classes)


    def shared_forward(self, batch, device ):
        (x1, y1), (x2, y2) = batch
        # ENCODE
        # encode -> representations
        # (b, 3, 32, 32) -> (b, 2048, 2, 2)

        x1 = x1.float().to(device).squeeze(1)
        x2 = x2.float().to(device).squeeze(1)

        if self.encoder_name == "vit":
            h1 = self.frontend(x1)
            h2 = self.frontend(x2)
        else:
            h1 = self.frontend.forward_features(x1)
            h1 = h1.squeeze(2)
            h1 = h1.permute(0, 2, 1)

            h2 = self.frontend.forward_features(x2)
            h2 = h2.squeeze(2)
            h2 = h2.permute(0, 2, 1)


        # PROJECT
        # img -> E -> h -> || -> z
        # (b, 2048, 2, 2) -> (b, 128)
        z1 = self.projection(h1.squeeze())
        z2 = self.projection(h2.squeeze())

        return z1, z2

    def nt_xent_loss(self, out_1, out_2, temperature, eps=1e-6):
        """
            assume out_1 and out_2 are normalized
            out_1: [batch_size, dim]
            out_2: [batch_size, dim]
        """
        # gather representations in case of distributed training
        # out_1_dist: [batch_size * world_size, dim]
        # out_2_dist: [batch_size * world_size, dim]
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            out_1_dist = SyncFunction.apply(out_1)
            out_2_dist = SyncFunction.apply(out_2)
            print("out dist shape: ", out_1_dist.shape)
        else:
            out_1_dist = out_1
            out_2_dist = out_2

        # out: [2 * batch_size, dim]
        # out_dist: [2 * batch_size * world_size, dim]
        out = torch.cat([out_1, out_2], dim=0)
        out_dist = torch.cat([out_1_dist, out_2_dist], dim=0)

        # cov and sim: [2 * batch_size, 2 * batch_size * world_size]
        # neg: [2 * batch_size]
        cov = torch.mm(out, out_dist.t().contiguous())
        sim = torch.exp(cov / temperature)
        neg = sim.sum(dim=-1)

        # from each row, subtract e^1 to remove similarity measure for x1.x1
        row_sub = torch.Tensor(neg.shape).fill_(math.e).to(neg.device)
        neg = torch.clamp(neg - row_sub, min=eps)  # clamp for numerical stability

        # Positive similarity, pos becomes [2 * batch_size]
        pos = torch.exp(torch.sum(out_1 * out_2, dim=-1) / temperature)
        pos = torch.cat([pos, pos], dim=0)

        loss = -torch.log(pos / (neg + eps)).mean()

        return loss


    def forward(self, batch, DEVICE):

        z1, z2 = self.shared_forward(batch, DEVICE)
        loss = self.nt_xent_loss(z1, z2, 0.5)
        return loss

    def classify(self, signal):
        if self.encoder_name == "vit":
            encoded_tokens = self.frontend(signal)
        else:

            encoded_tokens = self.frontend.forward_features(signal)
            encoded_tokens = encoded_tokens.squeeze(2)
            encoded_tokens = encoded_tokens.permute(0, 2, 1)


        latentFeature = self.avgpool(encoded_tokens.permute(0, 2, 1))
        latentFeature = latentFeature.view(latentFeature.size(0), -1)
        out = self.multi_label_classifier(latentFeature)
        return out

