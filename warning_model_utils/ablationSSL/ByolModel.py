
import math

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from copy import deepcopy
from torch import nn
from torch.optim.optimizer import Optimizer

from torch import nn
from torch.nn import functional as F

from warning_model_utils.pretrain_model import ViT1D



def mean(res, key1, key2=None):
    if key2 is not None:
        return torch.stack([x[key1][key2] for x in res]).mean()
    return torch.stack([x[key1] for x in res if type(x) == dict and key1 in x.keys()]).mean()

class MLP(nn.Module):
    def __init__(self, input_dim=512, hidden_size=4096, output_dim=256):
        super().__init__()
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_size, bias=False),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, output_dim, bias=True))

    def forward(self, x):
        x = self.model(x)
        return x


class SiameseArm(nn.Module):
    def __init__(self, encoder=None, out_dim=128, hidden_size=512, projector_dim=512, num_classes=5):
        super().__init__()

        self.encoder_name = encoder
        if "vit" in encoder:
            self.frontend = ViT1D()
            dim = 256
        else:
            print("wrong encoder name : {}".format(self.encoder_name))
            pass
        # Pooler
        self.pooler = nn.AdaptiveAvgPool1d((1))
        # Projector
        projector_dim = dim
        self.projector = MLP(
            input_dim=projector_dim, hidden_size=hidden_size, output_dim=out_dim)
        # Predictor
        self.predictor = MLP(
            input_dim=out_dim, hidden_size=hidden_size, output_dim=out_dim)

        # classifier head
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.multi_label_classifier = nn.Linear(dim, num_classes)

    def forward(self, x):
        x = x.squeeze(1)
        if self.encoder_name == "vit":
            y = self.frontend(x)
            y = y.permute(0, 2, 1)
        else:
            y = self.frontend(x)
            y = y.squeeze(2)

        y = self.pooler(y)
        y = y.view(y.size(0), -1)
        z = self.projector(y)
        h = self.predictor(z)
        return y, z, h

    def classify(self, x):

        if self.encoder_name == "vit":
            y = self.frontend(x)
            y = y.permute(0, 2, 1)
        else:
            y = self.frontend.forward_features(x)
            y = y.squeeze(2)
        y = self.avgpool(y)
        y = y.view(y.size(0), -1)
        y = self.multi_label_classifier(y)
        return y


class BYOLMAWeightUpdate():
    def __init__(self, initial_tau=0.996):
        """
        Args:
            initial_tau: starting tau. Auto-updates with every training step
        """
        super().__init__()
        self.initial_tau = initial_tau
        self.current_tau = initial_tau

    def on_train_batch_end(self, pl_module, global_step, max_step):
        # get networks
        online_net = pl_module.online_network
        target_net = pl_module.target_network

        # update weights
        self.update_weights(online_net, target_net)

        # update tau after
        self.current_tau = self.update_tau(global_step, max_step)

    def update_tau(self, global_step, max_step):
        max_steps = max_step
        tau = 1 - (1 - self.initial_tau) * (math.cos(math.pi *
                                                     global_step / max_steps) + 1) / 2
        return tau

    def update_weights(self, online_net, target_net):
        # apply MA weight update
        for (name, online_p), (_, target_p) in zip(online_net.named_parameters(), target_net.named_parameters()):
            if 'weight' in name:
                target_p.data = self.current_tau * target_p.data + \
                                (1 - self.current_tau) * online_p.data


class CustomBYOL(nn.Module):
    def __init__(self,
                 encoder="vit",
                 num_classes=3
                 ):
        """
        Args:
            datamodule: The datamodule
            learning_rate: the learning rate
            weight_decay: optimizer weight decay
            input_height: image input height
            batch_size: the batch size
            num_workers: number of workers
            warmup_epochs: num of epochs for scheduler warm up
            max_epochs: max epochs for scheduler
        """
        super().__init__()

        self.online_network = SiameseArm(
            encoder=encoder, num_classes=num_classes)
        self.target_network = deepcopy(self.online_network)
        self.weight_callback = BYOLMAWeightUpdate()

    def on_train_batch_end(self, global_step, max_step) -> None:
        # Add callback for user automatically since it's key to BYOL weight update
        self.weight_callback.on_train_batch_end(
             self, global_step, max_step)

    def forward(self, batch, device):
        loss_a, loss_b, total_loss = self.shared_step(batch, device)

        return total_loss

    def classify(self, x):
        y = self.online_network.classify(x)
        return y

    def cosine_similarity(self, a, b):
        a = F.normalize(a, dim=-1)
        b = F.normalize(b, dim=-1)
        sim = (a * b).sum(-1).mean()
        return sim

    def shared_step(self, batch, device):
        # (img_1, img_2), y = batch
        (x1, y1), (x2, y2) = batch
        x1 = x1.float().to(device)
        x2 = x2.float().to(device)

        # Image 1 to image 2 loss
        y1, z1, h1 = self.online_network(x1)
        with torch.no_grad():
            y2, z2, h2 = self.target_network(x2)
        loss_a = 2. - 2 * self.cosine_similarity(h1, z2)

        # Image 2 to image 1 loss
        y1, z1, h1 = self.online_network(x2)
        with torch.no_grad():
            y2, z2, h2 = self.target_network(x1)
        # L2 normalize
        loss_b = 2. - 2 * self.cosine_similarity(h1, z2)

        # Final loss
        total_loss = loss_a + loss_b

        return loss_a, loss_b, total_loss




