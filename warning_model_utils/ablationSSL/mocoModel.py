import math

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from torch.optim.optimizer import Optimizer

from torch import nn
from torch.nn import functional as F

from model.ConvneXt import convnextv2_atto, convnextv2_nano
from model.pretrain_model import ViT1D


@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
                      for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output



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


class encoderModel(nn.Module):
    def __init__(self, encoder=None, out_dim=128, hidden_size=512, projector_dim=512, num_classes=5):
        super().__init__()

        self.encoder_name = encoder
        if encoder == "convnextv2_atto":
            self.frontend = convnextv2_atto(in_chans=1)
            dim = 320
        elif encoder == "convnextv2_nano":
            self.frontend = convnextv2_nano(in_chans=1)
            dim = 640
        elif "vit" in encoder:
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

        # classifier head
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.multi_label_classifier = nn.Linear(dim, num_classes)

    def forward(self, x):
        if self.encoder_name == "vit":
            y = self.frontend(x)
            y = y.permute(0, 2, 1)
        else:
            y = self.frontend.forward_features(x)
            y = y.squeeze(2)

        y = self.pooler(y)
        y = y.view(y.size(0), -1)
        z = self.projector(y)
        return y, z

    def classify(self, x):

        if self.encoder_name == "vit":
            y = self.frontend(x)
            y = y.permute(0, 2, 1)
        else:
            y = self.frontend(x)
            y = y.squeeze(2)
        y = self.avgpool(y)
        y = y.view(y.size(0), -1)
        y = self.multi_label_classifier(y)
        return y


class CustomMoCo(nn.Module):

    def __init__(self,
                 encoder="vit",
                 num_classes = 5,
                 emb_dim: int = 128,
                 num_negatives: int = 65536,
                 encoder_momentum: float = 0.999,
                 softmax_temperature: float = 0.07,
                 learning_rate: float = 0.03,
                 momentum: float = 0.9,
                 weight_decay: float = 1e-6,
                ):

        super(CustomMoCo, self).__init__()
        self.encoder_name = encoder
        self.encoder_q, self.encoder_k = self.init_encoders(self.encoder_name, num_classes)
        self.emb_dim = emb_dim
        self.num_negatives = num_negatives
        self.encoder_momentum = encoder_momentum
        self.softmax_temperature = softmax_temperature
        self.momentum = momentum
        self.weight_decay = weight_decay


        # create the encoders
        # num_classes is the output fc dimension


        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the queue
        self.register_buffer("queue", torch.randn(emb_dim, num_negatives))
        self.queue = nn.functional.normalize(self.queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        # self.warmup_epochs = config["warm_up"]
    def init_encoders(self, base_encoder, num_classes=5):
        """
        Override to add your own encoders
        """

        encoder_q = encoderModel(encoder=base_encoder, num_classes=num_classes)
        encoder_k = encoderModel(encoder=base_encoder, num_classes=num_classes)

        return encoder_q, encoder_k

    def exclude_from_wt_decay(self, named_params, weight_decay, skip_list=['bias', 'bn']):
        params = []
        excluded_params = []

        for name, param in named_params:
            if not param.requires_grad:
                continue
            elif any(layer_name in name for layer_name in skip_list):
                excluded_params.append(param)
            else:
                params.append(param)

        return [
            {'params': params, 'weight_decay': weight_decay},
            {'params': excluded_params, 'weight_decay': 0.}
        ]

    def training_step(self, img_q, img_k):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            logits, targets
        """

        # compute query features
        q = self.encoder_q(img_q)[1] # queries: NxC
        q = nn.functional.normalize(q, dim=1)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            # shuffle for making use of BN
            # if self.use_ddp or self.use_ddp2:
            #     img_k, idx_unshuffle = self._batch_shuffle_ddp(img_k)

            k = self.encoder_k(img_k)[1]  # keys: NxC
            k = nn.functional.normalize(k, dim=1)

            # undo shuffle
            # if self.use_ddp or self.use_ddp2:
            #     k = self._batch_unshuffle_ddp(k, idx_unshuffle)

        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1

        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        # negative logits: NxK
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])

        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        logits /= self.softmax_temperature

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long)
        labels = labels.type_as(logits)

        # dequeue and enqueue
        self._dequeue_and_enqueue(k)

        return logits, labels

    def forward(self, batch, device):
        (img_1, _), (img_2, _) = batch
        output, target = self.training_step(img_q=img_1.float().to(device), img_k=img_2.float().to(device))
        loss = F.cross_entropy(output.float(), target.long())

        return loss

    def classify(self, x):
        y = self.encoder_q.classify(x)
        return y


    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        # gather keys before updating queue
        # if self.use_ddp or self.use_ddp2:
        #     keys = concat_all_gather(keys)

        batch_size = keys.shape[0]
        # import pdb
        # pdb.set_trace()
        ptr = int(self.queue_ptr)

        # replace the keys at ptr (dequeue and enqueue)
        remainder = self.queue[:, ptr:ptr + batch_size].shape[1]
        if remainder < batch_size:
            self.queue[:, -remainder:] = keys.T[:, :remainder]
            self.queue[:, :batch_size - remainder] = keys.T[:, remainder:]
            ptr = batch_size - remainder
        else:
            self.queue[:, ptr:ptr + batch_size] = keys.T
            ptr = (ptr + batch_size) % self.num_negatives  # move pointer

        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):  # pragma: no-cover
        """
        Batch shuffle, for making use of BatchNorm.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # random shuffle index
        idx_shuffle = torch.randperm(batch_size_all).cuda()

        # broadcast to all gpus
        torch.distributed.broadcast(idx_shuffle, src=0)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        # shuffled index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):  # pragma: no-cover
        """
        Undo batch shuffle.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this]

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            em = self.encoder_momentum
            param_k.data = param_k.data * em + param_q.data * (1. - em)





