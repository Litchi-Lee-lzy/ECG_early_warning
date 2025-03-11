
import torch
from torch import nn
from einops import rearrange, repeat
from einops.layers.torch import Rearrange



class Config:
    # 信号参数
    target_sample_rate = 128
    input_signal_len = 1280
    # ecg-Transformer参数2
    vit_patch_length = 64
    vit_dim = 256
    vit_dim_head = 8
    vit_depth = 8
    vit_heads = 8
    vit_mlp_dim = 256
    mae_decoder_dim = 256
    mae_masking_ratio = 0.5
    mae_masking_method = 'random'  # random,mean,block
    mae_decoder_depth = 8
    mae_decoder_heads = 8
    mae_decoder_dim_head = 8
    # 训练时的batch大小
    batch_size = 64
    lr = 1e-3
    min_lr = 1e-5
    max_epoch = 100

    # 存储参数
    output_dir = "/media/lzy/Elements SE/early_warning/pretrain_result/checkpoint/"


config = Config()


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x):
        x_list = []
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
            # x_list.append(x)
        return x


class ViT1D(nn.Module):
    def __init__(self, *, signal_lenth=config.input_signal_len, patch_lenth=config.vit_patch_length, num_classes=1000,
                 dim=config.vit_dim, depth=config.vit_depth, heads=config.vit_heads, mlp_dim=config.vit_mlp_dim,
                 pool='cls', channels=1, dim_head=config.vit_dim_head, dropout=0., emb_dropout=0.):
        super().__init__()

        assert signal_lenth % patch_lenth == 0, 'signal length must be divisible by the patch length.'

        num_patches = signal_lenth // patch_lenth
        patch_dim = channels * patch_lenth
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b (sl pl) -> b sl pl', pl=patch_lenth),  # sl-原始信号长度 pl-patch长度
            nn.Linear(patch_dim, dim)
            # nn.Conv1d(in_channels=patch_dim,out_channels=dim,kernel_size=1)
        )
        # o_o#
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        # x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]

        # x = self.to_latent(x)
        return x


class Linear_mask(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, patches, masked_indices):
        new_patches = patches
        batch = new_patches.shape[0]
        patch_length = new_patches.shape[2]
        for b in range(batch):
            masked_index = masked_indices[b]
            for index in masked_index:
                start = new_patches[b, index, 0]
                end = new_patches[b, index, -1]
                new_patch = torch.linspace(start, end, steps=patch_length, out=None)
                new_patches[b, index] = new_patch
        return new_patches


class MAE_linearmask(nn.Module):
    def __init__(
            self,
            *,
            decoder_dim=config.mae_decoder_dim,
            masking_ratio=config.mae_masking_ratio,
            masking_method=config.mae_masking_method,
            decoder_depth=config.mae_decoder_depth,
            decoder_heads=config.mae_decoder_heads,
            decoder_dim_head=config.mae_decoder_dim_head,
            pre_train='train',
            plot=False
    ):
        super().__init__()
        assert masking_ratio >= 0 and masking_ratio < 1, 'masking ratio must be kept between 0 and 1'
        self.masking_ratio = masking_ratio
        self.masking_method = masking_method
        # extract some hyperparameters and functions from encoder (vision transformer to be trained)
        self.pre_train = pre_train
        self.encoder = ViT1D()
        encoder = self.encoder
        num_patches, encoder_dim = encoder.pos_embedding.shape[-2:]

        self.to_patch, self.patch_to_emb = encoder.to_patch_embedding[:2]  # to_patch-切片，patch_to_emb-切片嵌入
        pixel_values_per_patch = self.patch_to_emb.weight.shape[-1]  # ？

        # decoder parameters
        self.enc_to_dec = nn.Linear(encoder_dim, decoder_dim) if encoder_dim != decoder_dim else nn.Identity()
        self.mask_token = nn.Parameter(torch.randn(decoder_dim))
        self.cls_token = nn.Parameter(torch.randn(encoder_dim))
        self.decoder = Transformer(dim=decoder_dim, depth=decoder_depth, heads=decoder_heads, dim_head=decoder_dim_head,
                                   mlp_dim=decoder_dim * 4)
        self.decoder_pos_emb = nn.Embedding(num_patches, decoder_dim)
        self.to_pixels = nn.Linear(decoder_dim, pixel_values_per_patch)
        # self.to_pixels = nn.Conv1d(decoder_dim,pixel_values_per_patch,kernel_size=1)
        # self.loss_funtion = nn.MSELoss(reduce=None, size_average=False)
        self.plot = plot

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        # print(x.shape)
        len_keep = int(L * (1 - mask_ratio))
        # print(len_keep)
        # 按概率选择随机掩码还是next seg 掩码
        if torch.rand(1) > 0:
            noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        else:
            sequence = torch.arange(L)
            noise = sequence.repeat(N, 1).to(x.device)
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        # print(torch.sort(ids_shuffle, dim=1))
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        ids_mask = ids_shuffle[:, len_keep:]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore, ids_mask, ids_keep

    def loss_function(self,
                      pred,
                      input):

        # print(vq_loss)
        # print(self.calculate_cos_loss(recons, input))
        recons_loss = nn.MSELoss()(pred, input)
        loss = recons_loss
        return {'loss': loss, }

    def calculate_cos_loss(self, rec, target):
        # 计算每一个片段的cos sim
        target = target / (target.norm(dim=-1, keepdim=True) + 1e-11)
        rec = rec / (rec.norm(dim=-1, keepdim=True) + 1e-11)
        rec_loss = (1 - (target * rec).sum(-1)).mean()

        return rec_loss


    def getEmbeding(self, signal):

        with torch.no_grad():
            patches = self.to_patch(signal)
            batch, num_patches, *_ = patches.shape

            # random mask
            tokens = self.patch_to_emb(patches)
            tokens = tokens + self.encoder.pos_embedding[:, 1:(num_patches + 1)]
            # x, mask, ids_restore, ids_mask, ids_keep = self.random_masking(tokens, self.masking_ratio)
            cls_token = self.cls_token + self.encoder.pos_embedding[:, :1, :]
            cls_tokens = cls_token.expand(tokens.shape[0], -1, -1)
            x = torch.cat((cls_tokens, tokens), dim=1)
            encoded_tokens = self.encoder.transformer(x)[:, 1:, :]
        return encoded_tokens

    def forward(self, signal, train=True):
        device = signal.device
        # get patches
        patches = self.to_patch(signal)
        batch, num_patches, *_ = patches.shape
        all_indices = torch.arange(num_patches + 1, device=device).repeat(batch, 1)
        batch_range = torch.arange(batch, device=device)[:, None]
        # random mask
        tokens = self.patch_to_emb(patches)
        tokens = tokens + self.encoder.pos_embedding[:, 1:(num_patches + 1)]
        x, mask, ids_restore, ids_mask, ids_keep = self.random_masking(tokens, self.masking_ratio)
        cls_token = self.cls_token + self.encoder.pos_embedding[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        encoded_tokens = self.encoder.transformer(x)

        # encoded_tokens = all_encoded_tokens[-1]

        x = self.enc_to_dec(encoded_tokens)

        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token
        # reapply decoder position embedding to unmasked tokens

        decoder_tokens = x + self.decoder_pos_emb(all_indices)

        decoded_tokens = self.decoder(decoder_tokens)
        # decoded_tokens = all_decoded_tokens[-1]
        decoded_tokens = decoded_tokens[:, 1:, :]
        # splice out the mask tokens and project to pixel values

        masked_indices = ids_mask
        unmasked_indices = ids_keep
        mask_tokens = decoded_tokens[batch_range, masked_indices]
        pred_pixel_values = self.to_pixels(decoded_tokens)

        pred_pixel_values_masked = pred_pixel_values[batch_range, masked_indices]
        pred_pixel_values_unmasked = pred_pixel_values[batch_range, unmasked_indices]
        masked_patches = patches[batch_range, masked_indices]
        unmasked_patches = patches[batch_range, unmasked_indices]
        if self.plot:
            return patches, pred_pixel_values, masked_indices

        if self.pre_train == 'train':
            return self.loss_function(pred_pixel_values_masked, masked_patches), \
                   self.loss_function(pred_pixel_values_unmasked, unmasked_patches)





if __name__ == "__main__":
    pass
