import torch
from torch import nn
from warning_model_utils.pretrain_model import ViT1D, Transformer
from torch.nn import functional as F
class taskConfig:
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
    # mae_decoder_dim = 128
    # mae_masking_ratio = 0.5
    # mae_masking_method = 'random'  # random,mean,block
    # mae_decoder_depth = 2
    # mae_decoder_heads = 8
    # mae_decoder_dim_head = 8
    #
    mae_decoder_dim = 256
    mae_masking_ratio = 0.5
    mae_masking_method = 'random'  # random,mean,block
    mae_decoder_depth = 8
    mae_decoder_heads = 8
    mae_decoder_dim_head = 8







config = taskConfig()


class indicator_cls_model(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        # extract some hyperparameters and functions from encoder (vision transformer to be trained)
        self.encoder = ViT1D()
        encoder = self.encoder
        self.num_classes = num_classes
        self.to_patch, self.patch_to_emb = encoder.to_patch_embedding[:2]  # to_patch-切片，patch_to_emb-切片嵌入

        self.rnn = nn.LSTM(input_size=config.vit_dim, hidden_size=256, batch_first=True, bidirectional=True)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.Classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )
        self.class_center = nn.Parameter(torch.zeros(num_classes, 512), requires_grad=False)


    def update_center(self, center):
        for i in range(self.num_classes):
            self.class_center[i] = torch.mean(center[i], dim=0)


    def getStatusDistance(self, signal):
        with torch.no_grad():
            latentFeature = self.getEmbedding(signal)
        dis_list = []
        for i in range(self.num_classes):
            distance_status0 = torch.norm(latentFeature - self.class_center[i], p=2, dim=1)
            dis_list.append(distance_status0)
        return dis_list

    def getInfoDistance(self, signal):

        with torch.no_grad():
            T = torch.tensor(100.)
            latentFeature = self.getEmbedding(signal)
            distance_status0 = torch.exp(torch.matmul(latentFeature, self.class_center[0].t()) / T)

            distance_status1 = torch.exp(torch.matmul(latentFeature, self.class_center[1].t()) / T)

            prob_info = distance_status1 / (distance_status1 + distance_status0)

        return prob_info


    def getEmbedding(self, signal):
        patches = self.to_patch(signal)
        batch, num_patches, *_ = patches.shape
        # random mask
        tokens = self.patch_to_emb(patches)
        tokens = tokens + self.encoder.pos_embedding[:, 1:(num_patches + 1)]
        # x, mask, ids_restore, ids_mask, ids_keep = self.random_masking(tokens, self.masking_ratio)
        # cls_token = self.cls_token + self.encoder.pos_embedding[:, :1, :]
        # cls_tokens = cls_token.expand(tokens.shape[0], -1, -1)
        # x = torch.cat((cls_tokens, tokens), dim=1)
        encoded_tokens = self.encoder.transformer(tokens)[:, 1:, :]
        lstm_output, (final_hidden_state, final_cell_state) = self.rnn(encoded_tokens)
        latentFeature = self.avgpool(lstm_output.permute(0, 2, 1))
        latentFeature = latentFeature.view(latentFeature.size(0), -1)
        return latentFeature

    def forward(self, signal):
        patches = self.to_patch(signal)
        batch, num_patches, *_ = patches.shape
        tokens = self.patch_to_emb(patches)
        tokens = tokens + self.encoder.pos_embedding[:, 1:(num_patches + 1)]
        # attend with vision transformer
        encoded_tokens = self.encoder.transformer(tokens)
        lstm_output, (final_hidden_state, final_cell_state) = self.rnn(encoded_tokens)
        latentFeature = self.avgpool(lstm_output.permute(0, 2, 1))
        latentFeature = latentFeature.view(latentFeature.size(0), -1)
        logit = self.Classifier(latentFeature)
        return logit


class indicator_clu_model(nn.Module):
    def __init__(self,
                 decoder_dim=config.mae_decoder_dim,

                 decoder_depth=config.mae_decoder_depth,
                 decoder_heads=config.mae_decoder_heads,
                 decoder_dim_head=config.mae_decoder_dim_head, ):
        super().__init__()
        # extract some hyperparameters and functions from encoder (vision transformer to be trained)
        self.encoder = ViT1D()
        encoder = self.encoder
        self.masking_ratio = 0.5
        num_patches, encoder_dim = encoder.pos_embedding.shape[-2:]
        self.cls_token = nn.Parameter(torch.randn(encoder_dim))
        self.to_patch, self.patch_to_emb = encoder.to_patch_embedding[:2]  # to_patch-切片，patch_to_emb-切片嵌入
        pixel_values_per_patch = self.patch_to_emb.weight.shape[-1]  # ？
        # print(pixel_values_per_patch)
        # decoder parameters
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.hidden_dim = 128
        self.num_layers = 1
        self.rnn = nn.LSTM(input_size=config.vit_dim, hidden_size=self.hidden_dim, batch_first=True, bidirectional=True)
        # self.enc_to_dec_new = nn.Sequential(
        #     nn.Linear(encoder_dim, 2),
        #     nn.Linear(2, decoder_dim)
        # )
        self.enc_to_dec_new = nn.Linear(self.hidden_dim * 2, 20 * decoder_dim)


        # self.mask_token = nn.Parameter(torch.randn(decoder_dim))
        # self.cls_token = nn.Parameter(torch.randn(encoder_dim))
        self.decoder_new = Transformer(dim=decoder_dim, depth=decoder_depth, heads=decoder_heads, dim_head=decoder_dim_head,
                                   mlp_dim=decoder_dim * 4)
        self.decoder_pos_emb_new = nn.Embedding(num_patches, decoder_dim)
        self.to_pixels_new = nn.Linear(decoder_dim, pixel_values_per_patch)

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
        noise = torch.rand(N, L // 2, device=x.device)  # noise in [0, 1]
        noise = torch.repeat_interleave(noise, repeats=2, dim=1)
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

        return x_masked, mask, ids_restore, ids_mask


    def loss_function(self,
                      pred,
                      input, train=True):
        # print(vq_loss)
        # print(self.calculate_cos_loss(recons, input))
        recons_loss = nn.MSELoss()(pred, input)
        if train:
            cos_sim_loss = torch.tensor(0., device=input.device)
        else:
            cos_sim_loss = self.calculate_cos_loss(pred, input)

        loss = recons_loss + cos_sim_loss
        return {'loss': loss,
                'Reconstruction_Loss': recons_loss,
                'Cos_sim_Loss': cos_sim_loss,
                }

    def calculate_cos_loss(self, rec, target):
        # 计算每一个片段的cos sim
        target = target / (target.norm(dim=-1, keepdim=True) + 1e-11)
        rec = rec / (rec.norm(dim=-1, keepdim=True) + 1e-7)
        rec_loss = (1 - (target * rec).sum(-1)).mean()

        return rec_loss

    def reconstruction(self, signal):
        device = signal.device
        patches = self.to_patch(signal)
        batch, num_patches, *_ = patches.shape
        h0 = torch.zeros(self.num_layers * 2, signal.size(0), self.hidden_dim).to(signal.device)  # 乘以2因为是双向
        c0 = torch.zeros(self.num_layers * 2, signal.size(0), self.hidden_dim).to(signal.device)
        # random mask
        tokens = self.patch_to_emb(patches)
        tokens = tokens + self.encoder.pos_embedding[:, 1:(num_patches + 1)]

        cls_token = self.cls_token + self.encoder.pos_embedding[:, :1, :]
        cls_tokens = cls_token.expand(tokens.shape[0], -1, -1)
        tokens = torch.cat((cls_tokens, tokens), dim=1)
        encoded_tokens = self.encoder.transformer(tokens)
        encoded_tokens = encoded_tokens[:, 1:, :]
        # lstm_output, (final_hidden_state, final_cell_state) = self.rnn(encoded_tokens)
        # encoded_tokens_avg = self.avgpool(lstm_output.permute(0, 2, 1)).view(encoded_tokens.size(0), -1)
        lstm_output, (final_hidden_state, final_cell_state) = self.rnn(encoded_tokens, (h0, c0))
        # out = torch.cat((lstm_output[:, -1, :self.hidden_dim], lstm_output[:, 0, self.hidden_dim:]), dim=1)
        out = self.avgpool(lstm_output.permute(0, 2, 1)).view(encoded_tokens.size(0), -1)
        x = self.enc_to_dec_new(out).view(encoded_tokens.size(0), 20, -1)
        all_indices = torch.arange(1, num_patches + 1, device=device).repeat(batch, 1)
        decoder_tokens = x + self.decoder_pos_emb_new(all_indices)
        decoded_tokens = self.decoder_new(decoder_tokens)
        # decoded_tokens = decoded_tokens[:, 1:, :]

        pred_pixel_values = self.to_pixels_new(decoded_tokens)

        return pred_pixel_values.view(batch, -1)

    def forward(self, signal, train=True):
        device = signal.device
        patches = self.to_patch(signal)
        batch, num_patches, *_ = patches.shape
        h0 = torch.zeros(self.num_layers * 2, signal.size(0), self.hidden_dim).to(signal.device)  # 乘以2因为是双向
        c0 = torch.zeros(self.num_layers * 2, signal.size(0), self.hidden_dim).to(signal.device)
        # random mask
        tokens = self.patch_to_emb(patches)
        tokens = tokens + self.encoder.pos_embedding[:, 1:(num_patches + 1)]

        cls_token = self.cls_token + self.encoder.pos_embedding[:, :1, :]
        cls_tokens = cls_token.expand(tokens.shape[0], -1, -1)
        tokens = torch.cat((cls_tokens, tokens), dim=1)
        encoded_tokens = self.encoder.transformer(tokens)
        encoded_tokens = encoded_tokens[:, 1:, :]

        lstm_output, (final_hidden_state, final_cell_state) = self.rnn(encoded_tokens, (h0, c0))
        # out = torch.cat((lstm_output[:, -1, :self.hidden_dim], lstm_output[:, 0, self.hidden_dim:]), dim=1)
        out = self.avgpool(lstm_output.permute(0, 2, 1)).view(encoded_tokens.size(0), -1)
        # encoded_tokens_avg = self.avgpool(lstm_output.permute(0, 2, 1)).view(encoded_tokens.size(0), -1)
        x = self.enc_to_dec_new(out).view(encoded_tokens.size(0), 20, -1)
        all_indices = torch.arange(1, num_patches + 1, device=device).repeat(batch, 1)
        decoder_tokens = x + self.decoder_pos_emb_new(all_indices)
        #
        decoded_tokens = self.decoder_new(decoder_tokens)
        # decoded_tokens = decoded_tokens[:, 1:, :]

        pred_pixel_values = self.to_pixels_new(decoded_tokens)

        return self.loss_function(pred_pixel_values, patches, train=train)



class VectorQuantizer(nn.Module):
    """
    Reference:
    [1] https://github.com/deepmind/sonnet/blob/v2/sonnet/src/nets/vqvae.py
    """

    def __init__(self,
                 num_embeddings: int,
                 embedding_dim: int,
                 beta: float = 0.25):
        super(VectorQuantizer, self).__init__()
        self.K = num_embeddings
        self.D = embedding_dim
        self.beta = beta

        self.embedding = nn.Embedding(self.K, self.D)
        self.embedding.weight.data.uniform_(-1 / self.K, 1 / self.K)

    def forward(self, latents):
        latents = latents.contiguous()
        latents_shape = latents.shape
        flat_latents = latents.view(-1, self.D)  # [BL x D]

        # Compute L2 distance between latents and embedding weights
        dist = torch.sum(flat_latents ** 2, dim=1, keepdim=True) + \
               torch.sum(self.embedding.weight ** 2, dim=1) - \
               2 * torch.matmul(flat_latents, self.embedding.weight.t())  # [BL x K]

        # Get the encoding that has the min distance
        encoding_inds = torch.argmin(dist, dim=1).unsqueeze(1)  # [BL, 1]

        # Convert to one-hot encodings
        device = latents.device
        encoding_one_hot = torch.zeros(encoding_inds.size(0), self.K, device=device)
        encoding_one_hot.scatter_(1, encoding_inds, 1)  # [BL x K]

        # Quantize the latents
        quantized_latents = torch.matmul(encoding_one_hot, self.embedding.weight)  # [BL, D]
        quantized_latents = quantized_latents.view(latents_shape)  # [B x L x D]

        # Compute the VQ Losses
        commitment_loss = F.mse_loss(quantized_latents.detach(), latents)
        embedding_loss = F.mse_loss(quantized_latents, latents.detach())

        vq_loss = commitment_loss * self.beta + embedding_loss

        # Add the residue back to the latents
        quantized_latents = latents + (quantized_latents - latents).detach()

        return quantized_latents.contiguous(), vq_loss,   # [B x D x L]


class indicator_vqvae_model(nn.Module):
    def __init__(self,
                 decoder_dim=config.mae_decoder_dim,

                 decoder_depth=config.mae_decoder_depth,
                 decoder_heads=config.mae_decoder_heads,
                 decoder_dim_head=config.mae_decoder_dim_head, ):
        super().__init__()
        # extract some hyperparameters and functions from encoder (vision transformer to be trained)
        self.encoder = ViT1D()
        encoder = self.encoder
        self.masking_ratio = 0.5
        num_patches, encoder_dim = encoder.pos_embedding.shape[-2:]
        self.cls_token = nn.Parameter(torch.randn(encoder_dim))
        self.to_patch, self.patch_to_emb = encoder.to_patch_embedding[:2]  # to_patch-切片，patch_to_emb-切片嵌入
        pixel_values_per_patch = self.patch_to_emb.weight.shape[-1]  # ？
        # print(pixel_values_per_patch)
        # decoder parameters
        self.vq_layer = VectorQuantizer(256,
                                        256,
                                        0.25)
        self.enc_to_dec_new = nn.Linear(encoder_dim, decoder_dim) if encoder_dim != decoder_dim else nn.Identity()
        # self.mask_token = nn.Parameter(torch.randn(decoder_dim))
        # self.cls_token = nn.Parameter(torch.randn(encoder_dim))
        self.decoder_new = Transformer(dim=decoder_dim, depth=decoder_depth, heads=decoder_heads, dim_head=decoder_dim_head,
                                   mlp_dim=decoder_dim * 4)
        self.decoder_pos_emb_new = nn.Embedding(num_patches, decoder_dim)
        self.to_pixels_new = nn.Linear(decoder_dim, pixel_values_per_patch)




    def loss_function(self,
                      pred,
                      input):
        # print(vq_loss)
        # print(self.calculate_cos_loss(recons, input))
        recons_loss = nn.MSELoss()(pred, input)
        cos_sim_loss = self.calculate_cos_loss(pred, input)

        loss = recons_loss + cos_sim_loss
        return {'loss': loss,
                'Reconstruction_Loss': recons_loss,
                'Cos_sim_Loss': cos_sim_loss,
                }

    def calculate_cos_loss(self, rec, target):
        # 计算每一个片段的cos sim
        target = target / (target.norm(dim=-1, keepdim=True) + 1e-11)
        rec = rec / (rec.norm(dim=-1, keepdim=True) + 1e-7)
        rec_loss = (1 - (target * rec).sum(-1)).mean()

        return rec_loss

    def reconstruction(self, signal):
        device = signal.device
        patches = self.to_patch(signal)
        batch, num_patches, *_ = patches.shape
        all_indices = torch.arange(num_patches + 1, device=device).repeat(batch, 1)
        # random mask
        tokens = self.patch_to_emb(patches)
        tokens = tokens + self.encoder.pos_embedding[:, 1:(num_patches + 1)]

        cls_token = self.cls_token + self.encoder.pos_embedding[:, :1, :]
        cls_tokens = cls_token.expand(tokens.shape[0], -1, -1)
        tokens = torch.cat((cls_tokens, tokens), dim=1)
        encoded_tokens = self.encoder.transformer(tokens)
        quantized_inputs, vq_loss = self.vq_layer(encoded_tokens)

        x = self.enc_to_dec_new(quantized_inputs)

        decoder_tokens = x + self.decoder_pos_emb_new(all_indices)
        #
        decoded_tokens = self.decoder_new(decoder_tokens)
        decoded_tokens = decoded_tokens[:, 1:, :]
        pred_pixel_values = self.to_pixels_new(decoded_tokens)


        return pred_pixel_values.view(batch, -1)

    def forward(self, signal):
        device = signal.device
        patches = self.to_patch(signal)
        batch, num_patches, *_ = patches.shape
        all_indices = torch.arange(num_patches + 1, device=device).repeat(batch, 1)
        # random mask
        tokens = self.patch_to_emb(patches)
        tokens = tokens + self.encoder.pos_embedding[:, 1:(num_patches + 1)]

        cls_token = self.cls_token + self.encoder.pos_embedding[:, :1, :]
        cls_tokens = cls_token.expand(tokens.shape[0], -1, -1)
        tokens = torch.cat((cls_tokens, tokens), dim=1)
        encoded_tokens = self.encoder.transformer(tokens)
        quantized_inputs, vq_loss = self.vq_layer(encoded_tokens)

        x = self.enc_to_dec_new(quantized_inputs)

        decoder_tokens = x + self.decoder_pos_emb_new(all_indices)
        #
        decoded_tokens = self.decoder_new(decoder_tokens)
        decoded_tokens = decoded_tokens[:, 1:, :]

        pred_pixel_values = self.to_pixels_new(decoded_tokens)
        all_loss = self.loss_function(pred_pixel_values, patches)
        loss = all_loss["loss"] + vq_loss
        all_loss["loss"] = loss
        all_loss["vq_loss"] = vq_loss
        return all_loss



class indicator_mae_model(nn.Module):
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

        recons_loss = nn.MSELoss()(pred, input)
        cos_sim_loss = self.calculate_cos_loss(pred, input)

        loss = recons_loss + cos_sim_loss
        return {'loss': loss,
                'Reconstruction_Loss': recons_loss,
                'Cos_sim_Loss': cos_sim_loss,
                }

    def calculate_cos_loss(self, rec, target):
        # 计算每一个片段的cos sim
        target = target / (target.norm(dim=-1, keepdim=True) + 1e-11)
        rec = rec / (rec.norm(dim=-1, keepdim=True) + 1e-11)
        rec_loss = (1 - (target * rec).sum(-1)).mean()

        return rec_loss

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
            return pred_pixel_values.view(batch, -1)

        if self.pre_train == 'train':
            return self.loss_function(pred_pixel_values_masked, masked_patches), \
                   self.loss_function(pred_pixel_values_unmasked, unmasked_patches)
        else:
            mask_loss = self.loss_function(pred_pixel_values_masked, masked_patches)
            unmask_loss = self.loss_function(pred_pixel_values_unmasked, unmasked_patches)
            return {"Reconstruction_Loss": mask_loss["Reconstruction_Loss"] + unmask_loss["Reconstruction_Loss"],
                    "Cos_sim_Loss": mask_loss["Cos_sim_Loss"] + unmask_loss["Cos_sim_Loss"],
                    "mask_Cos_sim_Loss": mask_loss["Cos_sim_Loss"],
                    "mask_Reconstruction_Loss": mask_loss["Reconstruction_Loss"]}




if __name__ == "__main__":
    pass
