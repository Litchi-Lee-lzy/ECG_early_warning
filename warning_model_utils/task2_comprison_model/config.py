# -*- coding: utf-8 -*-
import os


# todo experi
# pl48 m50 tianchi ccdd
# pl48 m25 tianchi ccdd
class Config:
    onserver = True
    server = 'ecnu'  # ecnu local juchi
    debug = False
    task = "segment"  # pre_train segment classifier
    # fine-tune segment
    freeze = True

    pre_train_ckpt = 'None'

    #信号参数
    target_sample_rate = 500
    input_signal_len = 5000
    # 模型参数2
    # patch的个数必须满足是10的倍数
    vit_patch_length = 50
    vit_patch_num = input_signal_len // vit_patch_length
    vit_dim = 512
    vit_dim_head = 6
    vit_depth = 12
    vit_heads = 8
    vit_mlp_dim = 512
    mae_decoder_dim = 512
    mae_masking_ratio = 0.3
    mae_masking_method = 'QRS'  # random,mean,block
    mae_decoder_depth = 12
    mae_decoder_heads = 8
    mae_decoder_dim_head = 6
    # 在第几个epoch进行到下一个state,调整lr为lr/=lr_decay
    stage_epoch = [16, 24, 40, 64, 80, 128]
    # 训练时的batch大小
    batch_size = 256
    if not task == 'pre_train':
        batch_size = 64
    # 最大训练多少个epoch
    max_epoch = 1000





config = Config()

