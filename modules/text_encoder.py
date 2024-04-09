# -*- coding: utf-8 -*-
# @Time    : 2021/3/22
# @Author  : Aspen Stars
# @Contact : aspenstars@qq.com
# @FileName: text_encoder.py
import logging

import ipdb

import torch
import torch.nn as nn
import torch.nn.functional as F

import copy
import math
import numpy as np

from modules.Transformer import MultiHeadedAttention, PositionalEncoding, PositionwiseFeedForward, Encoder, \
    EncoderLayer, Embeddings, SublayerConnection, clones


class TextEncoder(nn.Module):
    def __init__(self, d_model, d_ff, num_layers, tgt_vocab, num_labels=14, h=3, dropout=0.1):
        super(TextEncoder, self).__init__()
        # TODO:
        #  将eos,pad的index改为参数输入
        self.eos_idx = 0
        self.pad_idx = 0

        attn = MultiHeadedAttention(h, d_model)
        ff = PositionwiseFeedForward(d_model, d_ff, dropout)
        position = PositionalEncoding(d_model, dropout)
        self.classifier = nn.Linear(d_model, num_labels)

        self.encoder = Encoder(EncoderLayer(d_model, attn, ff, dropout), num_layers)
        self.src_embed = nn.Sequential(Embeddings(d_model, tgt_vocab), position)
        # print("TextEncoder init completed")
        logging.info(f"TextEncoder init completed")

    def prepare_mask(self, seq):
        # 只有报告文本序列中的元素值不为0(即既不是eos也不是pad)的位置，对应的mask值才为1
        seq_mask = (seq.data != self.eos_idx) & (seq.data != self.pad_idx)
        # 需要将 seq_mask 张量的每一行的第一个位置标记为 1，以表示这个位置是序列的开始？
        seq_mask[:, 0] = 1  # bos
        # 维度变成了[batch_size, 1, seq_len]
        seq_mask = seq_mask.unsqueeze(-2) # 在倒数第2个维度上增加一个新的维度,那么新增的这个维度依旧是倒数第二个维度
        return seq_mask

    def forward(self, src):
        # src的维度是[batch_size, max_seq_length]
        src_mask = self.prepare_mask(src)
        # 调用transformer的编码器结构对报告文本进行编码
        # 编码后feats的维度是[batch_size, max_seq_length, d_model]
        feats = self.encoder(self.src_embed(src), src_mask)
        pooled_output = feats[:, 0, :] # 通过取feats的第一个位置的输出来汇总整个序列的信息(类似于BERT模型中的开始标记符用于分类任务一样)
        labels = self.classifier(pooled_output) # 使用汇总的序列信息来预测疾病标签类别
        return feats, pooled_output, labels


class MHA_FF(nn.Module):
    def __init__(self, d_model, d_ff, h, dropout=0.1):
        super(MHA_FF, self).__init__()
        self.self_attn = MultiHeadedAttention(h, d_model)
        self.sublayer = SublayerConnection(d_model, dropout)

    def forward(self, x, feats, mask=None):
        # 只是一个多头注意力机制然后施加残差连接
        # 至于类名MHA_FF中的FF(前馈神经网络)并没有看到
        # 结合memory更新的代码来看，这里的x就是memory，而feats就是编码之后的文本特征
        # 结合select_prior函数的调用来看，这里x是标签嵌入，feats是memory
        x = self.sublayer(x, lambda x: self.self_attn(x, feats, feats))
        return x
