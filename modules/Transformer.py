# This file contains Transformer network
# Most of the code is copied from http://nlp.seas.harvard.edu/2018/04/03/attention.html

# The cfg name correspondance:
# N=num_layers
# d_model=input_encoding_size
# d_ff=rnn_size
# h is always 8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from . import utils

import copy
import math
import numpy as np

from .CaptionModel import CaptionModel
from .AttModel import sort_pack_padded_sequence, pad_unsort_packed_sequence, pack_wrapper, AttModel


class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many 
    other models.
    """

    def __init__(self, encoder, decoder, src_embed, tgt_embed):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed

    def forward(self, src, tgt, src_mask, tgt_mask):
        "Take in and process masked src and target sequences."
        return self.decode(self.encode(src, src_mask), src_mask,
                           tgt, tgt_mask)

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)


def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class Encoder(nn.Module):
    "Core encoder is a stack of N layers"

    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"

    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        "Follow Figure 1 (left) for connections."
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


class Decoder(nn.Module):
    "Generic N layer decoder with masking."

    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


class DecoderLayer(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"

    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        "Follow Figure 1 (right) for connections."
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)


def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0


def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k 
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch. 
        x, self.attn = attention(query, key, value, mask=mask,
                                 dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear. 
        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)


class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class TransformerModel(AttModel):

    def make_model(self, src_vocab, tgt_vocab, N_enc=6, N_dec=6,
                   d_model=512, d_ff=2048, h=8, dropout=0.1):
        "Helper: Construct a model from hyperparameters."
        c = copy.deepcopy
        attn = MultiHeadedAttention(h, d_model, dropout)
        ff = PositionwiseFeedForward(d_model, d_ff, dropout)
        position = PositionalEncoding(d_model, dropout)
        model = EncoderDecoder(
            Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N_enc),
            Decoder(DecoderLayer(d_model, c(attn), c(attn),
                                 c(ff), dropout), N_dec),
            lambda x: x,  # nn.Sequential(Embeddings(d_model, src_vocab), c(position)),
            nn.Sequential(Embeddings(d_model, tgt_vocab), c(position))
        )

        # This was important from their code. 
        # Initialize parameters with Glorot / fan_avg.
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        return model

    def __init__(self, opt, tokenizer):
        super(TransformerModel, self).__init__(opt, tokenizer)
        self.opt = opt
        # self.config = yaml.load(open(opt.config_file))

        self.N_enc = getattr(opt, 'N_enc', opt.num_layers)  # 不存在N_enc，因此就是num_layers
        self.N_dec = getattr(opt, 'N_dec', opt.num_layers)  # 不存在N_dec，因此就是num_layers
        self.d_model = getattr(opt, 'd_model', opt.d_model)
        self.d_ff = getattr(opt, 'd_ff', opt.d_ff)  # 前馈神经网络层的维度
        self.h = getattr(opt, 'num_heads', 8)
        self.dropout = getattr(opt, 'dropout', 0.1)
        tgt_vocab = self.vocab_size + 1  # 加1是因为开始或结束或填充标记符(三者算一个符号，因为索引都是0)

        # 将输入进来的图像特征进行embedding操作(供注意力计算？)，因为默认为0且未人为指定值，因此att_embed只会进行一次batch normalization
        # *符号可以用于解包一个序列或元组
        self.att_embed = nn.Sequential(*(
                ((nn.BatchNorm1d(self.att_feat_size),) if self.use_bn else ()) +
                (nn.Linear(self.att_feat_size, self.d_model),
                 # nn.ReLU(),
                 nn.Dropout(self.dropout)) +
                ((nn.BatchNorm1d(self.d_model),) if self.use_bn == 2 else ())))

        self.embed = lambda x: x # 保证代码的模块化和重用性，因此这里的embed函数只是一个占位符  
        self.fc_embed = lambda x: x # 保证代码的模块化和重用性，因此这里的fc_embed函数只是一个占位符  
        self.logit = nn.Linear(self.d_model, tgt_vocab)

        self.model = self.make_model(0, tgt_vocab,
                                     N_enc=self.N_enc,
                                     N_dec=self.N_dec,
                                     d_model=self.d_model,
                                     d_ff=self.d_ff,
                                     h=self.h,
                                     dropout=self.dropout)

        # print("TransformerModel init completed")
        logging.info(f"TransformerModel init completed")

    def init_hidden(self, bsz):
        return []

    def _prepare_feature(self, fc_feats, att_feats, att_masks):
        # 对视觉特征进行embedding操作(本质上是一个线性层+dropout层)，以及生成了一个视觉特征的掩码张量；seq和seq_mask都是None
        att_feats, seq, att_masks, seq_mask = self._prepare_feature_forward(att_feats, att_masks)
        memory = self.model.encode(att_feats, att_masks)

        return fc_feats[..., :0], att_feats[..., :0], memory, att_masks

    def _prepare_feature_forward(self, att_feats, att_masks=None, seq=None):
        # att_feats是提取的图像的特征，且注意过memory，维度是[batch_size, 98+14, 2048]
        # seq是报告文本的序列(在词典中对应的索引值序列)，维度是[batch_size, max_seq_length]
        # do nothing, since att_masks are always None
        att_feats, att_masks = self.clip_att(att_feats, att_masks)
        # embed att_features
        # 由于att_masks是None，因此pack_wrapper就是对视觉特征进行embedding操作(本质上是一个线性层+dropout层)
        # embedding之后的维度是[batch_size, 98+14, 512]
        att_feats = pack_wrapper(self.att_embed, att_feats, att_masks)
        # get att_masks
        if att_masks is None:
            # att_masks为None时会将其所有元素都设置为1，维度是[batch_size, 98+14]
            # 98+14可以认为是图像像素序列长度，因此att_masks全为1表示所有像素都是有效的，都会考虑进来
            att_masks = att_feats.new_ones(att_feats.shape[:2], dtype=torch.long)
        att_masks = att_masks.unsqueeze(-2) # 增加维度，维度变成了[batch_size, 1 , 98+14]
        # get seq_masks
        if seq is not None:
            # 有报告文本序列就有seq_mask;这里是再一次对报告文本进行编码，所以需要再次构建报告文本的掩码张量
            # crop the last one
            # seq = seq[:,:-1]
            # 维度与seq一致，即[batch_size, max_seq_length]
            seq_mask = (seq.data != self.eos_idx) & (seq.data != self.pad_idx) # 只有报告文本序列中的元素值不为0(即既不是eos也不是pad)的位置，对应的mask值才为1
            seq_mask[:, 0] = 1  # bos

            seq_mask = seq_mask.unsqueeze(-2) # 增加一个维度，变成[batch_size, 1, max_seq_length]
            # 维度变成[batch_size, max_seq_length, max_seq_length]
            seq_mask = seq_mask & subsequent_mask(seq.size(-1)).to(seq_mask) # 对应位置相与，相同为1，不同为0

            seq_per_img = seq.shape[0] // att_feats.shape[0] # seq_per_img 表示每张图片对应的报告文本序列数量
            if seq_per_img > 1:
                # 一张图片对应多个报告文本序列时；但是这里的图片特征是综合了两张图片的特征，然后一个患者只对应一个报告序列，所以这里暂时用不上
                att_feats, att_masks = utils.repeat_tensors(seq_per_img,
                                                            [att_feats, att_masks]
                                                            )
        else:
            seq_mask = None

        return att_feats, seq, att_masks, seq_mask

    def _forward(self, fc_feats, att_feats, seq, att_masks=None):
        # att_masks are always None
        # fc_feats是两张图像的平均池化特征的平均值，维度是[batch_size, 2048]
        # att_feats是提取的图像的特征，且注意过memory，维度是[batch_size, 98+14, 2048]
        # seq是报告文本的序列(在词典中对应的索引值序列)，维度是[batch_size, max_seq_length]
        if seq.ndim == 3:  # B * seq_per_img * seq_len
            seq = seq.reshape(-1, seq.shape[2])
        # make new att_mask, seq_mask, embed att_features
        # att_feats维度变成了[batch_size, 98+14, 512]
        # seq没有变化，因此维度还是[batch_size, max_seq_length]
        # att_masks被构建出来，维度是[batch_size, 1, 98+14]
        # seq_mask被构建出来，维度是[batch_size, max_seq_length, max_seq_length]
        att_feats, seq, att_masks, seq_mask = self._prepare_feature_forward(att_feats, att_masks, seq)
        # decoder output
        # out的维度是[batch_size, max_seq_length, 512]
        out = self.model(att_feats, seq, att_masks, seq_mask)
        # project to vocab size
        # 维度变成[batch_size, max_seq_length, vocab_size+1]
        # 并进行对数softmax操作
        outputs = F.log_softmax(self.logit(out), dim=-1)
        return outputs
        # return torch.cat([_.unsqueeze(1) for _ in outputs], 1)

    def core(self, it, fc_feats_ph, att_feats_ph, memory, state, mask):
        """
        :param it: 是一个序列，其中的元素都是开始符(即0)，维度是(16,)
        :param fc_feats_ph: 综合了两张图片特征之后的平均池化特征(即对两张图片的平均池化特征求平均)，但维度是(16,0)
        :param att_feats_ph: 提取出来的图像特征(综合了两张图片的特征，且注意过memory)，维度是(16, 112, 0)
        :param memory: 注意过`memory`的视觉特征经过编码器堆栈的计算结果，即编码器堆栈的输出memory，维度是(16,112,512)
        :param mask: 维度是[16,1,112]，是att_feats的掩码张量
        :param state: 是一个空列表
        state = [ys.unsqueeze(0)]
        core函数作用是：构建了一批空的目标序列，然后使用编码器堆栈输出的视觉特征对目标序列的下一个词进行预测
        """
        if len(state) == 0:
            # 验证的时候会执行这里
            ys = it.unsqueeze(1) # ys的维度是(16,1)，元素全是0
        else:
            ys = torch.cat([state[0][0], it.unsqueeze(1)], dim=1)
        # 相当于根据视觉特征，利用模型去预测开始符的下一个词
        out = self.model.decode(memory, mask, ys, subsequent_mask(ys.size(1)).to(memory.device))
        return out[:, -1], [ys.unsqueeze(0)]
