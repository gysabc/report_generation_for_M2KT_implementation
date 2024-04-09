# This file contains Att2in2, AdaAtt, AdaAttMO, UpDown model

# AdaAtt is from Knowing When to Look: Adaptive Attention via A Visual Sentinel for Image Captioning
# https://arxiv.org/abs/1612.01887
# AdaAttMO is a modified version with maxout lstm

# Att2in is from Self-critical Sequence Training for Image Captioning
# https://arxiv.org/abs/1612.00563
# In this file we only have Att2in2, which is a slightly different version of att2in,
# in which the img feature embedding and word embedding is the same as what in adaatt.

# UpDown is from Bottom-Up and Top-Down Attention for Image Captioning and VQA
# https://arxiv.org/abs/1707.07998
# However, it may not be identical to the author's architecture.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from . import utils
from torch.nn.utils.rnn import PackedSequence, pack_padded_sequence, pad_packed_sequence

from .CaptionModel import CaptionModel

bad_endings = ['a', 'an', 'the', 'in', 'for', 'at', 'of', 'with', 'before', 'after', 'on', 'upon', 'near', 'to', 'is',
               'are', 'am']
bad_endings += ['the']


def sort_pack_padded_sequence(input, lengths):
    sorted_lengths, indices = torch.sort(lengths, descending=True)
    tmp = pack_padded_sequence(input[indices], sorted_lengths, batch_first=True)
    inv_ix = indices.clone()
    inv_ix[indices] = torch.arange(0, len(indices)).type_as(inv_ix)
    return tmp, inv_ix


def pad_unsort_packed_sequence(input, inv_ix):
    tmp, _ = pad_packed_sequence(input, batch_first=True)
    tmp = tmp[inv_ix]
    return tmp


def pack_wrapper(module, att_feats, att_masks):
    # module是一个embedding层(本质上是一个线性层+dropout层)
    if att_masks is not None:
        packed, inv_ix = sort_pack_padded_sequence(att_feats, att_masks.data.long().sum(1))
        return pad_unsort_packed_sequence(PackedSequence(module(packed[0]), packed[1]), inv_ix)
    else:
        # att_masks默认是None，因此不会执行if语句中的语句，直接返回module(att_feats)
        return module(att_feats)


class AttModel(CaptionModel):
    def __init__(self, opt, tokenizer):
        super(AttModel, self).__init__()
        self.tokenizer = tokenizer
        self.vocab_size = len(tokenizer.idx2token)
        self.input_encoding_size = opt.d_model
        # self.rnn_type = opt.rnn_type
        self.rnn_size = opt.d_ff
        self.num_layers = opt.num_layers
        self.drop_prob_lm = opt.drop_prob_lm
        # 获取opt对象的'max_seq_length'属性值，如果该属性不存在，则返回默认值20
        # or运算符：如果第一个表达式的值为True，则返回第一个表达式的值；否则返回第二个表达式的值
        # 本项目中未设置seq_length参数，但是因为getattr(opt, 'max_seq_length', 20)返回值不会是None，因此opt.seq_length没有使用到，也不会报错
        self.seq_length = getattr(opt, 'max_seq_length', 20) or opt.seq_length  # maximum sample length
        self.fc_feat_size = opt.d_vf # 全连接层特征的大小，即输入到注意力模型的全连接层的特征向量的维度
        self.att_feat_size = opt.d_vf # 注意力特征的大小，即输入到注意力模型的卷积层特征图的通道数
        self.att_hid_size = opt.d_model # 注意力模型中隐藏层的大小，即注意力模型中使用的前馈神经网络的隐藏层的维度

        self.bos_idx = getattr(opt, 'bos_idx', 0)
        self.eos_idx = getattr(opt, 'eos_idx', 0)
        self.pad_idx = getattr(opt, 'pad_idx', 0)

        self.use_bn = getattr(opt, 'use_bn', 0) # whether to use batch normalization

        self.ss_prob = 0.0  # Schedule sampling probability,详见https://cloud.tencent.com/developer/article/1081168

        # For remove bad ending
        self.vocab = self.tokenizer
        self.bad_endings_ix = [int(k) for k, v in self.vocab.idx2token.items() if v in bad_endings] # 在生成句子时需要避免的单词的索引
        logging.info(f"AttModel init completed")
        # print("AttModel init completed")

    def clip_att(self, att_feats, att_masks):
        # Clip the length of att_masks and att_feats to the maximum length
        if att_masks is not None:
            max_len = att_masks.data.long().sum(1).max()
            att_feats = att_feats[:, :max_len].contiguous()
            att_masks = att_masks[:, :max_len].contiguous()
        return att_feats, att_masks

    def get_logprobs_state(self, it, fc_feats, att_feats, p_att_feats, att_masks, state, output_logsoftmax=1):
        '''
        :param it: 是一个序列，其中的元素都是开始符(即0)，维度是(16,)
        :param fc_feats: 综合了两张图片特征之后的平均池化特征(即对两张图片的平均池化特征求平均)，但维度是(16,0)
        :param att_feats: 提取出来的图像特征(综合了两张图片的特征，且注意过memory)，维度是(16, 112, 0)
        :param p_att_feats: 注意过`memory`的视觉特征经过编码器堆栈的计算结果，即编码器堆栈的输出memory，维度是(16,112,512)
        :param att_masks: 维度是[16,1,112]，是att_feats的掩码张量
        :param state: 是一个空列表
        :param output_logsoftmax: 默认是1
        '''

        # 'it' contains a word index
        # self.embed就是原先用于处理视觉特征的匿名函数，不做任何处理的
        xt = self.embed(it)

        # output相当于根据视觉特征去预测报告文本序列开始符的下一个词是什么，维度是[16, 512]
        # state存放了一个张量，该张量相当于目标序列，仅包含开始符，维度是[1, 16, 1]
        output, state = self.core(xt, fc_feats, att_feats, p_att_feats, state, att_masks)
        if output_logsoftmax:
            logprobs = F.log_softmax(self.logit(output), dim=1)
        else:
            logprobs = self.logit(output)

        return logprobs, state

    def _sample_beam(self, fc_feats, att_feats, att_masks=None, opt={}):
        # fc_feats：综合了两张图片特征之后的平均池化特征(即对两张图片的平均池化特征求平均)，维度是(16,2048)
        # att_feats：提取出来的图像特征(综合了两张图片的特征，且注意过memory)，维度是(16,112,2048)
        # opt：是一开始运行模型的时候使用到的相关参数
        # att_masks一直是None
        beam_size = getattr(opt, 'beam_size', 10)
        group_size = getattr(opt, 'group_size', 1)
        sample_n = getattr(opt, 'sample_n', 10)
        # when sample_n == beam_size then each beam is a sample.
        assert sample_n == 1 or sample_n == beam_size // group_size, 'when beam search, sample_n == 1 or beam search'
        batch_size = fc_feats.size(0)
        # 对对`fc_feats`和`att_feats`进行一些处理，包括：
        # ①对视觉特征进行embedding操作(本质上是一个线性层+dropout层)，以及生成了一个视觉特征的掩码张量
        # ②调用transformer模型的EncoderDecoder类，但只进行编码得到编码器堆栈的输出
        # 返回的时候对视觉特征进行了处理，因此：
        # p_fc_feats：维度是(16,0)；p_att_feats：维度是(16, 112, 0)；pp_att_feats：即编码器堆栈的输出memory，维度是(16,112,512)；
        # p_att_masks：维度是[16,1,112]，是p_att_feats的掩码张量
        p_fc_feats, p_att_feats, pp_att_feats, p_att_masks = self._prepare_feature(fc_feats, att_feats, att_masks) # 传入的att_masks一直是None

        assert beam_size <= self.vocab_size + 1, 'lets assume this for now, otherwise this corner case causes a few headaches down the road. can be dealt with in future if needed'
        # 初始化报告文本序列seq以及对应的log概率seqLogprobs，后面会将束搜索预测的结果存放到这两个变量中
        seq = fc_feats.new_full((batch_size * sample_n, self.seq_length), self.pad_idx, dtype=torch.long)
        seqLogprobs = fc_feats.new_zeros(batch_size * sample_n, self.seq_length, self.vocab_size + 1)
        # lets process every image independently for now, for simplicity

        self.done_beams = [[] for _ in range(batch_size)] # 用于存放处理结果
        # self.init_hidden就返回一个空的列表，没有任何操作
        state = self.init_hidden(batch_size)

        # first step, feed bos
        # it应该是初始化一个序列，其中的元素都是开始符(即0)，维度是(16,)
        it = fc_feats.new_full([batch_size], self.bos_idx, dtype=torch.long)
        # 在已有视觉特征、视觉特征掩码的情况下，对初始化了的仅有开始符的目标序列进行预测，预测下一个词，
        # 最后返回模型的概率输出logprobs和初始化的仅有开始符的目标序列state
        # logprobs的维度是[16, 761]，state是一个列表，其中有一个Tensor，的维度是[1, 16, 1]
        logprobs, state = self.get_logprobs_state(it, p_fc_feats, p_att_feats, pp_att_feats, p_att_masks, state)

        # 根据beam_size的大小将p_fc_feats、p_att_feats、pp_att_feats、p_att_masks重复beam_size次
        # p_fc_feats：维度变成是(48,0)；p_att_feats：维度变成是(48, 112, 0)；pp_att_feats：即编码器堆栈的输出memory，维度变成是(48,112,512)；
        # p_att_masks：维度变成[48,1,112]，是p_att_feats的掩码张量
        p_fc_feats, p_att_feats, pp_att_feats, p_att_masks = utils.repeat_tensors(beam_size,
                                                                                  [p_fc_feats, p_att_feats,
                                                                                   pp_att_feats, p_att_masks]
                                                                                  )
        # done_beams(束搜索的结果)，是一个长度为16(一批次)的列表，包含了预测的报告文本序列以及概率等信息，具体看笔记
        self.done_beams = self.beam_search(state, logprobs, p_fc_feats, p_att_feats, pp_att_feats, p_att_masks, opt=opt)
        for k in range(batch_size):
            if sample_n == beam_size:
                for _n in range(sample_n):
                    seq_len = self.done_beams[k][_n]['seq'].shape[0]
                    seq[k * sample_n + _n, :seq_len] = self.done_beams[k][_n]['seq']
                    seqLogprobs[k * sample_n + _n, :seq_len] = self.done_beams[k][_n]['logps']
            else:
                # 默认执行这里
                # 束搜索结果是3个，选择第一个束搜索结果的序列作为最终的序列
                seq_len = self.done_beams[k][0]['seq'].shape[0]
                seq[k, :seq_len] = self.done_beams[k][0]['seq']  # the first beam has highest cumulative score
                seqLogprobs[k, :seq_len] = self.done_beams[k][0]['logps']
        # return the samples and their log likelihoods
        return seq, seqLogprobs

    def _sample(self, fc_feats, att_feats, att_masks=None, opt={}):
        # fc_feats：综合了两张图片特征之后的平均池化特征(即对两张图片的平均池化特征求平均)，维度是(16,2048)
        # att_feats：提取出来的图像特征(综合了两张图片的特征，且注意过memory)，维度是(16,112,2048)
        # opt：是一开始运行模型的时候使用到的相关参数
        sample_method = getattr(opt, 'sample_method', 'greedy') # 参数设置中默认是beam_search
        beam_size = getattr(opt, 'beam_size', 1) # the beam size when beam searching,默认是3
        temperature = getattr(opt, 'temperature', 1.0) # the temperature when sampling,默认是1.0
        sample_n = int(getattr(opt, 'sample_n', 1)) # the sample number per image,默认是1
        group_size = getattr(opt, 'group_size', 1)
        output_logsoftmax = getattr(opt, 'output_logsoftmax', 1)
        decoding_constraint = getattr(opt, 'decoding_constraint', 0)
        block_trigrams = getattr(opt, 'block_trigrams', 0)
        remove_bad_endings = getattr(opt, 'remove_bad_endings', 0)
        if beam_size > 1 and sample_method in ['greedy', 'beam_search']:
            return self._sample_beam(fc_feats, att_feats, att_masks, opt)
        if group_size > 1:
            # group_size默认是1，所以不会执行这个if语句中的语句
            return self._diverse_sample(fc_feats, att_feats, att_masks, opt)

        batch_size = fc_feats.size(0)
        state = self.init_hidden(batch_size * sample_n)

        p_fc_feats, p_att_feats, pp_att_feats, p_att_masks = self._prepare_feature(fc_feats, att_feats, att_masks)

        if sample_n > 1:
            p_fc_feats, p_att_feats, pp_att_feats, p_att_masks = utils.repeat_tensors(sample_n,
                                                                                      [p_fc_feats, p_att_feats,
                                                                                       pp_att_feats, p_att_masks]
                                                                                      )

        trigrams = []  # will be a list of batch_size dictionaries

        seq = fc_feats.new_full((batch_size * sample_n, self.seq_length), self.pad_idx, dtype=torch.long)
        seqLogprobs = fc_feats.new_zeros(batch_size * sample_n, self.seq_length, self.vocab_size + 1)
        for t in range(self.seq_length + 1):
            if t == 0:  # input <bos>
                it = fc_feats.new_full([batch_size * sample_n], self.bos_idx, dtype=torch.long)

            logprobs, state = self.get_logprobs_state(it, p_fc_feats, p_att_feats, pp_att_feats, p_att_masks, state,
                                                      output_logsoftmax=output_logsoftmax)

            if decoding_constraint and t > 0:
                tmp = logprobs.new_zeros(logprobs.size())
                tmp.scatter_(1, seq[:, t - 1].data.unsqueeze(1), float('-inf'))
                logprobs = logprobs + tmp

            if remove_bad_endings and t > 0:
                tmp = logprobs.new_zeros(logprobs.size())
                prev_bad = np.isin(seq[:, t - 1].data.cpu().numpy(), self.bad_endings_ix)
                # Make it impossible to generate bad_endings
                tmp[torch.from_numpy(prev_bad.astype('uint8')), 0] = float('-inf')
                logprobs = logprobs + tmp

            # Mess with trigrams
            # Copy from https://github.com/lukemelas/image-paragraph-captioning
            if block_trigrams and t >= 3:
                # Store trigram generated at last step
                prev_two_batch = seq[:, t - 3:t - 1]
                for i in range(batch_size):  # = seq.size(0)
                    prev_two = (prev_two_batch[i][0].item(), prev_two_batch[i][1].item())
                    current = seq[i][t - 1]
                    if t == 3:  # initialize
                        trigrams.append({prev_two: [current]})  # {LongTensor: list containing 1 int}
                    elif t > 3:
                        if prev_two in trigrams[i]:  # add to list
                            trigrams[i][prev_two].append(current)
                        else:  # create list
                            trigrams[i][prev_two] = [current]
                # Block used trigrams at next step
                prev_two_batch = seq[:, t - 2:t]
                mask = torch.zeros(logprobs.size(), requires_grad=False).to(logprobs.device)  # batch_size x vocab_size
                for i in range(batch_size):
                    prev_two = (prev_two_batch[i][0].item(), prev_two_batch[i][1].item())
                    if prev_two in trigrams[i]:
                        for j in trigrams[i][prev_two]:
                            mask[i, j] += 1
                # Apply mask to log probs
                # logprobs = logprobs - (mask * 1e9)
                alpha = 2.0  # = 4
                logprobs = logprobs + (mask * -0.693 * alpha)  # ln(1/2) * alpha (alpha -> infty works best)

            # sample the next word
            if t == self.seq_length:  # skip if we achieve maximum length
                break
            it, sampleLogprobs = self.sample_next_word(logprobs, sample_method, temperature)

            # stop when all finished
            if t == 0:
                unfinished = it != self.eos_idx
            else:
                it[~unfinished] = self.pad_idx  # This allows eos_idx not being overwritten to 0
                logprobs = logprobs * unfinished.unsqueeze(1).to(logprobs)
                unfinished = unfinished & (it != self.eos_idx)
            seq[:, t] = it
            seqLogprobs[:, t] = logprobs
            # quit loop if all sequences have finished
            if unfinished.sum() == 0:
                break

        return seq, seqLogprobs

    def _diverse_sample(self, fc_feats, att_feats, att_masks=None, opt={}):

        sample_method = getattr(opt, 'sample_method', 'greedy')
        beam_size = getattr(opt, 'beam_size', 1)
        temperature = getattr(opt, 'temperature', 1.0)
        group_size = getattr(opt, 'group_size', 1)
        diversity_lambda = getattr(opt, 'diversity_lambda', 0.5)
        decoding_constraint = getattr(opt, 'decoding_constraint', 0)
        block_trigrams = getattr(opt, 'block_trigrams', 0)
        remove_bad_endings = getattr(opt, 'remove_bad_endings', 0)

        batch_size = fc_feats.size(0)
        state = self.init_hidden(batch_size)

        p_fc_feats, p_att_feats, pp_att_feats, p_att_masks = self._prepare_feature(fc_feats, att_feats, att_masks)

        trigrams_table = [[] for _ in range(group_size)]  # will be a list of batch_size dictionaries

        seq_table = [fc_feats.new_full((batch_size, self.seq_length), self.pad_idx, dtype=torch.long) for _ in
                     range(group_size)]
        seqLogprobs_table = [fc_feats.new_zeros(batch_size, self.seq_length) for _ in range(group_size)]
        state_table = [self.init_hidden(batch_size) for _ in range(group_size)]

        for tt in range(self.seq_length + group_size):
            for divm in range(group_size):
                t = tt - divm
                seq = seq_table[divm]
                seqLogprobs = seqLogprobs_table[divm]
                trigrams = trigrams_table[divm]
                if t >= 0 and t <= self.seq_length - 1:
                    if t == 0:  # input <bos>
                        it = fc_feats.new_full([batch_size], self.bos_idx, dtype=torch.long)
                    else:
                        it = seq[:, t - 1]  # changed

                    logprobs, state_table[divm] = self.get_logprobs_state(it, p_fc_feats, p_att_feats, pp_att_feats,
                                                                          p_att_masks, state_table[divm])  # changed
                    logprobs = F.log_softmax(logprobs / temperature, dim=-1)

                    # Add diversity
                    if divm > 0:
                        unaug_logprobs = logprobs.clone()
                        for prev_choice in range(divm):
                            prev_decisions = seq_table[prev_choice][:, t]
                            logprobs[:, prev_decisions] = logprobs[:, prev_decisions] - diversity_lambda

                    if decoding_constraint and t > 0:
                        tmp = logprobs.new_zeros(logprobs.size())
                        tmp.scatter_(1, seq[:, t - 1].data.unsqueeze(1), float('-inf'))
                        logprobs = logprobs + tmp

                    if remove_bad_endings and t > 0:
                        tmp = logprobs.new_zeros(logprobs.size())
                        prev_bad = np.isin(seq[:, t - 1].data.cpu().numpy(), self.bad_endings_ix)
                        # Impossible to generate remove_bad_endings
                        tmp[torch.from_numpy(prev_bad.astype('uint8')), 0] = float('-inf')
                        logprobs = logprobs + tmp

                    # Mess with trigrams
                    if block_trigrams and t >= 3:
                        # Store trigram generated at last step
                        prev_two_batch = seq[:, t - 3:t - 1]
                        for i in range(batch_size):  # = seq.size(0)
                            prev_two = (prev_two_batch[i][0].item(), prev_two_batch[i][1].item())
                            current = seq[i][t - 1]
                            if t == 3:  # initialize
                                trigrams.append({prev_two: [current]})  # {LongTensor: list containing 1 int}
                            elif t > 3:
                                if prev_two in trigrams[i]:  # add to list
                                    trigrams[i][prev_two].append(current)
                                else:  # create list
                                    trigrams[i][prev_two] = [current]
                        # Block used trigrams at next step
                        prev_two_batch = seq[:, t - 2:t]
                        mask = torch.zeros(logprobs.size(), requires_grad=False).cuda()  # batch_size x vocab_size
                        for i in range(batch_size):
                            prev_two = (prev_two_batch[i][0].item(), prev_two_batch[i][1].item())
                            if prev_two in trigrams[i]:
                                for j in trigrams[i][prev_two]:
                                    mask[i, j] += 1
                        # Apply mask to log probs
                        # logprobs = logprobs - (mask * 1e9)
                        alpha = 2.0  # = 4
                        logprobs = logprobs + (mask * -0.693 * alpha)  # ln(1/2) * alpha (alpha -> infty works best)

                    it, sampleLogprobs = self.sample_next_word(logprobs, sample_method, 1)

                    # stop when all finished
                    if t == 0:
                        unfinished = it != self.eos_idx
                    else:
                        unfinished = (seq[:, t - 1] != self.pad_idx) & (seq[:, t - 1] != self.eos_idx)
                        it[~unfinished] = self.pad_idx
                        unfinished = unfinished & (it != self.eos_idx)  # changed
                    seq[:, t] = it
                    seqLogprobs[:, t] = sampleLogprobs.view(-1)

        return torch.stack(seq_table, 1).reshape(batch_size * group_size, -1), torch.stack(seqLogprobs_table,
                                                                                           1).reshape(
            batch_size * group_size, -1)
