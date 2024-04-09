import logging

import ipdb

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from modules.visual_extractor import VisualExtractor
from modules.Transformer import TransformerModel
from modules.text_encoder import TextEncoder, MHA_FF


class _LAMRG(nn.Module):
    def __init__(self, args, tokenizer):
        super(_LAMRG, self).__init__()
        self.args = args
        self.tokenizer = tokenizer
        self.visual_extractor = VisualExtractor(args)
        self.encoder_decoder = TransformerModel(args, tokenizer)
        self.proj = nn.Linear(args.num_labels, args.d_vf)

        self._init_weight(self.proj)  # 手动初始化，权重经过了处理，初始的偏置全部变成0
        # print("_LAMRG init completed")
        logging.info(f"_LAMRG init completed")

    @staticmethod
    def _init_weight(f):
        nn.init.kaiming_normal_(f.weight)
        f.bias.data.fill_(0)

    def __str__(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + '\nTrainable parameters: {}'.format(params)

    def forward_iu_xray(self, images):
        # images: (bsz, 2, 3, 224, 224)，即一个批次的两张图片数据
        # att_feats_0表示提取的图像特征，fc_feats_0表示提取的图像特征的平均值(即平均池化之后的结果)，labels_0表示使用图像特征平均值预测的图像标签
        att_feats_0, fc_feats_0, labels_0 = self.visual_extractor(
            images[:, 0])  # 提取第一张图片的特征，传入的图像的维度为(bsz, 3, 224, 224)
        att_feats_1, fc_feats_1, labels_1 = self.visual_extractor(
            images[:, 1])  # 提取第二张图片的特征，传入的图像的维度为(bsz, 3, 224, 224)

        # fc_feats是两张图片综合的平均池化特征，维度是(16, 2048)
        fc_feats = torch.mean(torch.stack([fc_feats_0, fc_feats_1]), dim=0)  # 将两张图片的平均池化特征进行平均，得到的结果作为最终的图像特征
        # att_feats是两张图片综合的特征，维度是(16, 98, 2048)
        # 含义：16条数据，每条数据98个像素区域，每个区域的特征维度是2048
        att_feats = torch.cat((att_feats_0, att_feats_1), dim=1)  # 将两张图片的特征在空间维度(即第1维度)进行拼接
        # out_labels是两张图片综合的标签(平均之后的标签作为对当前患者的疾病标签的预测结果)，维度是(16, 14)
        out_labels = torch.mean(torch.stack([labels_0, labels_1]), dim=0)  # 将两张图片的标签在第0维度上进行平均
        return att_feats, fc_feats, out_labels

    def forward_mimic_cxr(self, images):
        att_feats, fc_feats, out_labels = self.visual_extractor(images)
        return att_feats, fc_feats, out_labels

    def forward(self, images, targets=None, labels=None, mode='train'):
        if self.args.dataset_name == 'iu_xray':
            att_feats, fc_feats, out_labels = self.forward_iu_xray(images)
        else:
            att_feats, fc_feats, out_labels = self.forward_mimic_cxr(images)

        label_feats = self.proj(out_labels).unsqueeze(1)
        att_feats = torch.cat((att_feats, label_feats), dim=1)
        if mode == 'train':
            output = self.encoder_decoder(fc_feats, att_feats, targets, mode='forward')
        elif mode == 'sample':
            output, _ = self.encoder_decoder(fc_feats, att_feats, opt=self.args, mode='sample')
        else:
            raise ValueError
        return output, out_labels


class LAMRGModel(_LAMRG):
    def __init__(self, args, tokenizer):
        super(LAMRGModel, self).__init__(args, tokenizer)

        self.m = nn.Parameter(torch.FloatTensor(1, args.num_labels, 40, args.d_vf))
        self.init_m()

        if args.dataset_name == 'iu_xray':
            self.forward = self.forward_iu_xray
        else:
            self.forward = self.forward_mimic_cxr

    def init_m(self):
        nn.init.normal_(self.m, 0, 1 / self.args.num_labels)

    def forward_iu_xray(self, images, targets=None, labels=None, mode='train'):
        att_feats_0, fc_feats_0, labels_0 = self.visual_extractor(images[:, 0])
        att_feats_1, fc_feats_1, labels_1 = self.visual_extractor(images[:, 1])

        fc_feats = torch.cat((fc_feats_0, fc_feats_1), dim=1)
        att_feats = torch.cat((att_feats_0, att_feats_1), dim=1)

        out_labels = labels_0

        bs, nf, d_f = att_feats.shape
        _, n_l, n_m, d_f = self.m.shape
        m = labels[:, :, None, None] * self.m.expand(bs, n_l, n_m, d_f)
        m = m.reshape(bs, -1, d_f)
        att_feats = torch.cat((att_feats, m), dim=1)

        if mode == 'train':
            output = self.encoder_decoder(fc_feats, att_feats, targets, mode='forward')
        elif mode == 'sample':
            output, _ = self.encoder_decoder(fc_feats, att_feats, mode='sample')
        else:
            raise ValueError
        return output, out_labels

    def forward_mimic_cxr(self, images, targets=None, labels=None, mode='train'):

        att_feats, fc_feats, out_labels = self.visual_extractor(images)

        bs, nf, d_f = att_feats.shape
        _, n_l, n_m, d_f = self.m.shape
        m = labels[:, :, None, None] * self.m.expand(bs, n_l, n_m, d_f)
        m = m.reshape(bs, -1, d_f)
        att_feats = torch.cat((att_feats, m), dim=1)

        if mode == 'train':
            output = self.encoder_decoder(fc_feats, att_feats, targets, mode='forward')
        elif mode == 'sample':
            output, _ = self.encoder_decoder(fc_feats, att_feats, mode='sample')
        else:
            raise ValueError
        return output, out_labels


class BasicModel(LAMRGModel):
    def forward_iu_xray(self, images, targets=None, labels=None, mode='train'):
        att_feats_0, fc_feats_0, labels_0 = self.visual_extractor(images[:, 0])
        att_feats_1, fc_feats_1, labels_1 = self.visual_extractor(images[:, 1])

        fc_feats = torch.cat((fc_feats_0, fc_feats_1), dim=1)
        att_feats = torch.cat((att_feats_0, att_feats_1), dim=1)

        out_labels = labels_0

        if mode == 'train':
            output = self.encoder_decoder(fc_feats, att_feats, targets, mode='forward')
        elif mode == 'sample':
            output, _ = self.encoder_decoder(fc_feats, att_feats, mode='sample')
        else:
            raise ValueError
        return output, out_labels

    def forward_mimic_cxr(self, images, targets=None, labels=None, mode='train'):

        att_feats, fc_feats, out_labels = self.visual_extractor(images)

        if mode == 'train':
            output = self.encoder_decoder(fc_feats, att_feats, targets, mode='forward')
        elif mode == 'sample':
            output, _ = self.encoder_decoder(fc_feats, att_feats, mode='sample')
        else:
            raise ValueError
        return output, out_labels


class LAMRGModel_v7(_LAMRG):
    """直接将visual_extractor输出的label concat到visual feature后"""

    def __init__(self, args, tokenizer):
        super(LAMRGModel_v7, self).__init__(args, tokenizer)

    def forward(self, images, targets=None, labels=None, mode='train'):
        if self.args.dataset_name == 'iu_xray':
            att_feats, fc_feats, out_labels = self.forward_iu_xray(images)
        else:
            att_feats, fc_feats, out_labels = self.forward_mimic_cxr(images)

        label_feats = self.proj(out_labels).unsqueeze(1)
        att_feats = torch.cat((att_feats, label_feats), dim=1)
        if mode == 'train':
            output = self.encoder_decoder(fc_feats, att_feats, targets, mode='forward')
        elif mode == 'sample':
            output, _ = self.encoder_decoder(fc_feats, att_feats, opt=self.args, mode='sample')
        else:
            raise ValueError
        return output, out_labels


class LAMRGModel_v8(_LAMRG):
    def __init__(self, args, tokenizer):
        super(LAMRGModel_v8, self).__init__(args, tokenizer)
        self.args = args
        self.d_model = args.d_model
        self.d_ff = args.d_ff
        self.num_layers = args.num_layers
        self.num_labels = args.num_labels
        self.tgt_vocab = len(tokenizer.idx2token) + 1
        self.h = args.num_heads
        self.num_slots = args.num_slots
        self.d_vf = args.d_vf
        self.dropout = args.dropout

        self.txt_encoder = TextEncoder(self.d_model, self.d_ff, self.num_layers, self.tgt_vocab, self.num_labels,
                                       self.h, self.dropout)
        self.prior_memory = MHA_FF(self.d_model, self.d_ff, self.h, self.dropout)
        self.select_prior = MHA_FF(self.d_model, self.d_ff, self.h, self.dropout)
        self.memory = self.init_memory()

        self.proj_label = nn.Linear(args.num_labels, args.d_model)
        self.proj_att = nn.Linear(args.d_vf, args.d_model)
        self.proj_feat = nn.Linear(args.d_model, args.d_vf)
        self.init_weight_()

    def init_weight_(self):
        nn.init.kaiming_normal_(self.proj_label.weight)
        self.proj_label.bias.data.fill_(0)
        nn.init.kaiming_normal_(self.proj_att.weight)
        self.proj_att.bias.data.fill_(0)

    def init_memory(self):
        memory = nn.Parameter(torch.eye(self.num_slots, device='cuda').unsqueeze(0))
        if self.d_model > self.num_slots:
            diff = self.d_model - self.num_slots
            pad = torch.zeros((1, self.num_slots, diff), device='cuda')
            memory = torch.cat([memory, pad], -1)
        elif self.d_model < self.num_slots:
            memory = memory[:, :, :self.d_model]
        return memory

    def forward(self, images, targets=None, labels=None, mode='train'):
        if self.args.dataset_name == 'iu_xray':
            att_feats, fc_feats, out_labels = self.forward_iu_xray(images)
        else:
            att_feats, fc_feats, out_labels = self.forward_mimic_cxr(images)

        bsz = att_feats.shape[0]
        memory = self.memory.expand(bsz, self.num_slots, self.d_model)
        if mode == 'train':
            txt_feats, z_txt, txt_labels = self.txt_encoder(targets)
            memory = self.prior_memory(memory, txt_feats)

        label_feats = self.proj_label(out_labels).unsqueeze(1)
        prior = self.select_prior(label_feats, memory)
        att_feats = torch.cat((att_feats, self.proj_feat(prior)), dim=1)

        if mode == 'train':
            output = self.encoder_decoder(fc_feats, att_feats, targets, mode='forward')
        elif mode == 'sample':
            output, _ = self.encoder_decoder(fc_feats, att_feats, opt=self.args, mode='sample')
        else:
            raise ValueError
        return output, out_labels


class LAMRGModel_v9(_LAMRG):
    def __init__(self, args, tokenizer):
        super(LAMRGModel_v9, self).__init__(args, tokenizer)
        self.args = args
        self.d_model = args.d_model
        self.d_ff = args.d_ff
        self.num_layers = args.num_layers
        self.num_labels = args.num_labels
        self.tgt_vocab = len(tokenizer.idx2token) + 1
        self.h = args.num_heads
        self.num_slots = args.num_slots  # the number of memory slots(插槽)?
        self.d_vf = args.d_vf
        self.dropout = args.dropout

        self.txt_encoder = TextEncoder(self.d_model, self.d_ff, self.num_layers, self.tgt_vocab, self.num_labels,
                                       self.h, self.dropout)
        self.prior_memory = MHA_FF(self.d_model, self.d_ff, self.h, self.dropout)  # 此处应该代表论文中的积累的知识库？
        self.select_prior = MHA_FF(self.d_model, self.d_ff, self.h, self.dropout)  # 这里具体啥意思还不清楚？
        self.memory = self.init_memory()  # v9版本的初始化操作没有初始化mask矩阵，只有记忆矩阵

        self.linear_mem = nn.Linear(self.d_model, self.d_model)
        self.linear_label = nn.Linear(args.num_labels, args.d_model)
        self.linear_feat = nn.Linear(args.d_model, args.d_vf)
        self.linear_fcfeat = nn.Linear(args.d_vf, args.d_model)
        self.init_weight_()
        # print("LAMRGModel_v9 init completed")
        logging.info(f"LAMRGModel_v9 init completed")

    def init_weight_(self):
        nn.init.kaiming_normal_(self.linear_mem.weight)
        self.linear_mem.bias.data.fill_(0)
        nn.init.kaiming_normal_(self.linear_label.weight)
        self.linear_label.bias.data.fill_(0)
        nn.init.kaiming_normal_(self.linear_feat.weight)
        self.linear_feat.bias.data.fill_(0)

    def init_memory(self):
        # # 创建了(num_slots,num_slots)的单位矩阵,并将维度变成(1, num_slots, num_slots)
        memory = nn.Parameter(torch.eye(self.num_slots, device='cuda').unsqueeze(0))  # 记忆矩阵或者说知识库本质是一个可训练的参数矩阵
        if self.d_model > self.num_slots:
            diff = self.d_model - self.num_slots  # 获取差值
            pad = torch.zeros((1, self.num_slots, diff), device='cuda')
            memory = torch.cat([memory, pad], -1)  # 在memory的最后一维上添加零，使维度变成：(1, num_slots, d_model)
        elif self.d_model < self.num_slots:
            memory = memory[:, :, :self.d_model]  # 将memory的最后一维截断为d_model，使维度变成：(1, num_slots, d_model)
        return memory

    def forward(self, images, targets=None, labels=None, mode='train'):
        if self.args.dataset_name == 'iu_xray':
            att_feats, fc_feats, vis_labels = self.forward_iu_xray(images)
        else:
            att_feats, fc_feats, vis_labels = self.forward_mimic_cxr(images)

        z_img = self.linear_fcfeat(fc_feats)

        bsz = att_feats.shape[0]
        # memory = self.linear_mem(self.memory).expand(bsz, -1, -1)
        memory = self.memory.expand(bsz, -1, -1)
        if mode == 'train':
            txt_feats, z_txt, txt_labels = self.txt_encoder(targets)
            memory = self.prior_memory(memory, txt_feats)

        label_feats = self.linear_label(vis_labels).unsqueeze(1)
        prior = self.select_prior(label_feats, memory)
        att_feats = torch.cat((att_feats, self.linear_feat(prior)), dim=1)

        if mode == 'train':
            output = self.encoder_decoder(fc_feats, att_feats, targets, mode='forward')
            return (output, vis_labels, txt_labels, z_img, z_txt)
        elif mode == 'sample':
            # ipdb.set_trace()
            output, _ = self.encoder_decoder(fc_feats, att_feats, opt=self.args, mode='sample')
            return output, vis_labels
        else:
            raise ValueError


class LAMRGModel_v10(LAMRGModel_v9):
    def forward(self, images, targets=None, labels=None, mode='train'):
        if self.args.dataset_name == 'iu_xray':
            att_feats, fc_feats, vis_labels = self.forward_iu_xray(images)
        else:
            att_feats, fc_feats, vis_labels = self.forward_mimic_cxr(images)

        z_img = self.linear_fcfeat(fc_feats)

        bsz = att_feats.shape[0]
        memory = self.linear_mem(self.memory).expand(bsz, -1, -1)
        # memory = self.memory.expand(bsz, -1, -1)
        if mode == 'train':
            txt_feats, z_txt, txt_labels = self.txt_encoder(targets)
            memory = self.prior_memory(memory, txt_feats)

        label_feats = self.linear_label(vis_labels).unsqueeze(1)
        prior = self.select_prior(label_feats, memory)
        att_feats = torch.cat((att_feats, self.linear_feat(prior)), dim=1)

        if mode == 'train':
            output = self.encoder_decoder(fc_feats, att_feats, targets, mode='forward')
            return (output, vis_labels, txt_labels, z_img, z_txt)
        elif mode == 'sample':
            # ipdb.set_trace()
            output, _ = self.encoder_decoder(fc_feats, att_feats, opt=self.args, mode='sample')
            return output, vis_labels
        else:
            raise ValueError


class LAMRGModel_v11(LAMRGModel_v9):
    def __init__(self, args, tokenizer):
        super(LAMRGModel_v11, self).__init__(args, tokenizer)
        self.args = args
        self.d_model = args.d_model
        self.d_ff = args.d_ff
        self.num_layers = args.num_layers
        self.num_labels = args.num_labels
        self.tgt_vocab = len(tokenizer.idx2token) + 1
        self.h = args.num_heads
        self.num_slots = args.num_slots
        self.d_vf = args.d_vf
        self.dropout = args.dropout

        self.txt_encoder = TextEncoder(self.d_model, self.d_ff, self.num_layers, self.tgt_vocab, self.num_labels,
                                       self.h, self.dropout)
        self.update_memory = MHA_FF(self.d_model, self.d_ff, self.h, self.dropout)
        self.select_prior = MHA_FF(self.d_model, self.d_ff, self.h, self.dropout)
        self.memory, self.mask = self.init_memory()

        self.linear_z = nn.Linear(args.d_vf, args.d_model)
        self.linear_label = nn.Linear(args.d_model, args.num_labels)
        self.query_mem = nn.Linear(self.d_model, self.d_model)
        self.linear_feat = nn.Linear(args.d_model, args.d_vf)
        self.init_weight()

    def init_weight(self):
        self._init_weight(self.linear_z)
        self._init_weight(self.linear_label)
        self._init_weight(self.query_mem)
        self._init_weight(self.linear_feat)

    def init_memory(self):
        memory = nn.Parameter(torch.eye(self.num_slots).unsqueeze(0))
        if self.d_model > self.num_slots:
            diff = self.d_model - self.num_slots
            pad = torch.zeros((1, self.num_slots, diff))
            memory = torch.cat([memory, pad], -1)
        elif self.d_model < self.num_slots:
            memory = memory[:, :, :self.d_model]
        mask = torch.ones((self.num_slots, self.d_model))
        mask[:, self.num_slots:] = 0
        return memory, mask

    def forward(self, images, targets=None, labels=None, mode='train'):
        bsz = images.shape[0]

        ve = self.forward_iu_xray if self.args.dataset_name == 'iu_xray' else self.forward_mimic_cxr
        att_feats, avg_feats, _ = ve(images)
        z_img = self.linear_z(avg_feats)
        vis_labels = self.linear_label(z_img)

        memory = self.query_mem(self.memory.to(images)).expand(bsz, -1, -1)
        mask = self.mask.to(images).expand(bsz, -1, -1)
        if mode == 'train':
            txt_feats, z_txt, txt_labels = self.txt_encoder(targets)
            memory = self.update_memory(memory, txt_feats, mask)

        prior = self.select_prior(z_img, memory)
        att_feats = torch.cat((att_feats, self.linear_feat(prior)), dim=1)

        if mode == 'train':
            output = self.encoder_decoder(avg_feats, att_feats, targets, mode='forward')
            return (output, vis_labels, txt_labels, z_img, z_txt)
        elif mode == 'sample':
            # ipdb.set_trace()
            output, _ = self.encoder_decoder(avg_feats, att_feats, opt=self.args, mode='sample')
            return output, vis_labels
        else:
            raise ValueError


class LAMRGModel_v12(LAMRGModel_v9):
    def __init__(self, args, tokenizer):
        super(LAMRGModel_v12, self).__init__(args, tokenizer)
        self.args = args
        self.d_model = args.d_model
        self.d_ff = args.d_ff
        self.num_layers = args.num_layers
        self.num_labels = args.num_labels
        self.tgt_vocab = len(tokenizer.idx2token) + 1
        self.h = args.num_heads
        self.num_slots = args.num_slots
        self.d_vf = args.d_vf
        self.dropout = args.dropout

        self.txt_encoder = TextEncoder(self.d_model, self.d_ff, self.num_layers, self.tgt_vocab, self.num_labels,
                                       self.h, self.dropout)
        self.update_memory = MHA_FF(self.d_model, self.d_ff, args.num_memory_heads, self.dropout)
        self.select_prior = MHA_FF(self.d_model, self.d_ff, args.num_memory_heads, self.dropout)
        self.memory, self.mask = self.init_memory()

        self.get_mem = nn.Linear(self.d_model, self.d_model)
        self.linear_z = nn.Linear(self.d_vf, self.d_model)
        self.linear_feat = nn.Linear(self.d_model, self.d_vf)

        self.classifier = nn.Linear(self.d_model, self.num_labels)
        self.embed_labels = nn.Linear(1, self.d_model)

        self.init_weight()
        # print("LAMRGModel_v12 init completed")
        logging.info(f"LAMRGModel_v12 init completed")

    def init_weight(self):
        self._init_weight(self.linear_z)
        self._init_weight(self.get_mem)
        self._init_weight(self.linear_feat)
        self._init_weight(self.classifier)
        self._init_weight(self.embed_labels)

    def init_memory(self):
        # 创建了(num_slots,num_slots)的单位矩阵,并将维度变成(1, num_slots, num_slots)
        memory = nn.Parameter(torch.eye(self.num_slots).unsqueeze(0))  # 记忆矩阵或者说知识库本质是一个可训练的参数矩阵
        if self.d_model > self.num_slots:
            diff = self.d_model - self.num_slots  # 获取差值
            pad = torch.zeros((1, self.num_slots, diff))
            memory = torch.cat([memory, pad], -1)  # 在memory的最后一维上添加零，使维度变成：(1, num_slots, d_model)
        elif self.d_model < self.num_slots:
            memory = memory[:, :, :self.d_model]  # 将memory的最后一维截断为d_model，使维度变成：(1, num_slots, d_model)
        mask = torch.ones((self.num_slots, self.d_model))  # (num_slots, d_model)的全1矩阵
        mask[:, self.num_slots:] = 0  # 其中前num_slots列为1，其余为0
        return memory, mask

    def forward(self, images, targets=None, labels=None, mode='train'):
        bsz = images.shape[0]  # batch size

        ve = self.forward_iu_xray if self.args.dataset_name == 'iu_xray' else self.forward_mimic_cxr  # 选择前向计算的函数
        # 返回的att_feats和avg_feats都是综合了两张图像的特征之后的结果
        # 这里只接收了att_feats和avg_feats，而没有接收out_labels
        att_feats, avg_feats, _ = ve(images)  # 传入图像，进行特征提取

        z_img = self.linear_z(avg_feats)
        vis_labels = self.classifier(z_img)
        # images的维度是[bsz, 2, 3, 224, 224]
        # self.memory.to(images)中的to方法是类型转换方法，将memory转换成images的数据类型(包括转换device属性)
        # 然后获取memory的线性变换结果，维度是[1, 60, 512]，并将其维度扩展成(bsz, num_slots, d_model)
        # expand(bsz, -1, -1)：在第一维上扩展bsz倍，第二维和第三维不变
        memory = self.get_mem(self.memory.to(images)).expand(bsz, -1, -1)
        # 对mask进行同样的操作：类型转换+线性变换+维度扩展
        mask = self.mask.to(images).expand(bsz, -1, -1)
        if mode == 'train':
            # txt_feats表示编码之后的文本特征，维度是[batch_size, max_seq_length, d_model]
            # z_txt表示文本特征的汇总，即开始符这个位置的特征，维度是[batch_size, d_model]
            # txt_labels表示用z_txt预测的疾病标签，维度是[batch_size, num_labels]
            txt_feats, z_txt, txt_labels = self.txt_encoder(targets)
            # 更新时memory的mask没有派上用场，因为mask的作用是将memory的后面的维度置为0，而更新时是直接将txt_feats加到memory上？
            memory = self.update_memory(memory, txt_feats, mask) # 更新前后的memory维度都是[batch_size, num_slots, d_model]
        # 先让标签张量在最后增加一个维度，变为[batch_size, num_labels, 1]，因为embed_labels的输入要求维度是1
        # 然后经过embed_labels(即线性变换)之后，维度变为[batch_size, num_labels, d_model]
        emb_labels = self.embed_labels(vis_labels.unsqueeze(-1)) # 对标签进行嵌入，维度是[batch_size, num_labels, d_model]
        # prior表示注意过memory的标签嵌入，维度与emb_labels相同，都是[batch_size, num_labels, d_model]
        prior = self.select_prior(emb_labels, memory) # 这句的效果：将emb_labels和memory进行多头注意力，让emb_labels去注意先前的memory
        # 将注意过memory的标签嵌入和提取的图像特征连接起来，作为最终的图像特征，所谓的注意过memory的视觉特征，应该是对应论文中的visual-knowledge attention
        # att_feats的维度变成[16, 98+14, 2048]
        att_feats = torch.cat((att_feats, self.linear_feat(prior)), dim=1)

        if mode == 'train':
            output = self.encoder_decoder(avg_feats, att_feats, targets, mode='forward')
            return (output, vis_labels, txt_labels, z_img, z_txt)
        elif mode == 'sample':
            # output是基于视觉特征，用束搜索算法生成的文本序列，维度是[batch_size, max_seq_length]
            # 第二个返回的参数是对应的概率，没有继续使用
            output, _ = self.encoder_decoder(avg_feats, att_feats, opt=self.args, mode='sample')
            return output, vis_labels
        else:
            raise ValueError


class LAMRGModel_v91(LAMRGModel_v12):
    """Ablation Study
        只用label的模型
    """

    def forward(self, images, targets=None, labels=None, mode='train'):
        bsz = images.shape[0]

        ve = self.forward_iu_xray if self.args.dataset_name == 'iu_xray' else self.forward_mimic_cxr
        att_feats, avg_feats, _ = ve(images)

        z_img = self.linear_z(avg_feats)
        vis_labels = self.classifier(z_img)

        txt_feats, z_txt, txt_labels = None, None, None
        # memory = self.get_mem(self.memory.to(images)).expand(bsz, -1, -1)
        # mask = self.mask.to(images).expand(bsz, -1, -1)
        # if mode == 'train':
        # txt_feats, z_txt, txt_labels = self.txt_encoder(targets)
        # memory = self.update_memory(memory, txt_feats, mask)

        # emb_labels = self.embed_labels(vis_labels.unsqueeze(-1))
        # prior = self.select_prior(emb_labels, memory)
        # att_feats = torch.cat((att_feats, self.linear_feat(prior)), dim=1)

        if mode == 'train':
            output = self.encoder_decoder(avg_feats, att_feats, targets, mode='forward')
            return (output, vis_labels, txt_labels, z_img, z_txt)
        elif mode == 'sample':
            output, _ = self.encoder_decoder(avg_feats, att_feats, opt=self.args, mode='sample')
            return output, vis_labels
        else:
            raise ValueError


class LAMRGModel_v92(LAMRGModel_v12):
    """Ablation Study
        用label loss + rank loss的模型
    """

    def forward(self, images, targets=None, labels=None, mode='train'):
        bsz = images.shape[0]

        ve = self.forward_iu_xray if self.args.dataset_name == 'iu_xray' else self.forward_mimic_cxr
        att_feats, avg_feats, _ = ve(images)

        z_img = self.linear_z(avg_feats)
        vis_labels = self.classifier(z_img)

        # memory = self.get_mem(self.memory.to(images)).expand(bsz, -1, -1)
        # mask = self.mask.to(images).expand(bsz, -1, -1)
        if mode == 'train':
            txt_feats, z_txt, txt_labels = self.txt_encoder(targets)
            # memory = self.update_memory(memory, txt_feats, mask)

        # emb_labels = self.embed_labels(vis_labels.unsqueeze(-1))
        # prior = self.select_prior(emb_labels, memory)
        # att_feats = torch.cat((att_feats, self.linear_feat(prior)), dim=1)

        if mode == 'train':
            output = self.encoder_decoder(avg_feats, att_feats, targets, mode='forward')
            return (output, vis_labels, txt_labels, z_img, z_txt)
        elif mode == 'sample':
            # ipdb.set_trace()
            output, _ = self.encoder_decoder(avg_feats, att_feats, opt=self.args, mode='sample')
            return output, vis_labels
        else:
            raise ValueError


class LAMRGModel_vRebuttal(LAMRGModel_v9):
    def __init__(self, args, tokenizer):
        super(LAMRGModel_vRebuttal, self).__init__(args, tokenizer)
        self.args = args
        self.d_model = args.d_model
        self.d_ff = args.d_ff
        self.num_layers = args.num_layers
        self.num_labels = args.num_labels
        self.tgt_vocab = len(tokenizer.idx2token) + 1
        self.h = args.num_heads
        self.num_slots = args.num_slots
        self.d_vf = args.d_vf
        self.dropout = args.dropout

        self.txt_encoder = TextEncoder(self.d_model, self.d_ff, self.num_layers, self.tgt_vocab, self.num_labels,
                                       self.h, self.dropout)

        self.memory, self.mask = self.init_memory()

        self.get_mem = nn.Linear(self.d_model, self.d_model)
        self.linear_z = nn.Linear(self.d_vf, self.d_model)
        self.linear_feat = nn.Linear(self.d_model, self.d_vf)

        self.classifier = nn.Linear(self.d_model, self.num_labels)
        self.embed_labels = nn.Linear(1, self.d_model)

        self.init_weight()

    @staticmethod
    def attention(query, key, value):
        "Compute 'Dot Product Attention'"
        import math

        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1))
        p_attn = F.softmax(scores, dim=-1)
        return torch.matmul(p_attn, value)

    def init_weight(self):
        self._init_weight(self.linear_z)
        self._init_weight(self.get_mem)
        self._init_weight(self.linear_feat)
        self._init_weight(self.classifier)
        self._init_weight(self.embed_labels)

    def init_memory(self):
        memory = nn.Parameter(torch.eye(self.num_slots).unsqueeze(0))
        if self.d_model > self.num_slots:
            diff = self.d_model - self.num_slots
            pad = torch.zeros((1, self.num_slots, diff))
            memory = torch.cat([memory, pad], -1)
        elif self.d_model < self.num_slots:
            memory = memory[:, :, :self.d_model]
        mask = torch.ones((self.num_slots, self.d_model))
        mask[:, self.num_slots:] = 0
        return memory, mask

    def forward(self, images, targets=None, labels=None, mode='train'):
        bsz = images.shape[0]

        ve = self.forward_iu_xray if self.args.dataset_name == 'iu_xray' else self.forward_mimic_cxr
        att_feats, avg_feats, _ = ve(images)

        z_img = self.linear_z(avg_feats)
        vis_labels = self.classifier(z_img)
        # ipdb.set_trace()
        memory = self.get_mem(self.memory.to(images)).expand(bsz, -1, -1)
        if mode == 'train':
            txt_feats, z_txt, txt_labels = self.txt_encoder(targets)
            memory = self.attention(memory, txt_feats, txt_feats)

        emb_labels = self.embed_labels(vis_labels.unsqueeze(-1))
        prior = self.attention(emb_labels, memory, memory)
        att_feats = torch.cat((att_feats, self.linear_feat(prior)), dim=1)

        if mode == 'train':
            output = self.encoder_decoder(avg_feats, att_feats, targets, mode='forward')
            return (output, vis_labels, txt_labels, z_img, z_txt)
        elif mode == 'sample':
            output, _ = self.encoder_decoder(avg_feats, att_feats, opt=self.args, mode='sample')
            return output, vis_labels
        else:
            raise ValueError
