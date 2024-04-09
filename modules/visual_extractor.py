import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from .efficientnet_pytorch.model import EfficientNet
import logging

class DenseNet121(nn.Module):
    """Model modified.

    The architecture of our model is the same as standard DenseNet121
    except the classifier layer which has an additional sigmoid function.

    """

    def __init__(self, out_size):
        super(DenseNet121, self).__init__()
        # 加载预训练的DenseNet121模型
        self.densenet121 = models.densenet121(pretrained=True)
        # 获取了DenseNet121模型的分类器层的输入特征数num_ftrs
        num_ftrs = self.densenet121.classifier.in_features
        # 将DenseNet121模型的分类器层替换为一个只有一个线性层的新分类器
        # 输出大小为out_size，即14种观察结果
        self.densenet121.classifier = nn.Sequential(
            nn.Linear(num_ftrs, out_size),
        )

    def forward(self, x):
        features = self.densenet121.features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        out = self.densenet121.classifier(out)
        return out


class VisualExtractor(nn.Module):
    def __init__(self, args):
        super(VisualExtractor, self).__init__()
        self.args = args
        # print(f"=> creating model '{args.visual_extractor}'")
        logging.info(f"creating model '{args.visual_extractor}")
        # 根据args.visual_extractor的值来创建不同的模型结构
        if args.visual_extractor == 'densenet':
            self.model = DenseNet121(args.num_labels)
            # 构建一个平均池化层，用于对输入的二维特征图进行降采样
            self.avg_fnt = torch.nn.AvgPool2d(kernel_size=7, stride=1, padding=0)
        elif args.visual_extractor == 'efficientnet':
            self.model = EfficientNet.from_pretrained('efficientnet-b0', num_classes=args.num_labels)
        elif 'resnet' in args.visual_extractor:
            self.visual_extractor = args.visual_extractor
            self.pretrained = args.visual_extractor_pretrained
            # 从torchvision的models模块获取预训练过的resnet101模型，pretrained为True，会加载预训练权重
            model = getattr(models, self.visual_extractor)(pretrained=self.pretrained)
            # model.children()返回一个包含模型所有直接子模块的迭代器，即模型resnet101中的所有层
            # 然后将resnet101模型中除了最后两层的所有层存储在modules列表中
            modules = list(model.children())[:-2]
            # *modules表示将一个列表或元组中的所有元素作为单独的参数传递给nn.Sequential函数，按顺序组合成用于特征提取的模型
            self.model = nn.Sequential(*modules)
            # 同时还有一个平均池化层和一个线性层，用于对提取的特征进行降采样和分类
            self.avg_fnt = torch.nn.AvgPool2d(kernel_size=7, stride=1, padding=0)
            self.classifier = nn.Linear(2048, args.num_labels)
        else:
            raise NotImplementedError

        # load pretrained visual extractor，即导入预训练的权重到构建好的模型当中
        if args.pretrain_cnn_file and args.pretrain_cnn_file != "":
            # 这里是导入自己的训练好的cnn模型，带实际导入过程
            # print(f'Load pretrained CNN model from: {args.pretrain_cnn_file}')
            logging.info(f"Load pretrained CNN model from: {args.pretrain_cnn_file}")
            checkpoint = torch.load(args.pretrain_cnn_file, map_location='cuda:{}'.format(args.gpu))
            self.model.load_state_dict(checkpoint['state_dict'])
        else:
            # 这里从官方导入
            # 前面加载模型的时候已经加载了预训练权重，因此从官方导入时这里只是提示一下，不进行其他操作
            # print(f'Load pretrained CNN model from: official pretrained in ImageNet')
            logging.info(f"Load pretrained CNN model from: official pretrained in ImageNet")

    def forward(self, images):
        # 对单张图像进行特征提取
        if self.args.visual_extractor == 'densenet':
            patch_feats = self.model.densenet121.features(images)
            avg_feats = self.avg_fnt(patch_feats).squeeze().reshape(-1, patch_feats.size(1))

            x = F.relu(patch_feats, inplace=True)
            x = F.adaptive_avg_pool2d(x, (1, 1))
            x = torch.flatten(x, 1)
            labels = self.model.densenet121.classifier(x)

        elif self.args.visual_extractor == 'efficientnet':
            patch_feats = self.model.extract_features(images)
            # Pooling and final linear layer
            avg_feats = self.model._avg_pooling(patch_feats)
            x = avg_feats.flatten(start_dim=1)
            x = self.model._dropout(x)
            labels = self.model._fc(x)
        elif 'resnet' in self.visual_extractor:
            patch_feats = self.model(images) # 提取图像特征，维度是[batch_size, 2048, 7, 7]
            avg_feats = self.avg_fnt(patch_feats).squeeze().reshape(-1, patch_feats.size(1)) # 对整个输入图像平均池化，维度是[batch_size, 2048]
            labels = self.classifier(avg_feats) # 将平均池化之后的特征输入到分类器中，来预测图像所属的疾病标签类别，这里维度是[batch_size, 14],即[16,14]
        else:
            raise NotImplementedError

        batch_size, feat_size, _, _ = patch_feats.shape
        # permute(0, 2, 1)：第一个维度(即0维度)保持不变，第2个维度和第三个维度交换位置
        # 因此交换位置之后，patch_feats的维度变为[batch_size, 49, 2048]
        # 即第二个维度为图像的空间维度，第三个维度为图像的特征维度，便于后续处理
        patch_feats = patch_feats.reshape(batch_size, feat_size, -1).permute(0, 2, 1)
        return patch_feats, avg_feats, labels
