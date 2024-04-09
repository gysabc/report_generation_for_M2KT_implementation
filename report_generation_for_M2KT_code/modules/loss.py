import torch
import torch.nn as nn


class LanguageModelCriterion(nn.Module):
    def __init__(self):
        super(LanguageModelCriterion, self).__init__()

    def forward(self, input, target, mask):
        # truncate to the same size(应该指的是根据input的大小，去截断target和mask)
        # 第一个参数input，即output[:, :-1]是模型预测的结果(去掉了结束符)，维度是(16,59,761)
        # 第二个参数target，即reports_ids[:, 1:]是真实的报告文本(去掉了开始符)，维度是(16,59)
        # 第三个参数mask，即reports_masks[:, 1:]是真实的报告文本的mask(去掉了开始符)，维度是(16,59)
        target = target[:, :input.size(1)] # 截断前后没有变化(因为传进来的时候进行了处理了，input.size(1)就是target.size(1))
        mask = mask[:, :input.size(1)] # 截断前后没有变化(因为传进来的时候进行了处理了，input.size(1)就是mask.size(1))
        output = -input.gather(2, target.long().unsqueeze(2)).squeeze(2) * mask # 计算交叉熵损失(具体看笔记)
        output = torch.sum(output) / torch.sum(mask)

        return output


class LabelSmoothing(nn.Module):
    "Implement label smoothing."

    def __init__(self, size=0, padding_idx=0, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(size_average=False, reduce=False)
        # self.criterion = nn.CrossEntropyLoss()
        # self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        # self.size = size
        self.true_dist = None

    def forward(self, input, target, mask):
        if target.ndim == 3:
            target = target.reshape(-1, target.shape[2])
            mask = mask.reshape(-1, mask.shape[2])
        # truncate to the same size
        target = target[:, :input.size(1)]
        mask = mask[:, :input.size(1)]

        input = input.reshape(-1, input.size(-1))
        target = target.reshape(-1)
        mask = mask.reshape(-1).to(input)

        # assert x.size(1) == self.size
        self.size = input.size(1)
        # true_dist = x.data.clone()
        true_dist = input.data.clone()
        # true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.fill_(self.smoothing / (self.size - 1))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        # true_dist[:, self.padding_idx] = 0
        # mask = torch.nonzero(target.data == self.padding_idx)
        # self.true_dist = true_dist
        return (self.criterion(input, true_dist).sum(1) * mask).sum() / mask.sum()


def compute_loss(output, reports_ids, reports_masks, labels=None, vis_label=None, txt_label=None,
                 z_img=None, z_txt=None, args={}, similarity_function='dot'):
    criterion = LanguageModelCriterion()
    # 第一个参数output[:, :-1]是模型预测的结果(去掉了结束符)，维度是(16,59,761)
    # 第二个参数reports_ids[:, 1:]是真实的报告文本(去掉了开始符)，维度是(16,59)
    # 第三个参数reports_masks[:, 1:]是真实的报告文本的mask(去掉了开始符)，维度是(16,59)
    loss = criterion(output[:, :-1], reports_ids[:, 1:], reports_masks[:, 1:]).mean()

    label_loss, match_loss = 0, 0
    if args.label_loss:
        label_criterion = torch.nn.BCEWithLogitsLoss() # 二分类交叉熵损失对象
        label_loss = label_criterion(vis_label, labels) # 传入预测值和真实值，计算损失
    if args.rank_loss:
        ranking_loss = RankingLoss()
        match_loss = ranking_loss(z_img, z_txt, labels, similarity_function) # 计算视觉-文本对齐损失
    return loss + 0.1 * label_loss + 0.1 * match_loss


class RankingLoss(nn.Module):
    def __init__(self):
        super(RankingLoss, self).__init__()

    def forward(self, z_image, z_text, labels, similarity_function='dot'):
        # z_image: (batch_size, 512):综合了(即求平均)两张图片的平均池化特征进行线性变换之后的结果
        # z_text: (batch_size, 512):文本特征的汇总，即开始符这个位置的特征
        # labels: (batch_size, 761):真实的标签
        # similarity_function: 'dot' or 'cosine' or 'l2'，即点积、余弦相似度或L2距离

        return self.imposter_img_loss(z_image, z_text, labels, similarity_function) + \
               self.imposter_txt_loss(z_image, z_text, labels, similarity_function)

    def imposter_img_loss(self, z_image, z_text, labels, similarity_function):
        """
        A custom loss function for computing the hinge difference
        between the similarity of an image-text pair and
        the similarity of an imposter image-text pair
        where the image is an imposter image chosen from the batch
        自定义损失函数，用于计算图像-文本对的相似度与冒名图像-文本对的相似度之间的铰链差，其中图像是从批次中选择的冒名图像
        """
        loss = torch.zeros(1, device=z_image.device, requires_grad=True)
        batch_size = z_image.size(0)

        for i in range(batch_size):
            # 对于每个样本，选择一个冒名图像，并计算最大边界，该边界基于图像标签之间的差异
            # Select an imposter image index and
            # compute the maximum margin based on the image label difference
            j = i + 1 if i < batch_size - 1 else 0
            if torch.equal(labels[i], labels[j]):
                # This means the imposter image comes from the same acquisition
                # 这里的margin就是所谓的image label difference,对应论文公式(14)中的μ
                # 如果两个图像的标签相同，则边界为0
                margin = 0
            else:
                # labels[i].int() | labels[j].int()：按位或，即两个标签中有一个为1，结果就为1
                # 然后求和，并使用item()方法将结果转换为Python标量，即获取了求和的结果
                n = (labels[i].int() | labels[j].int()).sum().item() # 或运算，再求和;用于对diff进行平均;与论文中的N_L稍有不同
                # labels[i].int() ^ labels[j].int()：按位异或，相同为0，不同为1
                diff = (labels[i].int() ^ labels[j].int()).sum().item() # 异或运算，再求和,对应论文公式(14)中的对应元素相减求绝对值再求和
                margin = max(0.5, diff / n)

            #  计算相似度，分别计算配对图像-文本对和冒名图像-文本对的相似度
            #  用冒名的图像替换配对的图像，与文本计算相似度
            if similarity_function == 'dot':
                # 计算点积相似度
                paired_similarity = torch.dot(z_image[i], z_text[i])
                imposter_similarity = torch.dot(z_image[j], z_text[i])
            elif similarity_function == 'cosine':
                # torch.norm:计算范数，即向量的模
                paired_similarity = \
                    torch.dot(z_image[i], z_text[i]) / (torch.norm(z_image[i]) * torch.norm(z_text[i]))
                imposter_similarity = \
                    torch.dot(z_image[j], z_text[i]) / (torch.norm(z_image[j]) * torch.norm(z_text[i]))
            elif similarity_function == 'l2':
                paired_similarity = -1 * torch.norm(z_image[i] - z_text[i])
                imposter_similarity = -1 * torch.norm(z_image[j] - z_text[i])

            # 最终的损失是越小越好，因此加上imposter_similarity，减去paired_similarity，这样就是最大化paired_similarity了
            # 这里直接用相似度值构建损失，与论文公式(12)不同，公式(12)是用1-相似度作为距离构建损失的，但是效果应该是一样的
            diff_similarity = imposter_similarity - paired_similarity + margin
            if diff_similarity > 0:
                loss = loss + diff_similarity

        return loss / batch_size  # 'mean' reduction

    def imposter_txt_loss(self, z_image, z_text, labels, similarity_function):
        """
        A custom loss function for computing the hinge difference
        between the similarity of an image-text pair and
        the similarity of an imposter image-text pair
        where the text is an imposter text chosen from the batch
        """
        loss = torch.zeros(1, device=z_image.device, requires_grad=True)
        batch_size = z_image.size(0)

        for i in range(batch_size):
            # Select an imposter text index and
            # compute the maximum margin based on the image label difference
            j = i + 1 if i < batch_size - 1 else 0
            if torch.equal(labels[i], labels[j]):
                # This means the imposter image comes from the same acquisition
                margin = 0
            else:
                n = (labels[i].int() | labels[j].int()).sum().item()
                diff = (labels[i].int() ^ labels[j].int()).sum().item()
                margin = max(0.5, diff / n)

            if similarity_function == 'dot':
                paired_similarity = torch.dot(z_image[i], z_text[i])
                imposter_similarity = torch.dot(z_text[j], z_image[i])
            elif similarity_function == 'cosine':
                paired_similarity = \
                    torch.dot(z_image[i], z_text[i]) / (torch.norm(z_image[i]) * torch.norm(z_text[i]))
                imposter_similarity = \
                    torch.dot(z_text[j], z_image[i]) / (torch.norm(z_text[j]) * torch.norm(z_image[i]))
            elif similarity_function == 'l2':
                paired_similarity = -1 * torch.norm(z_image[i] - z_text[i])
                imposter_similarity = -1 * torch.norm(z_text[j] - z_image[i])

            diff_similarity = imposter_similarity - paired_similarity + margin
            if diff_similarity > 0:
                loss = loss + diff_similarity

        return loss / batch_size  # 'mean' reduction
