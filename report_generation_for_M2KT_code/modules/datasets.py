import os
import json
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


class BaseDataset(Dataset):
    def __init__(self, args, tokenizer, split, transform=None):
        self.image_dir = args.image_dir
        self.ann_path = args.ann_path
        self.max_seq_length = args.max_seq_length
        self.split = split
        self.tokenizer = tokenizer
        self.transform = transform # 变换序列
        self.ann = json.loads(open(self.ann_path, 'r').read())
        # 读取的是某一个划分的样本(即训练集或者验证集或者测试集)
        self.examples = self.ann[self.split]
        for i in range(len(self.examples)):
            # 在原先的self.examples的每个元素中增加一个键值对,用于存放当前id的报告对应的词汇表的索引序列
            self.examples[i]['ids'] = tokenizer(self.examples[i]['report'])[:self.max_seq_length] # 对返回的索引序列按照最大序列长度进行了截断,如果不够长则不会截断也不会填充
            self.examples[i]['mask'] = [1] * len(self.examples[i]['ids']) # 掩码向量，可能是用来对输入的数据进行填充判断的？

    def __len__(self):
        return len(self.examples)


class IuxrayMultiImageDataset(BaseDataset):
    def __init__(self, args, tokenizer, split, transform=None):
        super(IuxrayMultiImageDataset, self).__init__(args, tokenizer, split, transform)
        # 读取标签这里并没有指定split，因此是将所有的报告的标签都读取进来了
        # 也说明标签集里面包含了三类数据集的所有内容
        self.label = self._load_data(args.label_path)

    def _load_data(self, label_path):
        # 字典中的键是id，值是一个长度为14的列表，列表中的元素是0或1
        label_dict = {}

        data = pd.read_csv(label_path)
        for index, row in data.iterrows():
            idx = row['id']
            label = row[1:].tolist()
            # map()将一个函数应用于一个可迭代对象的所有元素，并返回一个新的可迭代对象
            # 如果对于某个观察结果是1，则当前id对应的这个观察结果的标签值就是1，其他的情况(包括不确定、0、空值)都是0
            label_dict[idx] = list(map(lambda x: 1 if x == 1.0 else 0, label))

        return label_dict

    def __getitem__(self, idx):
        # __getitem__是Python中的一个魔法方法，用于实现对象的索引操作。
        # 当你使用方括号访问对象的元素时，例如obj[index],Python会自动调用该对象的__getitem__方法，并将index作为参数传递给它
        # 这个方法是在pytorch从训练集中fetch数据的时候会调用
        example = self.examples[idx] # 从self.examples中取出一个样本。
        image_id = example['id'] # 即当前报告对应的患者id
        image_path = example['image_path'] # 获取图像的路径(部分患者文件夹中包含2张以上的图像)
        image_1 = Image.open(os.path.join(self.image_dir, image_path[0])).convert('RGB') # 将图像的颜色模式转换为RGB
        image_2 = Image.open(os.path.join(self.image_dir, image_path[1])).convert('RGB')
        if self.transform is not None:
            image_1 = self.transform(image_1)
            image_2 = self.transform(image_2)
        image = torch.stack((image_1, image_2), 0) # 将两张图像沿着新的维度堆叠在一起，这里是在第0维度上进行堆叠
        report_ids = example['ids']
        report_masks = example['mask']
        seq_length = len(report_ids)

        # pid = image_id.split('_')[0][3:] # 这里不需要了
        # try:
        #     labels = torch.tensor(self.label[int(pid)], dtype=torch.float32)
        # except:
        #     # print('Except id ', pid)
        #     labels = torch.tensor([0 for _ in range(14)], dtype=torch.float32)
        try:
            labels = torch.tensor(self.label[image_id], dtype=torch.float32)
        except:
            # print('Except id ', pid)
            labels = torch.tensor([0 for _ in range(14)], dtype=torch.float32)

        sample = (image_id, image, report_ids, report_masks, seq_length, labels)
        return sample


class MimiccxrSingleImageDataset(BaseDataset):
    def __init__(self, args, tokenizer, split, transform=None):
        super(MimiccxrSingleImageDataset, self).__init__(args, tokenizer, split, transform)
        self.label = pd.read_csv(args.label_path)


    def __getitem__(self, idx):
        example = self.examples[idx]
        image_id = example['id']
        image_path = example['image_path']
        image = Image.open(os.path.join(self.image_dir, image_path[0])).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        report_ids = example['ids']
        report_masks = example['mask']
        seq_length = len(report_ids)

        d = self.label[self.label['dicom_id'] == image_id]
        labels = torch.tensor(d.values.tolist()[0][8:], dtype=torch.float32)
        sample = (image_id, image, report_ids, report_masks, seq_length, labels)
        return sample


class CovidSingleImageDataset(BaseDataset):
    def __init__(self, args, tokenizer, split, transform=None):
        super(CovidSingleImageDataset, self).__init__(args, tokenizer, split, transform)
        self.label = self._load_data(args.label_path)

    def _load_data(self, label_file):
        labels = {}

        # print(f"Loading data from {label_file}")

        data = pd.read_csv(label_file)
        # data = data[data['split'] == self.subset]
        for index, row in data.iterrows():
            idx = row['idx']
            label = [1, 0] if row['label'] == '轻型' else [0, 1]
            labels[idx] = label

        return labels

    def __getitem__(self, idx):
        example = self.examples[idx]
        image_id = example['id']
        image_path = example['image_path']
        image = Image.open(os.path.join(self.image_dir, image_path[0])).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        report_ids = example['ids']
        report_masks = example['mask']
        seq_length = len(report_ids)

        labels = torch.tensor(self.label[image_id], dtype=torch.float32)
        sample = (image_id, image, report_ids, report_masks, seq_length, labels)
        return sample


class CovidAllImageDataset(BaseDataset):
    def __getitem__(self, idx):
        example = self.examples[idx]
        image_id = example['id']
        image_path = example['image_path']
        image = Image.open(os.path.join(self.image_dir, image_path[0])).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        report_ids = example['ids']
        report_masks = example['mask']
        seq_length = len(report_ids)

        labels = torch.tensor(example['label'], dtype=torch.float32)
        sample = (image_id, image, report_ids, report_masks, seq_length, labels)
        return sample