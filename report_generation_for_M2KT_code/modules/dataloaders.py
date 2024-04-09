import torch
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader
from .datasets import IuxrayMultiImageDataset, MimiccxrSingleImageDataset, CovidSingleImageDataset, CovidAllImageDataset


class LADataLoader(DataLoader):
    def __init__(self, args, tokenizer, split, shuffle):
        self.args = args
        self.dataset_name = args.dataset_name
        self.batch_size = args.batch_size
        self.shuffle = shuffle
        self.num_workers = args.num_workers
        self.tokenizer = tokenizer
        self.split = split
        # 定义了一个 normalize 变量，它是一个 torchvision.transforms.Normalize 类的实例。
        # 这个实例将输入的张量进行标准化，使得每个通道的均值为 0.5，标准差为 0.275。
        # 由于读取的时候是使用RGB模式进行的读取，因此标准化操作的平均值和标准差的列表长度是3
        # 通常用于对图像数据进行预处理
        normalize = transforms.Normalize(mean=[0.500, 0.500, 0.500],
                                         std=[0.275, 0.275, 0.275])
        # 训练集和其他两个数据集的预处理方式不同
        if split == 'train':
            self.transform = transforms.Compose([
                # image_size为默认值256.
                # 文献中说的图像大小被变为224x224应该是参数crop_size控制的
                transforms.Resize(args.image_size), # 将图像大小变为256
                transforms.RandomCrop(args.crop_size), # 随机裁剪图像到args.crop_size大小，这里是224
                transforms.RandomHorizontalFlip(), # 以0.5的概率水平翻转图像
                transforms.RandomAffine(degrees=0, translate=(0.2, 0.2), scale=(0.8, 0.8), fillcolor=(0, 0, 0)), # 随机仿射变换，包括旋转、平移和缩放
                transforms.ToTensor(),
                normalize])
        else:
            self.transform = transforms.Compose([
                transforms.Resize(args.image_size),
                transforms.CenterCrop(args.crop_size),
                transforms.ToTensor(),
                normalize])

        if self.dataset_name == 'iu_xray':
            self.dataset = IuxrayMultiImageDataset(self.args, self.tokenizer, self.split, transform=self.transform)
        elif self.dataset_name == 'covid':
            self.dataset = CovidSingleImageDataset(self.args, self.tokenizer, self.split, transform=self.transform)
        elif self.dataset_name == 'covidall':
            self.dataset = CovidAllImageDataset(self.args, self.tokenizer, self.split, transform=self.transform)
        else:
            self.dataset = MimiccxrSingleImageDataset(self.args, self.tokenizer, self.split, transform=self.transform)

        # print("这是一处调试标记")
        self.init_kwargs = {
            'dataset': self.dataset,
            'batch_size': self.batch_size,
            'shuffle': self.shuffle,
            'collate_fn': self.collate_fn,
            'num_workers': self.num_workers, # 用于指定数据加载器使用的子进程数量
            'pin_memory': True # 设置为True，则会将数据复制到固定的内存区域（锁页内存），这样可以加速数据传输
        }
        # print("这是一处调试标记")
        # 这里在最后才调用了父类的初始化函数，原因应该是：父类的初始化函数需要用到上面定义的一些变量，并且这个子类没有额外的自己的属性
        super().__init__(**self.init_kwargs) # 调用父类 DataLoader 的构造函数，以创建一个数据加载器对象

    @staticmethod
    def collate_fn(data):
        # 用来处理DataLoader返回的每个batch的数据的
        images_id, images, reports_ids, reports_masks, seq_lengths, labels = zip(*data) # 将data中的每个元素按照索引进行解包
        images = torch.stack(images, 0) # 将这一批图像数据堆叠在一起，这里是在第0维度上进行堆叠，因此又新增了一个维度
        max_seq_length = max(seq_lengths) # 获取这一批数据中最长的报告长度

        targets = np.zeros((len(reports_ids), max_seq_length), dtype=int) # 这里还只是ndarray
        targets_masks = np.zeros((len(reports_ids), max_seq_length), dtype=int) # 这里还只是ndarray
        print("这是一处调试标记")

        for i, report_ids in enumerate(reports_ids):
            # targets每一行都是固定的50个元素，而report_ids的长度不一定是50，因此需要将report_ids中的元素填充到targets中，如果report_ids的长度小于50，则其余元素默认为0
            targets[i, :len(report_ids)] = report_ids

        for i, report_masks in enumerate(reports_masks):
            # 情况类似于上面的targets
            targets_masks[i, :len(report_masks)] = report_masks

        labels = torch.stack(labels, 0)

        # targets是目标类别，只在最后用于比较预测结果和真实结果是否一致，因此弄成整型就可以了
        # targets_masks是用于掩盖无效的报告单词的，需要参与模型计算，因此弄成浮点型
        # 另外，经过LongTensor和FloatTensor处理后，targets和targets_masks都是tensor类型的了
        return images_id, images, torch.LongTensor(targets), torch.FloatTensor(targets_masks), labels

