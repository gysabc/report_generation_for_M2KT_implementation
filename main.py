import warnings

warnings.simplefilter("ignore", UserWarning)

import logging
import torch
import numpy as np
from modules.tokenizers import Tokenizer
from modules.dataloaders import LADataLoader
from modules.metrics import compute_scores
from modules.optimizers import build_optimizer, build_lr_scheduler
from modules.trainer import Trainer
from modules.loss import compute_loss
import models
from config import opts


def main():
    # parse arguments
    # args = parse_agrs()
    args = opts.parse_opt()
    logging.info(str(args))

    # fix random seeds
    torch.manual_seed(args.seed)  # 设置PyTorch的随机数生成器种子
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)  # 设置numpy的随机数生成器种子

    # create tokenizer
    tokenizer = Tokenizer(args)

    # create data loader
    train_dataloader = LADataLoader(args, tokenizer, split='train', shuffle=True)
    val_dataloader = LADataLoader(args, tokenizer, split='val', shuffle=False)
    test_dataloader = LADataLoader(args, tokenizer, split='test', shuffle=False)

    # build model architecture
    model_name = f"LAMRGModel_v{args.version}"
    logging.info(f"Model name: {model_name} \tModel Layers:{args.num_layers}")
    # 使用了Python内置的getattr()函数，返回了models模块中的model_name属性，即模型名称对应的模型类
    # 所以简单说就是：创建对应的模型对象，并传入用于初始化模型的参数
    model = getattr(models, model_name)(args, tokenizer)
    # get function handles of loss and metrics
    criterion = compute_loss # 使用criterion变量就相当于直接调用compute_loss函数，可以传递相同的参数并得到相同的返回值
    metrics = compute_scores # 同上

    # build optimizer, learning rate scheduler
    optimizer = build_optimizer(args, model)
    lr_scheduler = build_lr_scheduler(args, optimizer)

    # build trainer and start to train
    trainer = Trainer(model, criterion, metrics, optimizer, args, lr_scheduler, train_dataloader, val_dataloader,
                      test_dataloader)
    trainer.train()
    logging.info(str(args))


if __name__ == '__main__':
    main()
