import logging

import torch


def build_optimizer(args, model):
    # id()函数用于获取对象的内存地址
    ve_params = list(map(id, model.visual_extractor.parameters())) # 获取视觉特征提取器的参数所在的内存地址
    ed_params = filter(lambda x: id(x) not in ve_params, model.parameters()) # 将模型中除了视觉特征提取器的参数外的其他参数放入ed_params中
    optimizer = getattr(torch.optim, args.optim)(
        [{'params': model.visual_extractor.parameters(), 'lr': args.lr_ve},
         {'params': ed_params, 'lr': args.lr_ed}],
        weight_decay=args.weight_decay,
        amsgrad=args.amsgrad
    )
    # print("build_optimizer completed")
    logging.info(f"build_optimizer completed")
    return optimizer


def build_lr_scheduler(args, optimizer):
    if args.lr_scheduler == "StepLR":
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.step_size, args.gamma) # 在每个step_size个epoch之后将学习率乘以gamma
    elif args.lr_scheduler == "cosine":
        print(f"Using CosineAnnealingWarmRestarts lr_scheduler, T_0={args.step_size}, T_mult={args.gamma}")
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,
                                                                            T_0=args.step_size,
                                                                            T_mult=int(args.gamma)
                                                                            )
    else:
        raise NotImplementedError

    # print("build_lr_scheduler completed")
    logging.info(f"build_lr_scheduler completed")
    return lr_scheduler
